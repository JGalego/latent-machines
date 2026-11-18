"""Tensor backends for ``cellular/v1``: convolution plus ReLU lookup, and int.

``ConvLookupCellular`` compiles a rule into layers that act on the whole
grid at once, in a floating-point format:

    halo  = pad(s)                       zero or circular, one cell wide
    n     = conv3x3(halo, K)             K = ones with a zero center
    h_c   = relu(n-c+1) - 2 relu(n-c) + relu(n-c-1)    c = 0..8 (one-hot n)
    alive = relu(s + h_c - 1)            live cell with c neighbors
    dead  = relu(h_c - s)                dead cell with c neighbors
    s'    = sum_c S[c] alive_c + B[c] dead_c

``S`` and ``B`` are the rule's survival and birth sets as 0/1 weights. With a
0/1 grid every intermediate is an integer from -8 to 9, so the circuit is
exact in float32, float64, bfloat16, and float16. ``residue`` measures the
stored grid's distance from {0, 1} before decoding.

``TorchIntCellular`` is the conventional vectorized baseline: integer
neighbor sums from shifted slices and a table lookup, with no floating
point.

Both read the population for the ``output`` event by thresholding the grid
and counting in int64; a sum in bfloat16 or float16 would round above 256
or 2048 live cells. Host code, written independently of the oracle,
validates ``Set`` before touching the grid. ``decode`` and ``encode`` exist
for conformance and snapshots only.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_machines.machines.cellular import (
    Advance,
    Cellular,
    CellularFault,
    Grid,
    Set,
)
from latent_machines.machines.events import Event, EventKind

PHASES = ("halo", "count", "lookup")


@dataclass(frozen=True)
class TensorGrid:
    cells: torch.Tensor  # (height, width), never modified in place
    generation: int


@dataclass(frozen=True)
class TensorTransition:
    next_state: TensorGrid
    event: Event
    work: int


class _GridBackend:
    """Shared ``Set`` handling, decoding, and phase timing."""

    dtype: torch.dtype

    def __init__(self, machine: Cellular, device: torch.device | str) -> None:
        self.machine, self.device = machine, device
        self.profile = False
        self.phase_seconds = dict.fromkeys(PHASES, 0.0)

    def initial_state(self) -> TensorGrid:
        return self.encode(self.machine.initial_state())

    def step(self, state: TensorGrid, action: Advance | Set) -> TensorTransition:
        machine = self.machine
        if tuple(state.cells.shape) != (machine.height, machine.width):
            raise ValueError(f"grid shape {tuple(state.cells.shape)} is wrong")
        if isinstance(action, Set):
            if not (
                0 <= action.row < machine.height and 0 <= action.col < machine.width
            ):
                return self._fault(state, CellularFault.CELL_OUT_OF_RANGE)
            if action.value not in (0, 1):
                return self._fault(state, CellularFault.VALUE_OUT_OF_RANGE)
            cells = state.cells.clone()
            cells[action.row, action.col] = action.value
            return TensorTransition(
                TensorGrid(cells, state.generation), Event(EventKind.CONTINUE), 1
            )
        if not isinstance(action, Advance):
            raise TypeError(f"not a cellular/v1 action: {action!r}")
        cells = self.advance(state.cells)
        population = int((cells > 0.5).sum(dtype=torch.int64))
        event = Event(EventKind.OUTPUT, value=population)
        return TensorTransition(
            TensorGrid(cells, state.generation + 1), event, machine.cells
        )

    def advance(self, cells: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _timed(self, phase: str, start: float) -> float:
        now = time.perf_counter()
        self.phase_seconds[phase] += now - start
        return now

    def decode(self, state: TensorGrid) -> Grid:
        """Canonical grid, for conformance only."""
        if state.cells.is_floating_point() and not torch.isfinite(state.cells).all():
            raise ValueError("non-finite cell")
        rows = (state.cells > 0.5).to(torch.int64).tolist()
        return Grid(tuple(tuple(row) for row in rows), state.generation)

    def encode(self, grid: Grid) -> TensorGrid:
        cells = torch.tensor(grid.cells, dtype=self.dtype, device=self.device)
        return TensorGrid(cells, grid.generation)

    def residue(self, state: TensorGrid) -> float:
        cells = state.cells.to(torch.float64)
        return torch.minimum(cells.abs(), (cells - 1).abs()).max().item()

    def _fault(self, state: TensorGrid, reason: CellularFault) -> TensorTransition:
        return TensorTransition(state, Event(EventKind.FAULT, reason=reason), 1)


class ConvLookupCellular(_GridBackend):
    """``cellular/v1`` as a 3x3 convolution followed by a ReLU lookup."""

    def __init__(
        self,
        machine: Cellular,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(machine, device)
        self.dtype = dtype
        kernel = torch.ones(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0
        counts = torch.arange(9.0)
        rule = machine.rule
        survive = torch.tensor([float(c in rule.survival) for c in range(9)])
        birth = torch.tensor([float(c in rule.birth) for c in range(9)])
        self.kernel, self.counts, self.survive, self.birth = (
            w.to(dtype=dtype, device=device) for w in (kernel, counts, survive, birth)
        )

    def circuit_stats(self) -> dict[str, int]:
        weights = (self.kernel, self.counts, self.survive, self.birth)
        return {
            "layers": 4,  # convolution, hat one-hot, alive/dead units, rule sum
            "hidden_units_per_cell": 3 * 9 + 2 * 9,
            "stored_weights": sum(w.numel() for w in weights),
            "nonzero_weights": sum(int(w.count_nonzero()) for w in weights),
        }

    def advance(self, cells: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        grid = cells[None, None]
        mode = "circular" if self.machine.boundary == "torus" else "constant"
        halo = F.pad(grid, (1, 1, 1, 1), mode=mode)
        if self.profile:
            start = self._timed("halo", start)
        n = F.conv2d(halo, self.kernel)[0, 0]
        if self.profile:
            start = self._timed("count", start)
        a = n[None] - self.counts[:, None, None]  # (9, height, width)
        hat = torch.relu(a + 1) - 2 * torch.relu(a) + torch.relu(a - 1)
        alive = torch.relu(cells[None] + hat - 1)
        dead = torch.relu(hat - cells[None])
        result = torch.einsum("c,chw->hw", self.survive, alive) + torch.einsum(
            "c,chw->hw", self.birth, dead
        )
        if self.profile:
            self._timed("lookup", start)
        return result


class TorchIntCellular(_GridBackend):
    """Conventional vectorized baseline: integer sums and a table lookup."""

    def __init__(self, machine: Cellular, device: torch.device | str = "cpu") -> None:
        super().__init__(machine, device)
        self.dtype = torch.uint8
        rule = machine.rule
        self.table = torch.tensor(
            [[int(c in rule.birth) for c in range(9)]]
            + [[int(c in rule.survival) for c in range(9)]],
            dtype=torch.uint8,
            device=device,
        )

    def advance(self, cells: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        height, width = cells.shape
        wide = cells.to(torch.int16)
        if self.machine.boundary == "torus":
            halo = torch.cat([wide[-1:], wide, wide[:1]])
            halo = torch.cat([halo[:, -1:], halo, halo[:, :1]], dim=1)
        else:
            halo = F.pad(wide, (1, 1, 1, 1))
        if self.profile:
            start = self._timed("halo", start)
        n = sum(
            halo[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
        if self.profile:
            start = self._timed("count", start)
        result = self.table[cells.long(), n.long()]
        if self.profile:
            self._timed("lookup", start)
        return result
