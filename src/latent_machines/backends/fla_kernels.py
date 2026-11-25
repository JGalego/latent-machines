"""Fused Gated DeltaNet kernels from flash-linear-attention, driven as E00 memory.

Runs ``chunk_gated_delta_rule`` and ``fused_recurrent_gated_delta_rule``
(https://github.com/fla-org/flash-linear-attention, package ``fla``) on the
lowered ``memory/v1`` actions of ``delta_chunked.lower_sequence``, so their
output can be compared with the integer oracle like any other backend. The
package needs CUDA and Triton; it is imported when a kernel runs, so this
module imports without it.

Mapping to the kernels' interface, one head with ``K = capacity`` and
``V = word_bits``:

- ``q``, ``k``: one-hot addresses ``[1, T, 1, capacity]`` in the working dtype,
  all-zero for actions with no key or query. ``k`` is normalized outside the
  kernel (a one-hot vector is already unit length), or inside it with
  ``use_qk_l2norm_in_kernel`` as a separate variant.
- ``v``: bit planes ``[1, T, 1, word_bits]``; ``beta``: 1 for a write, else 0.
- ``g = 0`` (log of ``alpha = 1``) in float32; ``scale = 1`` so ``o = S q``.
- The state is float32 ``[1, 1, K, V]``, so it is the transpose of the
  ``(word_bits, capacity)`` matrix of ``DeltaMemory``.

A sequence runs in segments of ``segment`` actions. The state returned by one
call is the initial state of the next, which gives a state to decode and check
at each segment end and keeps a mismatch localized to one segment.
"""

from collections.abc import Iterator
from importlib import metadata

import torch

from latent_machines.backends.delta_chunked import Chunk, Lowered, dense_inputs
from latent_machines.machines.memory import Memory

KERNELS = ("fla-chunk", "fla-recurrent")
SEGMENT = 4096


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("flash-linear-attention", "fla-core", "triton", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_fla(
    machine: Memory,
    lowered: Lowered,
    kernel: str,
    dtype: torch.dtype,
    *,
    normalize_in_kernel: bool = False,
    segment: int = SEGMENT,
    device: str = "cuda",
) -> Iterator[Chunk]:
    """Run ``kernel`` over ``lowered`` in segments, yielding each segment."""
    if kernel == "fla-chunk":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as op
    elif kernel == "fla-recurrent":
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as op
    else:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    state = torch.zeros(
        1, 1, machine.capacity, machine.word_bits, dtype=torch.float32, device=device
    )
    for start in range(0, len(lowered), segment):
        stop = min(start + segment, len(lowered))
        keys, values, beta, queries = dense_inputs(
            machine, lowered, start, stop, dtype, device
        )
        count = stop - start
        outputs, state = op(
            q=queries[None, :, None, :],
            k=keys[None, :, None, :],
            v=values[None, :, None, :],
            g=torch.zeros(1, count, 1, dtype=torch.float32, device=device),
            beta=beta[None, :, None],
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=normalize_in_kernel,
        )
        yield Chunk(start, stop, outputs[0, :, 0, :], state[0, 0].T)
