"""``lds/v1`` reference systems and their SSM implementations."""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.ssm import DiagonalSSM, DirectLDS  # noqa: E402
from latent_machines.machines import lds  # noqa: E402


def max_error(outputs, expected) -> float:
    return (
        (outputs.double() - torch.tensor(expected, dtype=torch.float64))
        .abs()
        .max()
        .item()
    )


def test_reference_impulse_response_of_the_oscillator() -> None:
    system = lds.damped_oscillator(radius=0.5, period=4.0)  # rotation by 90 degrees
    outputs = lds.reference(system, [1.0, 0.0, 0.0, 0.0, 0.0])
    # y_t = C A^(t-1) B: x rotates (1,0) -> (0,0.5) -> (-0.25,0) -> (0,-0.125)
    assert outputs == pytest.approx([0.0, 1.0, 0.0, -0.25, 0.0], abs=1e-15)


def test_butterworth_has_unit_dc_gain() -> None:
    outputs = lds.reference(lds.butterworth_lowpass(0.05), lds.step(400))
    assert outputs[-1] == pytest.approx(1.0, abs=1e-9)


def test_random_stable_system_is_stable() -> None:
    system = lds.random_stable(8)
    radius = torch.linalg.eigvals(torch.tensor(system.a)).abs().max().item()
    assert 0.9 <= radius <= 0.98 + 1e-12


@pytest.mark.parametrize("name", list(lds.SYSTEMS))
def test_float64_implementations_match_reference(name: str) -> None:
    system = lds.SYSTEMS[name]()
    inputs = lds.noise(500, 1)
    expected = lds.reference(system, inputs)
    signal = torch.tensor(inputs, dtype=torch.float64)
    for backend in (
        DirectLDS(system, torch.float64),
        DiagonalSSM(system, torch.float64, "sequential"),
        DiagonalSSM(system, torch.float64, "chunked", chunk=48),  # 500 % 48 != 0
    ):
        assert max_error(backend.run(signal), expected) < 1e-10


@pytest.mark.parametrize("backend", ["direct", "sequential", "chunked"])
def test_snapshot_resume_is_bit_identical(backend: str) -> None:
    system = lds.random_stable(4)
    signal = torch.tensor(lds.noise(256, 2), dtype=torch.float64)

    def make():
        if backend == "direct":
            return DirectLDS(system, torch.bfloat16)
        return DiagonalSSM(system, torch.bfloat16, backend, chunk=32)

    whole = make().run(signal)
    first = make()
    head = first.run(signal[:128])
    resumed = make()
    resumed.restore(json.loads(json.dumps(first.snapshot())))
    assert torch.equal(torch.cat([head, resumed.run(signal[128:])]), whole)


def test_bfloat16_rounds_the_slow_pole_away_from_its_radius() -> None:
    system = lds.damped_oscillator(radius=0.999)
    ssm = DiagonalSSM(system, torch.bfloat16)
    assert math.isclose(DiagonalSSM(system, torch.float64).cast_spectral_radius, 0.999)
    assert abs(ssm.cast_spectral_radius - 0.999) > 1e-4


def test_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError):
        DiagonalSSM(lds.damped_oscillator(), torch.float32, "parallel")
