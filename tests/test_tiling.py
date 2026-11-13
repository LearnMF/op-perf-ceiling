"""The tiled-GEMM model: feasible under the machine's capacity, never below the fused work, model class only."""
from pathlib import Path

import pytest

from opceil.model import tiling
from opceil.model.predict import TheoryMachine, predict

ROOT = Path(__file__).resolve().parent.parent
MACHINES = [ROOT / "machines" / "gfx936-bw1000.json", ROOT / "machines" / "gfx938-bw1102.json"]
SHAPES = [(1, 256, 6144), (16, 6144, 12288), (256, 4096, 4096), (512, 11008, 4096), (8192, 4096, 6144),
          (4096, 4096, 4096), (2048, 576, 4608)]


@pytest.mark.parametrize("path", MACHINES, ids=lambda p: p.stem)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_tiled_split_is_feasible_and_above_the_fused_work(path, shape):
    m = TheoryMachine.load(str(path))
    r = predict(m, "gemm", dict(zip("MNK", shape)), "bf16", "fwd")
    sp = r["T_split"]["tiled"]
    tl = sp["kernels"][0]["tiling"]
    assert tl["lds_kb"] * 1024 <= m.device["lds_per_cu"]
    assert tl["vgpr"] <= m.device.get("max_vgpr_per_thread", tiling.VGPR_MAX_DEFAULT)
    assert r["T_bound_us"] <= r["T_theory_us"] <= sp["T_split_us"] * (1 + 1e-9)
    cls = sp["kernels"][0]["terms_class"]
    assert cls["compute:tiles"] == "model" and cls.get("serial:kloop", "model") == "model"


def test_extra_terms_never_enter_the_bound():
    m = TheoryMachine.load(str(MACHINES[0]))
    key, peak = m.compute_peak("bf16")
    w, _ = tiling.price(m, 64, 256, 6144, 2, 2, key, peak, 16, 64, 32, 4, 64, 3)
    from opceil.model.predict import _terms
    cls = {}
    t = _terms(m, w, key, peak, strict=True, cls=cls)
    assert "compute:tiles" not in t and "serial:kloop" not in t
    assert set(cls.values()) == {"bound"}


def test_padding_and_small_grids_cost_compute():
    """M=1 computes whole 16-row tiles, and a 4-tile grid can keep 4 of the CUs busy."""
    m = TheoryMachine.load(str(MACHINES[0]))
    key, peak = m.compute_peak("bf16")
    w, fp = tiling.price(m, 1, 256, 4096, 2, 2, key, peak, 16, 64, 32, 1, 64, 2)
    assert fp["ctas"] == 4
    full = 2.0 * 16 * 256 * 4096 / peak
    assert w.extra_terms["compute:tiles"] == pytest.approx(full * m.cus / 4 / 0.25)


def test_tiling_can_be_switched_off():
    m = TheoryMachine(TheoryMachine.load(str(MACHINES[0])).raw, tiling=False)
    assert "tiled" not in predict(m, "gemm", {"M": 64, "N": 64, "K": 64}, "bf16", "fwd")["T_split"]
