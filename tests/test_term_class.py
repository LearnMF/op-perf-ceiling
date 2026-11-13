"""Every term says how far it can be trusted, and T_bound keeps only the provable ones."""
from pathlib import Path

import pytest

from opceil.model.predict import TERM_CLASSES, TheoryMachine, _terms, predict
from opceil.ops import get

ROOT = Path(__file__).resolve().parent.parent
MACHINES = [ROOT / "machines" / "gfx936-bw1000.json", ROOT / "machines" / "gfx938-bw1102.json"]
CASES = [
    ("gemm", {"M": 8192, "N": 4096, "K": 6144}, "fwd+bwd"),
    ("gemm", {"M": 1, "N": 256, "K": 6144}, "fwd"),
    ("rmsnorm", {"elements": 2 ** 26}, "fwd+bwd"),
    ("softmax", {"elements": 2 ** 24}, "fwd"),
    ("kda", {"T": 16384, "H": 12}, "fwd+bwd"),
    ("kda", {"T": 8192, "H": 48}, "fwd"),
]


@pytest.mark.parametrize("path", MACHINES, ids=lambda p: p.stem)
@pytest.mark.parametrize("peaks", ["measured", "theoretical"])
@pytest.mark.parametrize("op,shape,phase", CASES)
def test_bound_never_exceeds_theory(path, peaks, op, shape, phase):
    r = predict(TheoryMachine.load(str(path), peaks), op, shape, "bf16", phase)
    assert 0 < r["T_bound_us"] <= r["T_theory_us"] * (1 + 1e-12)
    assert set(r["term_class"].values()) <= set(TERM_CLASSES)
    for sp in r["T_split"].values():
        assert r["T_bound_us"] <= sp["T_split_us"]


@pytest.mark.parametrize("op,shape,phase", CASES)
def test_strict_terms_are_all_provable(op, shape, phase):
    m = TheoryMachine.load(str(MACHINES[0]))
    w = get(op).work(shape, "bf16", phase)
    key, peak = m.compute_peak("bf16")
    for _, pw in (w.parts or [(op, w)]):
        cls = {}
        _terms(m, pw, key, peak, strict=True, cls=cls)
        assert set(cls.values()) == {"bound"}, cls


def test_plain_gemm_is_all_provable():
    """No efficiency constant, no occupancy, no read/write split: T_bound is T_theory."""
    r = predict(TheoryMachine.load(str(MACHINES[0])), "gemm", {"M": 8192, "N": 4096, "K": 6144}, "bf16", "fwd")
    assert set(r["term_class"].values()) == {"bound"}
    assert r["T_bound_us"] == pytest.approx(r["T_theory_us"])
    assert r["T_theory_class"] == "bound"


def test_kda_serial_bottleneck_rests_on_a_calibrated_constant():
    """KDA 16K/TP8: the serial chain that binds T_theory uses the KDA-measured step latency; without it the
    provable floor is the HBM term, so the report must not present 'serial' as a provable bottleneck."""
    r = predict(TheoryMachine.load(str(MACHINES[0])), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert r["binding"] == "serial:chain" and r["term_class"]["serial:chain"] == "calibrated"
    assert r["T_theory_class"] == "calibrated"
    assert r["T_bound_binding"] == "memory:hbm"
    assert r["term_class"]["other:valu_lane"] == "model"
    assert r["term_class"]["compute:mmac_bf16"] == "calibrated"
    assert r["T_bound_us"] < 0.8 * r["T_theory_us"]


def test_split_search_labels_t_split_a_model_estimate():
    from opceil.model.capacity import solve
    b = solve(TheoryMachine.load(str(MACHINES[0])), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    c = b["ceiling"]
    assert c["T_split_class"] == "model"
    assert c["T_bound_us"] <= c["T_theory_us"] <= c["T_split_us"]
