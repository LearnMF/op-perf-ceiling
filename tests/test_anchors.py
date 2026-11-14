"""Anchor-kernel experiment on the PV SOC simulation sheet (opceil.model.anchors)."""
from opceil.model import anchors


def test_three_gemm_anchors_predict_the_family_on_an_unknown_card():
    w = anchors.exp_within(anchors.load())
    for card, v in w.items():
        assert v["combos"] >= 10
        assert v["median_of_median_pct"] < 15.0, (card, v["median_of_median_pct"])


def test_gemm_anchors_do_not_carry_to_other_families():
    x = anchors.exp_cross(anchors.load())
    best = min(e["median_pct"] for e in x["dsa"]["by_ratio"].values())
    assert best > 15.0                 # cross-family: no GEMM-derived ratio gets DSA within 15%
