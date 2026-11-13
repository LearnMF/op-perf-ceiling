"""The bound must hold on real kernel-level measurements (bench/gemm_sweep.py output in data/measured/)."""
from pathlib import Path

import pytest

from opceil.model.check import check, load
from opceil.model.predict import TheoryMachine

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = sorted((ROOT / "data" / "measured").glob("gemm_*.jsonl"))


def _machine_for(path: Path) -> Path:
    card = path.stem.split("_")[2]                    # gemm_bf16_<machine id>_<date>[_tunableop]
    return ROOT / "machines" / (card + ".json")


@pytest.mark.parametrize("peaks", ["measured", "theoretical"])
@pytest.mark.parametrize("sweep", SWEEPS, ids=lambda p: p.stem)
def test_gemm_bound_holds_on_every_measured_shape(sweep, peaks):
    meta, rows = load(str(sweep))
    r = check(TheoryMachine.load(str(_machine_for(sweep)), peaks), rows)
    assert r["n"] >= 100
    assert not r["violations"], [(x["M"], x["N"], x["K"]) for x in r["violations"]]
    # ranking agreement; the TunableOp subsets cover a narrower size range (<= 50 GFLOP), hence 0.9 not 0.95
    assert r["spearman_bound_vs_meas"] > 0.9
    # the tiled model must not claim more cost than a real library kernel pays
    assert not r["split_violations"], [(x["M"], x["N"], x["K"]) for x in r["split_violations"]]


def test_theoretical_peak_is_never_below_measured():
    import json
    for f in (ROOT / "machines").glob("*.json"):
        for k, v in json.loads(f.read_text())["peaks"].items():
            if isinstance(v, dict) and v.get("theoretical") is not None and v.get("value") is not None:
                if v.get("unit", "").startswith(("GB/s", "TFLOPS", "TOPS")):
                    assert v["theoretical"] >= v["value"], (f.name, k)
