"""KDA ceilings must hold against the optimised flash-train mk3 kernels measured on each card (data/measured/kda_*)."""
from pathlib import Path

import pytest

from opceil.model import check_kda
from opceil.model.predict import TheoryMachine

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = sorted((ROOT / "data" / "measured").glob("kda_*.jsonl"))


@pytest.mark.parametrize("sweep", SWEEPS, ids=lambda p: p.stem)
def test_kda_floors_never_above_the_optimised_kernels(sweep):
    card = sweep.stem.split("_")[2]                  # kda_bf16_<machine id>_<date>_<build>
    meta, rows = check_kda.load(str(sweep))
    r = check_kda.check(TheoryMachine.load(str(ROOT / "machines" / (card + ".json"))), rows)
    assert r["n"] >= 20
    for k, v in r["violations"].items():
        assert not v, (k, v)
    assert r["error"]["raw T_split"]["spearman"] > 0.95
    # every p82 kernel is mapped onto the mk3 kernels that implement it (no device time left unaccounted)
    assert all(abs(x["unmapped_us"]) < 0.02 * x["meas_us"] for x in r["rows"])
