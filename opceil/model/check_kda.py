"""KDA ceilings against an optimised implementation (bench/kda_bench.py output: flash-train mk3 on the card).

For every (T, H) the KDA plugin is priced on the machine file and compared with the measured fwd+bwd device time:

  T_bound / T_theory      fused-operator floors (provable / with model and calibrated terms)
  T_split p82             the production decomposition (8 kernels, forward saves h -- what mk3 does for H <= 64)
  per kernel              each p82 kernel against the mk3 kernels that implement it (MAP below), which is where
                          the gap sits: streaming kernels vs the ones dominated by small triangular products / scans
  calibrated transfer     eta = T_split / measured per kernel, taken from ANOTHER card's measurements (or from the
                          other shapes of the same card), applied to this card's T_split: the error a calibrated
                          prediction has on hardware it was not calibrated on.
"""
from __future__ import annotations

import json
import statistics

from .predict import TheoryMachine, predict
from .validate import _spearman

TOL = 1.01
# p82 kernel -> substrings of the flash-train mk3 kernel names that implement it
MAP = {"fwd_prep": ("fwd_prep_a", "fwd_prep_b", "kda_gn_all"), "fwd_h_o": ("fwd_h_",), "head_gate": ("gate_kernel",),
       "wu": ("wu_kernel",), "dav": ("dav_kernel",), "dhu_bn": ("dhu_",), "wy_s4_split": ("wy_s4a", "wy_s4b"),
       "bwd_tail": ("bwd_tail_kernel", "bwd_tail_reduce")}


def load(path: str) -> tuple:
    meta, rows = {}, []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
        else:
            rows.append(d)
    return meta, rows


def rows_for(mach: TheoryMachine, rows: list) -> list:
    out = []
    for d in rows:
        p = predict(mach, "kda", {"T": d["T"], "H": d["H"]}, "bf16", "fwd+bwd")
        sp = p["T_split"]["p82"]
        pk = {k["kernel"]: k["T_us"] for k in sp["kernels"]}
        mk = {n: sum(v for x, v in d["tot_kernels"].items() if any(s in x for s in pats)) for n, pats in MAP.items()}
        out.append({"T": d["T"], "H": d["H"], "meas_us": d["tot_us_kernel"], "T_bound_us": p["T_bound_us"],
                    "T_theory_us": p["T_theory_us"], "T_split_us": sp["T_split_us"], "pred_k": pk, "meas_k": mk,
                    "unmapped_us": d["tot_us_kernel"] - sum(mk.values())})
    return out


def _err(pred: list, meas: list) -> dict:
    e = [abs(a - b) / b for a, b in zip(pred, meas)]
    return {"median_pct": 100 * statistics.median(e), "mean_pct": 100 * statistics.mean(e), "max_pct": 100 * max(e),
            "within_10pct": sum(x <= .10 for x in e), "n": len(e), "spearman": _spearman(pred, meas)}


def calibrated(target: list, source: list, leave_one_out: bool = False) -> list:
    """Per-kernel eta from `source` applied to `target`'s p82 kernels (unmapped glue kernels taken as measured)."""
    pred = []
    for i, r in enumerate(target):
        src = [x for j, x in enumerate(source) if not (leave_one_out and j == i)]
        eta = {n: statistics.median(x["pred_k"][n] / x["meas_k"][n] for x in src) for n in MAP}
        pred.append(sum(r["pred_k"][n] / eta[n] for n in MAP) + r["unmapped_us"])
    return pred


def check(mach: TheoryMachine, rows: list, other: list | None = None) -> dict:
    rs = rows_for(mach, rows)
    meas = [r["meas_us"] for r in rs]
    ratio = lambda key: sorted(r[key] / r["meas_us"] for r in rs)
    out = {"schema": "opceil.check_kda/1", "machine": mach.id, "n": len(rs), "rows": rs,
           "violations": {k: [(r["T"], r["H"]) for r in rs if r[k] > r["meas_us"] * TOL]
                          for k in ("T_bound_us", "T_theory_us", "T_split_us")},
           "ratio_median": {k: statistics.median(ratio(k)) for k in ("T_bound_us", "T_theory_us", "T_split_us")},
           "per_kernel": {n: statistics.median(r["pred_k"][n] / r["meas_k"][n] for r in rs) for n in MAP},
           "error": {"raw T_split": _err([r["T_split_us"] for r in rs], meas),
                     "same card, per-kernel eta, leave one shape out": _err(calibrated(rs, rs, True), meas)}}
    if other:
        out["error"]["per-kernel eta from the other card"] = _err(calibrated(rs, other), meas)
    return out


def markdown(r: dict, meta: dict) -> str:
    L = ["# opceil check: KDA ceilings vs %s (%s, flash-train %s)" % (
             meta.get("device", "?"), meta.get("host", "?"), meta.get("flash_train_commit", "?")), "",
         "machine file `%s`; %d shapes, fwd+bwd device time; T_split = production decomposition p82." % (r["machine"], r["n"]), "",
         "| | median predicted / measured | above measured |", "| --- | ---: | ---: |"]
    for k, lab in (("T_bound_us", "T_bound"), ("T_theory_us", "T_theory"), ("T_split_us", "T_split p82")):
        L.append("| %s | %.3f | %d |" % (lab, r["ratio_median"][k], len(r["violations"][k])))
    L += ["", "## prediction error of the measured time", "", "| predictor | median | mean | max | within 10% | Spearman |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, e in r["error"].items():
        L.append("| %s | %.1f%% | %.1f%% | %.1f%% | %d / %d | %.3f |" % (k, e["median_pct"], e["mean_pct"], e["max_pct"],
                                                                    e["within_10pct"], e["n"], e["spearman"]))
    L += ["", "## per kernel (p82 kernel vs the mk3 kernels implementing it)", "", "| p82 kernel | mk3 kernels | median predicted / measured |",
          "| --- | --- | ---: |"]
    L += ["| %s | %s | %.2f |" % (n, " + ".join(MAP[n]), v) for n, v in r["per_kernel"].items()]
    L += ["", "## all shapes", "", "| T | H | measured us | T_bound | T_theory | T_split p82 | p82 / measured |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    L += ["| %d | %d | %.0f | %.0f | %.0f | %.0f | %.3f |" % (x["T"], x["H"], x["meas_us"], x["T_bound_us"], x["T_theory_us"],
                                                         x["T_split_us"], x["T_split_us"] / x["meas_us"]) for x in r["rows"]]
    return "\n".join(L) + "\n"
