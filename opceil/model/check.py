"""`python -m opceil check --machine M --measured gemm.jsonl`: bounds against kernel-level measurements.

Input is the JSONL written by bench/gemm_sweep.py (one `meta` line, then one row per shape with `us_kernel` --
device time of the kernels -- and/or `us_event`).  For every row the GEMM plugin is priced on the machine file,
and the report answers the two questions a ceiling predictor has to answer before anything else:

  1. does the bound hold?   T_bound <= measured on every shape (a violation is a bug in the model or the
                            machine file, not noise -- the tolerance is 1% for timer resolution);
  2. how far off is it?     eta = T_bound / measured, by bottleneck side and by size; a large gap on shapes a
                            vendor BLAS should handle well is a term the model is missing, not slack in BLAS.
"""
from __future__ import annotations

import argparse
import json
import statistics

from .predict import TheoryMachine, predict
from .validate import _spearman

TOL = 1.01


def load(path: str) -> tuple:
    meta, rows = {}, []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
        else:
            rows.append(d)
    return meta, rows


def _q(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _size(m: int, n: int) -> str:
    s = min(m, n)
    return "min(M,N) <= 16" if s <= 16 else "17-256" if s <= 256 else "257-2048" if s <= 2048 else "> 2048"


def check(mach: TheoryMachine, rows: list) -> dict:
    out = []
    for r in rows:
        meas = r.get("us_kernel") or r["us_event"]
        p = predict(mach, "gemm", {"M": r["M"], "N": r["N"], "K": r["K"]},
                    {"bfloat16": "bf16", "float16": "f16"}.get(r["dtype"], r["dtype"]), "fwd")
        side = p["T_bound_binding"].split(":")[0]
        out.append({"M": r["M"], "N": r["N"], "K": r["K"], "meas_us": meas, "T_bound_us": p["T_bound_us"],
                    "binding": p["T_bound_binding"], "side": side, "eta": p["T_bound_us"] / meas,
                    "tflops": p["work"]["flops"] / meas / 1e6, "gbps": p["work"]["bytes_min"] / meas / 1e3,
                    "violation": p["T_bound_us"] > meas * TOL, "size": _size(r["M"], r["N"]),
                    "kernels": r.get("kernels", {})})
    etas = [x["eta"] for x in out]

    def dist(xs):
        e = [x["eta"] for x in xs]
        return {"n": len(e), "p10": _q(e, .1), "median": statistics.median(e), "p90": _q(e, .9),
                "max": max(e)} if e else {"n": 0}
    by_side = {s: dist([x for x in out if x["side"] == s]) for s in sorted({x["side"] for x in out})}
    by_size = {s: dist([x for x in out if x["size"] == s])
               for s in ("min(M,N) <= 16", "17-256", "257-2048", "> 2048")}
    return {"schema": "opceil.check/1", "machine": mach.id, "peaks": mach.which, "n": len(out),
            "violations": [x for x in out if x["violation"]], "eta": dist(out),
            "spearman_bound_vs_meas": _spearman([x["T_bound_us"] for x in out], [x["meas_us"] for x in out]),
            "by_side": by_side, "by_size": by_size, "rows": out,
            "peak_tflops": mach.peaks.get("mmac_bf16", 0) / 1e12, "hbm_GBps": mach.bw / 1e9}


def markdown(r: dict, meta: dict) -> str:
    L = ["# opceil check: GEMM T_bound vs kernel-level measurement", "",
         "machine file `%s` (%s peaks: mmac_bf16 %.0f TFLOPS, HBM %.0f GB/s); measured on %s %s (%s CUs), torch %s, "
         "%s" % (r["machine"], r["peaks"], r["peak_tflops"], r["hbm_GBps"], meta.get("host", "?"),
                 meta.get("device", "?"), meta.get("cus", "?"), meta.get("torch", "?"), meta.get("date", "")), "",
         "| | value |", "| --- | ---: |",
         "| shapes | %d |" % r["n"],
         "| **bound violations** (T_bound > 1.01 x measured) | **%d** |" % len(r["violations"]),
         "| eta = T_bound / measured: p10 / median / p90 | %.3f / %.3f / %.3f |" % (
             r["eta"]["p10"], r["eta"]["median"], r["eta"]["p90"]),
         "| Spearman(T_bound, measured) | %.3f |" % r["spearman_bound_vs_meas"], "",
         "eta = 1 means the measured kernel runs at the bound; eta = 0.5 means it takes twice the bound.", "",
         "## by bottleneck side of the bound", "", "| side | n | p10 | median | p90 | max |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s, d in r["by_side"].items():
        L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (s, d["n"], d["p10"], d["median"], d["p90"], d["max"]))
    L += ["", "## by size", "", "| min(M,N) | n | p10 | median | p90 | max |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s, d in r["by_size"].items():
        if d["n"]:
            L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (s, d["n"], d["p10"], d["median"], d["p90"], d["max"]))
    if r["violations"]:
        L += ["", "## violations", "", "| M | N | K | measured us | T_bound us | binding |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
        L += ["| %d | %d | %d | %.2f | %.2f | %s |" % (x["M"], x["N"], x["K"], x["meas_us"], x["T_bound_us"], x["binding"])
              for x in r["violations"]]
    L += ["", "## all shapes (sorted by eta)", "",
          "| M | N | K | measured us | T_bound us | eta | side | TFLOPS | GB/s | kernels |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for x in sorted(r["rows"], key=lambda x: x["eta"]):
        L.append("| %d | %d | %d | %.2f | %.2f | %.3f | %s | %.1f | %.0f | %s |" % (
            x["M"], x["N"], x["K"], x["meas_us"], x["T_bound_us"], x["eta"], x["side"], x["tflops"], x["gbps"],
            ", ".join("%s x%d" % (k[:40], c) for k, c in x["kernels"].items())))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opceil check")
    p.add_argument("--machine", required=True)
    p.add_argument("--measured", required=True, help="JSONL from bench/gemm_sweep.py")
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    meta, rows = load(a.measured)
    r = check(TheoryMachine.load(a.machine, a.peaks), rows)
    md = markdown(r, meta)
    if a.out:
        open(a.out + ".check.json", "w", encoding="utf-8").write(json.dumps(r, indent=1) + "\n")
        open(a.out + ".check.md", "w", encoding="utf-8").write(md)
        print("wrote %s.check.json / .md" % a.out)
    else:
        print(md)
    return 0
