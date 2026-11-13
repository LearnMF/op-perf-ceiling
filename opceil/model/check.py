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
        sp = p["T_split"].get("tiled")
        ts = sp["T_split_us"] if sp else None
        tl = sp["kernels"][0]["tiling"] if sp else {}
        out.append({"M": r["M"], "N": r["N"], "K": r["K"], "meas_us": meas, "T_bound_us": p["T_bound_us"],
                    "T_split_us": ts, "eta_split": ts / meas if ts else None,
                    "split_binding": sp["kernels"][0]["binding"] if sp else None,
                    "tile": "%dx%dx%d/s%d" % (tl["Mt"], tl["Nt"], tl["Kt"], tl["split_k"]) if tl else "",
                    "split_violation": bool(ts and ts > meas * TOL),
                    "binding": p["T_bound_binding"], "side": side, "eta": p["T_bound_us"] / meas,
                    "tflops": p["work"]["flops"] / meas / 1e6, "gbps": p["work"]["bytes_min"] / meas / 1e3,
                    "violation": p["T_bound_us"] > meas * TOL, "size": _size(r["M"], r["N"]),
                    "kernels": r.get("kernels", {})})
    etas = [x["eta"] for x in out]

    def dist(xs, key="eta"):
        e = [x[key] for x in xs if x[key] is not None]
        return {"n": len(e), "p10": _q(e, .1), "median": statistics.median(e), "p90": _q(e, .9),
                "max": max(e)} if e else {"n": 0}
    by_side = {s: dist([x for x in out if x["side"] == s]) for s in sorted({x["side"] for x in out})}
    sizes = ("min(M,N) <= 16", "17-256", "257-2048", "> 2048")
    by_size = {s: dist([x for x in out if x["size"] == s]) for s in sizes}
    by_size_split = {s: dist([x for x in out if x["size"] == s], "eta_split") for s in sizes}
    return {"schema": "opceil.check/1", "machine": mach.id, "peaks": mach.which, "n": len(out),
            "violations": [x for x in out if x["violation"]], "eta": dist(out),
            "split_violations": [x for x in out if x["split_violation"]], "eta_split": dist(out, "eta_split"),
            "by_size_split": by_size_split,
            "spearman_split_vs_meas": _spearman([x["T_split_us"] or 0 for x in out], [x["meas_us"] for x in out]),
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
         "| Spearman(T_bound, measured) | %.3f |" % r["spearman_bound_vs_meas"],
         "| T_split (tiled, model) above measured (> 1.01 x) | %d |" % len(r["split_violations"]),
         "| eta_split = T_split / measured: p10 / median / p90 | %.3f / %.3f / %.3f |" % (
             r["eta_split"]["p10"], r["eta_split"]["median"], r["eta_split"]["p90"]),
         "| Spearman(T_split, measured) | %.3f |" % r["spearman_split_vs_meas"], "",
         "eta = 1 means the measured kernel runs at the bound; eta = 0.5 means it takes twice the bound.  "
         "T_split above measured means the tiled model claims more cost than the vendor BLAS actually pays: "
         "the model is wrong there, not the library.", "",
         "## by bottleneck side of the bound", "", "| side | n | p10 | median | p90 | max |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s, d in r["by_side"].items():
        L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (s, d["n"], d["p10"], d["median"], d["p90"], d["max"]))
    L += ["", "## by size (T_bound | T_split)", "",
          "| min(M,N) | n | eta p10 | eta median | eta p90 | eta_split p10 | eta_split median | eta_split p90 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s, d in r["by_size"].items():
        q = r["by_size_split"][s]
        if d["n"]:
            L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                s, d["n"], d["p10"], d["median"], d["p90"], q["p10"], q["median"], q["p90"]))
    if r["split_violations"]:
        L += ["", "## T_split above measured", "", "| M | N | K | measured us | T_split us | binding | tile |",
              "| ---: | ---: | ---: | ---: | ---: | --- | --- |"]
        L += ["| %d | %d | %d | %.2f | %.2f | %s | %s |" % (x["M"], x["N"], x["K"], x["meas_us"], x["T_split_us"],
                                                         x["split_binding"], x["tile"]) for x in r["split_violations"]]
    if r["violations"]:
        L += ["", "## violations", "", "| M | N | K | measured us | T_bound us | binding |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
        L += ["| %d | %d | %d | %.2f | %.2f | %s |" % (x["M"], x["N"], x["K"], x["meas_us"], x["T_bound_us"], x["binding"])
              for x in r["violations"]]
    L += ["", "## all shapes (sorted by eta)", "",
          "| M | N | K | measured us | T_bound us | eta | side | T_split us | eta_split | split binding | tile | "
          "TFLOPS | GB/s | kernels |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- |"]
    for x in sorted(r["rows"], key=lambda x: x["eta"]):
        L.append("| %d | %d | %d | %.2f | %.2f | %.3f | %s | %.2f | %.3f | %s | %s | %.1f | %.0f | %s |" % (
            x["M"], x["N"], x["K"], x["meas_us"], x["T_bound_us"], x["eta"], x["side"], x["T_split_us"] or 0,
            x["eta_split"] or 0, x["split_binding"], x["tile"], x["tflops"], x["gbps"],
            ", ".join("%s x%d" % (k[:40], c) for k, c in x["kernels"].items())))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opceil check")
    p.add_argument("--machine", required=True)
    p.add_argument("--measured", required=True, help="JSONL from bench/gemm_sweep.py")
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--out", default=None)
    p.add_argument("--other", default=None, help="KDA only: the other card's measurements, for calibrated transfer "
                                                 "(format: machine.json:measured.jsonl)")
    a = p.parse_args(argv)
    meta, rows = load(a.measured)
    if rows and "T" in rows[0] and "H" in rows[0]:                     # bench/kda_bench.py output
        from . import check_kda
        other = None
        if a.other:
            om, of = a.other.split(":", 1)
            other = check_kda.rows_for(TheoryMachine.load(om, a.peaks), check_kda.load(of)[1])
        r = check_kda.check(TheoryMachine.load(a.machine, a.peaks), rows, other)
        md = check_kda.markdown(r, meta)
        if a.out:
            open(a.out + ".check.json", "w", encoding="utf-8").write(json.dumps(r, indent=1) + "\n")
            open(a.out + ".check.md", "w", encoding="utf-8").write(md)
            print("wrote %s.check.json / .md" % a.out)
        else:
            print(md)
        return 0
    r = check(TheoryMachine.load(a.machine, a.peaks), rows)
    md = markdown(r, meta)
    if a.out:
        open(a.out + ".check.json", "w", encoding="utf-8").write(json.dumps(r, indent=1) + "\n")
        open(a.out + ".check.md", "w", encoding="utf-8").write(md)
        print("wrote %s.check.json / .md" % a.out)
    else:
        print(md)
    return 0
