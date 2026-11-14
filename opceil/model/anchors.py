"""Unknown hardware from a few anchor kernels: S4000 / S2000 (SOC simulation results, data/pv/) as the test case.

The cards cannot be run; the PV sheet is the ground truth.  Two questions:

1. anchors on the target card only -- pick one floor-like, one memory-bound and one compute-bound GEMM row of the
   card, derive its effective machine (fixed floor, streaming bandwidth, fp8 compute) from them, and predict every
   other GEMM row with the L0 roofline  T = max(floor, bytes / BW, flops / P).  Every anchor combination is tried.
2. card A -> card B for other operator families -- MoE and DSA latencies of S2000 scaled to S4000 by ratios learned
   from the GEMM anchors (floor / bandwidth / compute / whole-family median), and, for MoE, by the ratio of the L0
   model evaluated on each card's effective machine with that card's own EP size (S4000 EP 8, S2000 EP 16): the tool
   normalises the per-rank work before the measured latency is transferred.

`python -m opceil anchors`
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PV = ROOT / "data" / "pv" / "pv_soc_sim_20261016.xlsx"
CARDS = ("s4000", "s2000")
E = {"bfloat16": 2, "fp8_block": 1}


def load() -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(PV, data_only=True)
    out = {}
    for c in CARDS:
        g = [dict(dtype=r[0], M=r[1], N=r[2], K=r[3], t=float(r[4])) for r in list(wb["gemm_" + c].iter_rows(values_only=True))[1:]
             if isinstance(r[4], (int, float))]
        d = [dict(batch=r[4], step=r[7], t=float(r[8])) for r in list(wb[c + "_dsa_generation"].iter_rows(values_only=True))[1:]
             if isinstance(r[8], (int, float))]
        m = [dict(tokens=r[3], hidden=r[4], inter=r[5], topk=r[6], experts=r[7], ep=r[10], t=float(r[11]))
             for r in list(wb[c + "_moe"].iter_rows(values_only=True))[1:] if len(r) > 11 and isinstance(r[11], (int, float))]
        out[c] = {"gemm": g, "dsa": d, "moe": m}
    return out


def gemm_work(r: dict) -> tuple:
    e = E[r["dtype"]]
    return 2.0 * r["M"] * r["N"] * r["K"], float((r["M"] * r["K"] + r["K"] * r["N"]) * e + r["M"] * r["N"] * 2)


def moe_work(r: dict) -> tuple:
    """Per-rank fp8 expert GEMMs (gate+up then down) under EP: routed tokens and the weights of the experts touched."""
    per_rank = r["experts"] // r["ep"]
    routed = r["tokens"] * r["topk"] / r["ep"]                      # token-expert pairs landing on this rank
    w_expert = 3.0 * r["hidden"] * r["inter"]                       # fp8 bytes of one expert's three matrices
    touched = min(per_rank, routed)
    flops = 2.0 * routed * w_expert
    act = routed * (r["hidden"] * 2 + r["inter"] * 2 * 2)            # activations in/out of both GEMMs, bf16
    return flops, touched * w_expert + act


def machine(anchors: list) -> dict:
    """Effective floor / bandwidth / fp8 compute of a card from its three anchor rows."""
    f, mem, comp = anchors
    return {"floor": f["t"], "bw": gemm_work(mem)[1] / mem["t"], "peak": gemm_work(comp)[0] / comp["t"]}


def l0(m: dict, flops: float, nbytes: float) -> float:
    return max(m["floor"], nbytes / m["bw"], flops / m["peak"])


def _err(pred: list, meas: list) -> dict:
    e = [abs(p / t - 1) for p, t in zip(pred, meas)]
    return {"n": len(e), "median_pct": 100 * statistics.median(e), "mean_pct": 100 * statistics.mean(e),
            "max_pct": 100 * max(e), "within_10": sum(x <= .1 for x in e), "within_20": sum(x <= .2 for x in e)}


def roles(rows: list) -> dict:
    """Anchor candidates by role, chosen from the shape alone (no latency): smallest work = floor; M <= 2 bf16 with
    the largest weight = memory; fp8 with M >= 4096 = compute."""
    return {"floor": [r for r in rows if r["M"] <= 2 and r["N"] * r["K"] <= 256 * 6144],
            "memory": [r for r in rows if r["M"] <= 2 and r["N"] * r["K"] >= 6144 * 6144],
            "compute": [r for r in rows if r["dtype"] == "fp8_block" and r["M"] >= 4096]}


def exp_within(data: dict) -> dict:
    out = {}
    for c in CARDS:
        rows = data[c]["gemm"]
        R = roles(rows)
        errs, combos = [], 0
        for a in itertools.product(R["floor"], R["memory"], R["compute"]):
            m = machine(list(a))
            rest = [r for r in rows if r not in a]
            pred = [l0(m, *gemm_work(r)) for r in rest]
            errs.append(_err(pred, [r["t"] for r in rest]))
            combos += 1
        out[c] = {"combos": combos, "median_of_median_pct": statistics.median(e["median_pct"] for e in errs),
                  "worst_median_pct": max(e["median_pct"] for e in errs),
                  "median_of_max_pct": statistics.median(e["max_pct"] for e in errs),
                  "best_combo": min(errs, key=lambda e: e["median_pct"]), "worst_combo": max(errs, key=lambda e: e["median_pct"])}
    return out


def exp_cross(data: dict) -> dict:
    a, b = "s2000", "s4000"                        # predict S4000 from S2000
    R = {c: roles(data[c]["gemm"]) for c in CARDS}
    pick = lambda c, role: R[c][role][-1]
    key = lambda r: (r["dtype"], r["M"], r["N"], r["K"])
    ga = {key(r): r["t"] for r in data[a]["gemm"]}
    gb = {key(r): r["t"] for r in data[b]["gemm"]}
    ratios = {"floor anchor": pick(b, "floor")["t"] / pick(a, "floor")["t"],
              "memory anchor": pick(b, "memory")["t"] / pick(a, "memory")["t"],
              "compute anchor": pick(b, "compute")["t"] / pick(a, "compute")["t"],
              "whole GEMM median": statistics.median(gb[k] / ga[k] for k in ga if k in gb),
              "'S2200 = 83% x S2000' style: one flat 1/1.0": 1.0}
    ma, mb = (machine([pick(c, "floor"), pick(c, "memory"), pick(c, "compute")]) for c in (a, b))
    out = {"ratios_b_over_a": ratios, "machines": {a: ma, b: mb}}
    # DSA: same configs on both cards
    da = {(r["batch"], r["step"]): r["t"] for r in data[a]["dsa"]}
    db = {(r["batch"], r["step"]): r["t"] for r in data[b]["dsa"]}
    ks = sorted(set(da) & set(db))
    out["dsa"] = {"n": len(ks), "measured_ratio_median": statistics.median(db[k] / da[k] for k in ks),
                  "by_ratio": {n: _err([da[k] * r for k in ks], [db[k] for k in ks]) for n, r in ratios.items()}}
    # MoE: different EP per card -> plain ratios compare different per-rank work; the L0 ratio normalises it
    ma_rows = {r["tokens"]: r for r in data[a]["moe"]}
    mb_rows = {r["tokens"]: r for r in data[b]["moe"]}
    ks = sorted(set(ma_rows) & set(mb_rows))
    by = {n: _err([ma_rows[k]["t"] * r for k in ks], [mb_rows[k]["t"] for k in ks]) for n, r in ratios.items()}
    by["L0 model ratio on the anchor machines, each card's own EP"] = _err(
        [ma_rows[k]["t"] * l0(mb, *moe_work(mb_rows[k])) / l0(ma, *moe_work(ma_rows[k])) for k in ks],
        [mb_rows[k]["t"] for k in ks])
    by["L0 model alone on S4000's anchor machine (no S2000 data)"] = _err(
        [l0(mb, *moe_work(mb_rows[k])) for k in ks], [mb_rows[k]["t"] for k in ks])
    out["moe"] = {"tokens": ks, "ep": {a: ma_rows[ks[0]]["ep"], b: mb_rows[ks[0]]["ep"]}, "by_ratio": by,
                  "measured_ratio": {k: mb_rows[k]["t"] / ma_rows[k]["t"] for k in ks}}
    return out


def markdown(w: dict, x: dict) -> str:
    L = ["# Unknown hardware from anchor kernels (PV SOC simulation: S4000 / S2000)", "",
         "## 1. GEMM, anchors on the target card only (1 floor + 1 memory + 1 compute row -> L0 roofline for the rest)", "",
         "| card | anchor combos | median error (median over combos) | worst combo median | typical max error |",
         "| --- | ---: | ---: | ---: | ---: |"]
    L += ["| %s | %d | %.1f%% | %.1f%% | %.1f%% |" % (c, v["combos"], v["median_of_median_pct"], v["worst_median_pct"],
                                                   v["median_of_max_pct"]) for c, v in w.items()]
    L += ["", "## 2. S2000 -> S4000 for other families", "",
          "ratios S4000/S2000 from the GEMM anchors: " + ", ".join("%s %.3f" % kv for kv in x["ratios_b_over_a"].items()), "",
          "### DSA decode (%d configs, measured ratio median %.3f)" % (x["dsa"]["n"], x["dsa"]["measured_ratio_median"]), "",
          "| ratio used | median | mean | max | within 10% |", "| --- | ---: | ---: | ---: | ---: |"]
    L += ["| %s | %.1f%% | %.1f%% | %.1f%% | %d / %d |" % (k, e["median_pct"], e["mean_pct"], e["max_pct"], e["within_10"], e["n"])
          for k, e in x["dsa"]["by_ratio"].items()]
    L += ["", "### MoE (tokens %s; EP %s)" % (x["moe"]["tokens"], x["moe"]["ep"]), "",
          "| ratio used | median | mean | max | within 10% |", "| --- | ---: | ---: | ---: | ---: |"]
    L += ["| %s | %.1f%% | %.1f%% | %.1f%% | %d / %d |" % (k, e["median_pct"], e["mean_pct"], e["max_pct"], e["within_10"], e["n"])
          for k, e in x["moe"]["by_ratio"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opceil anchors")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    data = load()
    w, x = exp_within(data), exp_cross(data)
    md = markdown(w, x)
    if a.out:
        open(a.out + ".anchors.json", "w").write(json.dumps({"within": w, "cross": x}, indent=1, default=str) + "\n")
        open(a.out + ".anchors.md", "w").write(md)
        print("wrote %s.anchors.json / .md" % a.out)
    print(md)
    return 0
