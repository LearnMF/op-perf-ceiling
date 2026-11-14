"""Cross-card transfer from measurements on a reference card: how well does what was measured on card A predict
card B, for the same implementation, for another implementation of the same operator, and for another operator?

`python -m opceil transfer` runs the experiment on the committed data (gfx936 -> gfx938 and back):

  same implementation    FLA KDA measured on A, predicted on B by: spec ratios (compute / HBM), the tool's own
                         per-shape model ratio (T_split p82, T_bound), the mk3 implementation's measured ratio, and
                         the per-pattern ratios learned from FLA GDN (another operator)
  another operator       FLA GDN measured on A, predicted on B from FLA KDA's ratios (whole-op and per pattern):
                         the "new operator, run the open-source version on a card we have" rung
  optimisation factor    FLA / mk3 on each card: whether "how much hand optimisation buys" carries across cards

Patterns are kernel classes shared by the FLA KDA and GDN kernels (PATTERNS); a kernel matching none is "stream".
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .predict import TheoryMachine, predict

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "measured"
CARDS = ("gfx936-bw1000", "gfx938-bw1102")
PATTERNS = [("scan_fwd", ("fwd_kernel_h",)), ("scan_bwd", ("bwd_kernel_dhu",)),
            ("chunk_tri", ("kernel_intra", "inter_solve", "kkt_solve", "dAv", "dv_local")),
            ("wy_fwd", ("recompute_w_u",)), ("out_fwd", ("fwd_kernel_o",)),
            ("bwd_wy_dqk", ("wy_dqkg", "prepare_wy_repr_bwd", "dqkwg"))]


def pattern(name: str) -> str:
    return next((p for p, keys in PATTERNS if any(k in name for k in keys)), "stream")


def load(path: Path, key) -> dict:
    out = {}
    for d in map(json.loads, open(path)):
        if "meta" in d or "error" in d:
            continue
        by = {}
        for n, v in d["tot_kernels"].items():
            by[pattern(n)] = by.get(pattern(n), 0.0) + v
        out[key(d)] = {"t": d["tot_us_kernel"], "p": by}
    return out


def _err(pred: dict, meas: dict) -> dict:
    keys = sorted(set(pred) & set(meas))
    e = [abs(pred[k] / meas[k] - 1) for k in keys]
    return {"n": len(e), "median_pct": 100 * statistics.median(e), "mean_pct": 100 * statistics.mean(e),
            "max_pct": 100 * max(e), "within_10": sum(x <= .10 for x in e), "within_20": sum(x <= .20 for x in e)}


def class_ratios(src_a: dict, src_b: dict) -> tuple:
    """Per-pattern and whole-op median time ratio B / A over the shapes both cards measured."""
    keys = set(src_a) & set(src_b)
    pats = {p for k in keys for p in src_a[k]["p"]}
    per = {p: statistics.median(src_b[k]["p"][p] / src_a[k]["p"][p] for k in keys
                                if src_a[k]["p"].get(p) and src_b[k]["p"].get(p)) for p in pats}
    return per, statistics.median(src_b[k]["t"] / src_a[k]["t"] for k in keys)


def by_pattern(src: dict, per: dict, whole: float) -> dict:
    return {k: sum(v * per.get(p, whole) for p, v in r["p"].items()) for k, r in src.items()}


def run(a: str, b: str) -> dict:
    M = {c: TheoryMachine.load(str(ROOT / "machines" / (c + ".json"))) for c in CARDS}
    kda = lambda d: (d["T"], d["H"])
    gdn = lambda d: d["name"]
    f = lambda op, c, tag: DATA / ("%s_bf16_%s_20261113_%s.jsonl" % (op, c, tag))
    fk = {c: load(f("fla-kda", c, "fla052"), kda) for c in CARDS}
    fg = {c: load(f("fla-gdn", c, "fla052"), gdn) for c in CARDS}
    mk = {c: load(next(DATA.glob("kda_bf16_%s_20261113_mk3-*.jsonl" % c)), kda) for c in CARDS}
    pk = lambda c: M[c].peaks["mmac_bf16"]
    spec = {"compute peak ratio": pk(a) / pk(b), "HBM bandwidth ratio": M[a].bw / M[b].bw}
    out = {"from": a, "to": b, "kda_fla": {}, "gdn_fla": {}, "spec_ratios_b_over_a": spec}
    # ---- same implementation (FLA KDA)
    meas = {k: v["t"] for k, v in fk[b].items()}
    E = out["kda_fla"]
    for name, r in spec.items():
        E["spec: " + name] = _err({k: v["t"] * r for k, v in fk[a].items()}, meas)
    model = {}
    for k in fk[a]:
        pa = predict(M[a], "kda", {"T": k[0], "H": k[1]}, "bf16", "fwd+bwd")
        pb = predict(M[b], "kda", {"T": k[0], "H": k[1]}, "bf16", "fwd+bwd")
        model[k] = (pb["T_split"]["p82"]["T_split_us"] / pa["T_split"]["p82"]["T_split_us"],
                    pb["T_bound_us"] / pa["T_bound_us"])
    E["tool model ratio (T_split p82, per shape)"] = _err({k: v["t"] * model[k][0] for k, v in fk[a].items()}, meas)
    # T_bound is not reported: it prices gfx936 at its datasheet HBM peak and gfx938 (no datasheet value) at its
    # measured one, so its cross-card ratio mixes two peak sources; on consistent measured peaks it is the HBM row.
    _, mk_whole = class_ratios(mk[a], mk[b])
    E["mk3's measured ratio (other implementation, median)"] = _err({k: v["t"] * mk_whole for k, v in fk[a].items()}, meas)
    per_g, whole_g = class_ratios(fg[a], fg[b])
    E["FLA GDN ratio, whole op (other operator)"] = _err({k: v["t"] * whole_g for k, v in fk[a].items()}, meas)
    E["FLA GDN ratios per pattern (other operator)"] = _err(by_pattern(fk[a], per_g, whole_g), meas)
    # ---- another operator (FLA GDN): only measured on card a, ratios learned from KDA
    meas = {k: v["t"] for k, v in fg[b].items()}
    E = out["gdn_fla"]
    for name, r in spec.items():
        E["spec: " + name] = _err({k: v["t"] * r for k, v in fg[a].items()}, meas)
    per_k, whole_k = class_ratios(fk[a], fk[b])
    E["FLA KDA ratio, whole op"] = _err({k: v["t"] * whole_k for k, v in fg[a].items()}, meas)
    E["FLA KDA ratios per pattern"] = _err(by_pattern(fg[a], per_k, whole_k), meas)
    E["(reference) same-op LOO: FLA GDN's own median ratio, leave one shape out"] = _err(
        {k: fg[a][k]["t"] * statistics.median(fg[b][j]["t"] / fg[a][j]["t"] for j in fg[a] if j != k and j in fg[b])
         for k in fg[a] if k in fg[b]}, meas)
    out["pattern_ratios_b_over_a"] = {"FLA KDA": per_k, "FLA GDN": per_g}
    out["whole_ratio_b_over_a"] = {"FLA KDA": whole_k, "FLA GDN": whole_g, "mk3 KDA": mk_whole}
    # ---- optimisation factor FLA / mk3 per card
    F = {c: {k: fk[c][k]["t"] / mk[c][k]["t"] for k in fk[c] if k in mk[c]} for c in CARDS}
    keys = sorted(set(F[a]) & set(F[b]))
    out["fla_over_mk3"] = {c: {"median": statistics.median(F[c][k] for k in keys), "min": min(F[c][k] for k in keys),
                               "max": max(F[c][k] for k in keys)} for c in CARDS}
    out["mk3_on_b_from_fla_on_b_divided_by_factor_of_a"] = _err(
        {k: fk[b][k]["t"] / F[a][k] for k in keys}, {k: mk[b][k]["t"] for k in keys})
    return out


def markdown(rs: list) -> str:
    L = ["# Cross-card transfer from reference-card measurements", ""]
    for r in rs:
        L += ["## %s -> %s" % (r["from"], r["to"]), "",
              "whole-op median time ratio (to / from): FLA KDA %.2f, FLA GDN %.2f, mk3 KDA %.2f; spec: compute %.2f, HBM %.2f"
              % (r["whole_ratio_b_over_a"]["FLA KDA"], r["whole_ratio_b_over_a"]["FLA GDN"],
                 r["whole_ratio_b_over_a"]["mk3 KDA"], r["spec_ratios_b_over_a"]["compute peak ratio"],
                 r["spec_ratios_b_over_a"]["HBM bandwidth ratio"]), ""]
        for sec, title in (("kda_fla", "FLA KDA on `to`, predicted from FLA KDA on `from` (same implementation)"),
                           ("gdn_fla", "FLA GDN on `to`, predicted from FLA GDN on `from` (another operator)")):
            L += ["### " + title, "", "| predictor | median | mean | max | within 10% | within 20% |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
            L += ["| %s | %.1f%% | %.1f%% | %.1f%% | %d / %d | %d / %d |" % (k, e["median_pct"], e["mean_pct"], e["max_pct"],
                                                                         e["within_10"], e["n"], e["within_20"], e["n"])
                  for k, e in r[sec].items()]
            L.append("")
        L += ["### pattern time ratios (to / from)", "", "| pattern | FLA KDA | FLA GDN |", "| --- | ---: | ---: |"]
        pk, pg = r["pattern_ratios_b_over_a"]["FLA KDA"], r["pattern_ratios_b_over_a"]["FLA GDN"]
        L += ["| %s | %s | %s |" % (p, "%.2f" % pk[p] if p in pk else "-", "%.2f" % pg[p] if p in pg else "-")
              for p in sorted(set(pk) | set(pg))]
        fo = r["fla_over_mk3"]
        L += ["", "### optimisation factor FLA / mk3", "",
              "| card | median | min | max |", "| --- | ---: | ---: | ---: |"]
        L += ["| %s | %.2f | %.2f | %.2f |" % (c, v["median"], v["min"], v["max"]) for c, v in fo.items()]
        e = r["mk3_on_b_from_fla_on_b_divided_by_factor_of_a"]
        L += ["", "mk3 on `to` predicted as (FLA measured on `to`) / (FLA/mk3 factor of `from`): median %.1f%%, max %.1f%%"
              % (e["median_pct"], e["max_pct"]), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opceil transfer")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rs = [run(CARDS[0], CARDS[1]), run(CARDS[1], CARDS[0])]
    md = markdown(rs)
    if a.out:
        open(a.out + ".transfer.json", "w").write(json.dumps(rs, indent=1, default=str) + "\n")
        open(a.out + ".transfer.md", "w").write(md)
        print("wrote %s.transfer.json / .md" % a.out)
    else:
        print(md)
    return 0
