"""`python -m opceil trace --machine M --script block.py [--dtype bf16]`: L0 ceiling of a PyTorch function.

block.py defines `build()` returning `(fn, args)` or `(fn, args, kwargs)`, with every tensor created on
device="meta" (requires_grad=True on the tensors whose gradients the function's backward produces).  Nothing
runs on a device.  The report gives the fused ceiling (the function as one kernel), the eager floor (one
kernel per aten op) and the ops that dominate the eager work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json

from ..model.predict import TheoryMachine, markdown, predict
from .torch_meta import summary, to_work, trace


def _load(path: str):
    spec = importlib.util.spec_from_file_location("opceil_trace_target", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    got = mod.build()
    fn, args = got[0], tuple(got[1])
    kwargs = dict(got[2]) if len(got) > 2 else {}
    return fn, args, kwargs


def run(mach: TheoryMachine, script: str, dtype: str) -> dict:
    fn, args, kwargs = _load(script)
    tr = trace(fn, *args, **kwargs)
    name = script.rsplit("/", 1)[-1].removesuffix(".py")
    fused = predict(mach, name, {}, dtype, "fwd", work=to_work(tr, "fused"))
    eager = predict(mach, name, {}, dtype, "fwd", work=to_work(tr, "eager"))
    return {"schema": "opceil.trace/1", "script": script, "fused": fused, "eager": eager,
            "ops": summary(tr), "unknown_ops": sorted(set(tr.unknown)),
            "fusion_headroom": eager["T_bound_us"] / fused["T_bound_us"] if fused["T_bound_us"] else None}


def trace_markdown(r: dict) -> str:
    f, e = r["fused"], r["eager"]
    L = ["# torch trace: `%s` on %s" % (r["script"], f["machine"]["id"]), "",
         "| | T_bound | T_theory | binding | kernels |", "| --- | ---: | ---: | --- | ---: |",
         "| fused (one kernel) | %.1f us | %.1f us | %s | 1 |" % (f["T_bound_us"], f["T_theory_us"], f["binding"]),
         "| eager (one kernel per op) | %.1f us | %.1f us | %s | %d |" % (
             e["T_bound_us"], e["T_theory_us"], e["binding"], e["op"]["kernels"]), "",
         "eager / fused = **%.2fx**: the most that fusing this function into one kernel can recover." % (
             r["fusion_headroom"] or 0), "",
         "## ops by work", "", "| op | count | GFLOP | MB moved |", "| --- | ---: | ---: | ---: |"]
    L += ["| %s | %d | %.3f | %.1f |" % (o["op"], o["count"], o["flops"] / 1e9, o["bytes"] / 1e6) for o in r["ops"]]
    if r["unknown_ops"]:
        L += ["", "ops without a flop formula (custom): " + ", ".join(r["unknown_ops"])]
    L += ["", "---", "", markdown(f)]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opceil trace")
    p.add_argument("--machine", required=True)
    p.add_argument("--script", required=True, help="python file defining build() -> (fn, args[, kwargs])")
    p.add_argument("--dtype", default="bf16", help="dtype whose compute peak the matrix flops run at")
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    r = run(TheoryMachine.load(a.machine, a.peaks), a.script, a.dtype)
    md = trace_markdown(r)
    if a.out:
        open(a.out + ".trace.json", "w", encoding="utf-8").write(json.dumps(r, indent=1, default=str) + "\n")
        open(a.out + ".trace.md", "w", encoding="utf-8").write(md)
        print("wrote %s.trace.json / .md" % a.out)
    else:
        print(md)
    return 0
