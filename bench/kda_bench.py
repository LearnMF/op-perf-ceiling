"""KDA (flash-train kimi_delta_attention) kernel-level timing on one card, for checking opceil's KDA ceilings.

Run in the card's container with the flash-train build under test first on PYTHONPATH:

    PYTHONPATH=<site dir of the flash-train wheel> HIP_VISIBLE_DEVICES=0 python bench/kda_bench.py --out kda.jsonl

Inputs follow flash-train's own benchmark (benchmarks/ftrain_bench/ops/kda.py): one sequence [T, H, 128] bf16,
Kimi K3 gate (A_log / dt_bias), K = V = 128, training forward (inputs require grad) and backward via
torch.autograd.grad.  Per shape it records

  fwd_us_kernel / fwd_us_event   training forward only
  tot_us_kernel / tot_us_event   forward + backward
  bwd_*                          tot - fwd
where *_us_kernel is the summed device time of every kernel the call launched (torch.profiler), i.e. what the
operator costs on the device, glue kernels included, and *_us_event the wall time between two device events.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = [(t, h) for h in (64, 32, 16, 8) for t in (1024, 2048, 4096, 8192, 16384, 32768)] + \
         [(8192, 12), (16384, 12), (32768, 12)]


def make_args(T: int, H: int, seed: int = 2026):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k, v, g = ((r(T, H, 128) * 0.3).to(torch.bfloat16) for _ in range(4))
    beta = r(T, H).sigmoid()
    a_log = torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen))
    dt_bias = r(H * 128) * 0.1
    d_o = r(T, H, 128).to(torch.bfloat16)
    return [q, k, v, g, beta, a_log, dt_bias], d_o


def _timed(fn, reps: int) -> tuple:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    ev = s.elapsed_time(e) * 1e3 / reps
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    evs = [x for x in p.events() if x.device_type.name == "CUDA"]
    names = {}
    for x in evs:
        names[x.name] = names.get(x.name, 0.0) + x.device_time / reps
    return ev, sum(names.values()), dict(sorted(names.items(), key=lambda kv: -kv[1])[:12])


def run_shape(kda, T: int, H: int, target_ms: float) -> dict:
    leaves0, d_o = make_args(T, H)

    def fwd():
        leaves = [x.detach().requires_grad_(True) for x in leaves0]
        q, k, v, g, beta, a_log, dt_bias = leaves
        return kda(q, k, v, g, beta, A_log=a_log, dt_bias=dt_bias), leaves

    def fwd_only():
        fwd()

    def fwd_bwd():
        o, leaves = fwd()
        torch.autograd.grad(o, leaves, d_o)

    t0 = time.perf_counter()
    fwd_bwd()
    torch.cuda.synchronize()
    once = max(time.perf_counter() - t0, 1e-5)
    reps = int(max(3, min(200, target_ms * 1e-3 / once)))
    fe, fk, fnames = _timed(fwd_only, reps)
    te, tk, tnames = _timed(fwd_bwd, reps)
    return {"T": T, "H": H, "reps": reps, "fwd_us_event": fe, "fwd_us_kernel": fk, "tot_us_event": te,
            "tot_us_kernel": tk, "bwd_us_event": te - fe, "bwd_us_kernel": tk - fk,
            "fwd_kernels": fnames, "tot_kernels": tnames}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target-ms", type=float, default=300.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import flash_train
    import flash_train.torch as ft
    prop = torch.cuda.get_device_properties(0)
    meta = {"host": socket.gethostname(), "device": prop.name, "arch": getattr(prop, "gcnArchName", ""),
            "cus": prop.multi_processor_count, "torch": torch.__version__, "flash_train": flash_train.__file__,
            "flash_train_commit": os.environ.get("FT_COMMIT", ""), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, (T, H) in enumerate(SHAPES[: a.limit or None]):
            row = run_shape(ft.kimi_delta_attention, T, H, a.target_ms)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print("%2d/%d T=%5d H=%2d  fwd %9.1f us  bwd %9.1f us  tot %9.1f us (kernel)  tot event %9.1f us" % (
                i + 1, len(SHAPES), T, H, row["fwd_us_kernel"], row["bwd_us_kernel"], row["tot_us_kernel"],
                row["tot_us_event"]), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
