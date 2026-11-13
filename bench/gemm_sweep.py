"""Kernel-level GEMM timing on one HCU/DCU card, for checking opceil's bounds against measurements.

Standalone (needs only PyTorch on the device); run inside the card's container:

    HIP_VISIBLE_DEVICES=0 python bench/gemm_sweep.py --out gemm_gfx938.jsonl

What is timed is C = A @ B (torch.matmul -> the vendor BLAS) in bf16, one shape at a time:
  * us_event  -- mean of `iters` back-to-back calls between two device events (dispatch overlaps execution,
                 so for kernels longer than a few us this is the kernel time);
  * us_kernel -- device time of the kernels the call launched, from torch.profiler, summed per call (BLAS may
                 launch more than one kernel, e.g. split-K + reduction); absent if the profiler gives nothing.
The A/B/C buffers rotate over enough sets that the inputs are not L2-resident between calls (the bound charges
HBM bytes; a warm L2 would let a small GEMM beat it legitimately and say nothing about the model).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time

import torch

# (N, K) of the projection shapes used in practice (LLaMA / DeepSeek / Qwen families, PV sheet), swept over M.
NK = [(4096, 4096), (4096, 11008), (11008, 4096), (6144, 2048), (4096, 6144), (6144, 12288),
      (256, 6144), (24576, 6144), (2304, 1152), (576, 4608)]
MS = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SQUARE = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]


def shapes() -> list:
    out = [(m, n, k) for n, k in NK for m in MS] + [(s, s, s) for s in SQUARE]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _sets(m: int, n: int, k: int, e: int, l2_bytes: int) -> int:
    per = (m * k + k * n + m * n) * e
    return max(1, min(64, -(-4 * l2_bytes // per)))


def time_shape(m: int, n: int, k: int, dtype, target_ms: float, l2_bytes: int, profile: bool) -> dict:
    e = torch.tensor([], dtype=dtype).element_size()
    ns = _sets(m, n, k, e, l2_bytes)
    A = [torch.randn(m, k, device="cuda", dtype=dtype) for _ in range(ns)]
    B = [torch.randn(k, n, device="cuda", dtype=dtype) for _ in range(ns)]
    C = [torch.empty(m, n, device="cuda", dtype=dtype) for _ in range(ns)]
    for i in range(3 * ns):
        torch.matmul(A[i % ns], B[i % ns], out=C[i % ns])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(ns):
        torch.matmul(A[i], B[i], out=C[i])
    torch.cuda.synchronize()
    once = max((time.perf_counter() - t0) / ns, 1e-6)
    iters = int(max(ns, min(20000, target_ms * 1e-3 / once)))
    s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(iters):
        torch.matmul(A[i % ns], B[i % ns], out=C[i % ns])
    t.record()
    torch.cuda.synchronize()
    row = {"M": m, "N": n, "K": k, "dtype": str(dtype).replace("torch.", ""), "sets": ns, "iters": iters,
           "us_event": s.elapsed_time(t) * 1e3 / iters}
    if profile:
        reps = min(iters, 50)
        try:
            from torch.profiler import ProfilerActivity, profile as prof
            with prof(activities=[ProfilerActivity.CUDA]) as p:
                for i in range(reps):
                    torch.matmul(A[i % ns], B[i % ns], out=C[i % ns])
                torch.cuda.synchronize()
            ev = [x for x in p.events() if x.device_type.name == "CUDA"]
            if ev:
                row["us_kernel"] = sum(x.device_time for x in ev) / reps
                names = {}
                for x in ev:
                    names[x.name] = names.get(x.name, 0) + 1
                row["kernels"] = {n_: c // reps or 1 for n_, c in sorted(names.items(), key=lambda kv: -kv[1])[:4]}
        except Exception as ex:  # profiler support differs across DTK builds; the event time stands alone
            row["profiler_error"] = repr(ex)[:200]
    del A, B, C
    torch.cuda.empty_cache()
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--target-ms", type=float, default=60.0, help="timed window per shape")
    p.add_argument("--l2-bytes", type=int, default=8 << 20)
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N shapes only (smoke test)")
    a = p.parse_args()
    dtype = getattr(torch, a.dtype)
    prop = torch.cuda.get_device_properties(0)
    meta = {"host": socket.gethostname(), "device": prop.name, "arch": getattr(prop, "gcnArchName", ""),
            "cus": prop.multi_processor_count, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
            "visible": os.environ.get("HIP_VISIBLE_DEVICES", ""), "python": platform.python_version(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    todo = shapes()[: a.limit or None]
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, (m, n, k) in enumerate(todo):
            row = time_shape(m, n, k, dtype, a.target_ms, a.l2_bytes, not a.no_profile)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print("%3d/%d M=%5d N=%5d K=%5d  event %9.2f us  kernel %s" % (
                i + 1, len(todo), m, n, k, row["us_event"],
                "%9.2f us" % row["us_kernel"] if "us_kernel" in row else "-"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
