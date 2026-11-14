"""Open-source (FLA, Triton) KDA and GDN timing on one card: the "reference implementation on a known card" rung.

    HIP_VISIBLE_DEVICES=0 python bench/fla_bench.py --op kda --out fla_kda.jsonl
    HIP_VISIBLE_DEVICES=0 python bench/fla_bench.py --op gdn --out fla_gdn.jsonl

kda: fla.ops.kda.chunk_kda with the Kimi K3 contract flash-train documents (A_log / dt_bias gate in kernel,
     lower_bound -5, q/k l2norm in kernel, safe_gate), [1, T, H, 128] bf16, same 27 (T, H) as bench/kda_bench.py.
gdn: fla.ops.gated_delta_rule.chunk_gated_delta_rule, q/k [B, S, H, K], v [B, S, HV, V] bf16, fp32 log decay
     g < 0 and sigmoid beta per value head, q/k l2norm in kernel; shapes from HYGON-AI/flash-train issue #29.
Per shape: training forward and forward + backward, summed device time of every kernel (torch.profiler) and
event time, plus the per-kernel device time (for mapping kernels onto patterns).
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import torch
from torch.profiler import ProfilerActivity, profile

KDA_SHAPES = [(t, h) for h in (64, 32, 16, 8) for t in (1024, 2048, 4096, 8192, 16384, 32768)] + \
             [(8192, 12), (16384, 12), (32768, 12)]
# issue #29: name -> (B, S, H, HV, K, V)
GDN_SHAPES = {"small2": (1, 128, 8, 8, 64, 64), "medium": (1, 256, 16, 32, 64, 64), "medium2": (1, 512, 16, 16, 128, 128),
              "baseline": (1, 1024, 32, 32, 128, 128), "v256": (1, 1024, 16, 16, 128, 256),
              "narrowK": (1, 2048, 32, 32, 64, 128), "narrowV": (1, 2048, 32, 32, 128, 64),
              "gqa4": (1, 2048, 8, 32, 128, 128), "llm2k": (1, 2048, 32, 32, 128, 128), "wide": (2, 2048, 16, 32, 128, 128),
              "bigbatch": (8, 512, 32, 32, 128, 128), "batch4": (4, 1024, 32, 32, 128, 128),
              "llm4k": (1, 4096, 32, 32, 128, 128), "llm8k": (1, 8192, 32, 32, 128, 128),
              "llm16k": (1, 16384, 32, 32, 128, 128)}


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
    names = {}
    for x in p.events():
        if x.device_type.name == "CUDA":
            names[x.name] = names.get(x.name, 0.0) + x.device_time / reps
    return ev, sum(names.values()), dict(sorted(names.items(), key=lambda kv: -kv[1]))


def kda_case(T: int, H: int):
    from fla.ops.kda import chunk_kda
    gen = torch.Generator(device="cuda").manual_seed(2026)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k, v, g = ((r(1, T, H, 128) * 0.3).to(torch.bfloat16) for _ in range(4))
    beta = r(1, T, H).sigmoid()
    a_log = torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen))
    dt_bias = r(H * 128) * 0.1
    d_o = r(1, T, H, 128).to(torch.bfloat16)
    leaves0 = [q, k, v, g, beta, a_log, dt_bias]

    def fwd():
        L = [x.detach().requires_grad_(True) for x in leaves0]
        o, _ = chunk_kda(L[0], L[1], L[2], L[3], L[4], A_log=L[5], dt_bias=L[6], lower_bound=-5.0,
                         use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True)
        return o, L
    return fwd, d_o


def gdn_case(B: int, S: int, H: int, HV: int, K: int, V: int):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    gen = torch.Generator(device="cuda").manual_seed(2026)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k = ((r(B, S, H, K) * 0.3).to(torch.bfloat16) for _ in range(2))
    v = (r(B, S, HV, V) * 0.3).to(torch.bfloat16)
    g = torch.nn.functional.logsigmoid(r(B, S, HV)).float() * 0.1
    beta = r(B, S, HV).sigmoid()
    d_o = r(B, S, HV, V).to(torch.bfloat16)
    leaves0 = [q, k, v, g, beta]

    def fwd():
        L = [x.detach().requires_grad_(True) for x in leaves0]
        o, _ = chunk_gated_delta_rule(L[0], L[1], L[2], L[3], L[4], use_qk_l2norm_in_kernel=True)
        return o, L
    return fwd, d_o


def run(fwd, d_o, target_ms: float) -> dict:
    def fwd_only():
        fwd()

    def fwd_bwd():
        o, L = fwd()
        torch.autograd.grad(o, L, d_o)

    fwd_bwd()                                   # Triton autotune / compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fwd_bwd()
    torch.cuda.synchronize()
    once = max(time.perf_counter() - t0, 1e-5)
    reps = int(max(3, min(200, target_ms * 1e-3 / once)))
    fe, fk, fn = _timed(fwd_only, reps)
    te, tk, tn = _timed(fwd_bwd, reps)
    return {"reps": reps, "fwd_us_event": fe, "fwd_us_kernel": fk, "tot_us_event": te, "tot_us_kernel": tk,
            "bwd_us_kernel": tk - fk, "fwd_kernels": fn, "tot_kernels": tn}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=("kda", "gdn"))
    p.add_argument("--out", required=True)
    p.add_argument("--target-ms", type=float, default=300.0)
    a = p.parse_args()
    import fla
    import triton
    prop = torch.cuda.get_device_properties(0)
    meta = {"host": socket.gethostname(), "device": prop.name, "arch": getattr(prop, "gcnArchName", ""),
            "cus": prop.multi_processor_count, "torch": torch.__version__, "fla": fla.__version__,
            "triton": triton.__version__, "op": a.op, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    cases = ([({"T": t, "H": h}, kda_case(t, h) if False else (t, h)) for t, h in KDA_SHAPES] if a.op == "kda" else
             [(dict(name=n, **dict(zip(("B", "S", "H", "HV", "K", "V"), s))), s) for n, s in GDN_SHAPES.items()])
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, (key, args) in enumerate(cases):
            try:
                fwd, d_o = kda_case(*args) if a.op == "kda" else gdn_case(*args)
                row = dict(key, **run(fwd, d_o, a.target_ms))
            except Exception as ex:                         # unsupported shape: record and continue
                row = dict(key, error=repr(ex)[:300])
            f.write(json.dumps(row) + "\n")
            f.flush()
            print("%2d/%d %s  %s" % (i + 1, len(cases), key, "tot %.1f us" % row["tot_us_kernel"]
                                     if "tot_us_kernel" in row else row["error"][:120]), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
