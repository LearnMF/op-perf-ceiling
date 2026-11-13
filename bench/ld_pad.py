"""Power-of-two leading dimensions vs padded ones, same GEMM: separates a memory-channel effect from the tiling.

    HIP_VISIBLE_DEVICES=1 python bench/ld_pad.py
"""
import json
import torch

PAD = 64


def t(m, n, k, pad, reps=20):
    A = torch.randn(m, k + pad, device="cuda", dtype=torch.bfloat16)[:, :k]
    B = torch.randn(k, n + pad, device="cuda", dtype=torch.bfloat16)[:, :n]
    C = torch.empty(m, n + pad, device="cuda", dtype=torch.bfloat16)[:, :n]
    for _ in range(3):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        torch.matmul(A, B, out=C)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


out = []
for sh in [(4096, 4096, 4096), (8192, 8192, 8192), (12288, 12288, 12288), (16384, 16384, 16384), (8192, 4096, 4096)]:
    a, b = t(*sh, 0), t(*sh, PAD)
    out.append({"shape": sh, "us_ld_pow2": a, "us_ld_padded": b, "tflops_pow2": 2 * sh[0] * sh[1] * sh[2] / a / 1e6,
                "tflops_padded": 2 * sh[0] * sh[1] * sh[2] / b / 1e6})
    print(json.dumps(out[-1]), flush=True)
