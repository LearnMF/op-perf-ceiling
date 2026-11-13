"""Shortest device-side duration of a kernel on this card: the floor every kernel pays, whatever it computes.

Standalone (PyTorch only), run in the card's container:

    HIP_VISIBLE_DEVICES=0 python bench/kernel_floor.py

Times trivial kernels -- an in-place add / fill / copy on 1 element up to 64 KiB, which read and write a
few cache lines and do no other work -- with torch.profiler and reports the minimum and median device
duration per kernel.  This is a machine property measured without any operator (the same standing as the
launch microbenchmark); `launch.kernel_min_us` in the machine file takes the overall minimum.
"""
from __future__ import annotations

import json
import socket

import torch
from torch.profiler import ProfilerActivity, profile

REPS = 200


def durations(fn) -> list:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
    return sorted(x.device_time for x in p.events() if x.device_type.name == "CUDA" and x.device_time > 0)


def main() -> int:
    out = {"host": socket.gethostname(), "device": torch.cuda.get_device_properties(0).name, "kernels": {}}
    for n in (1, 256, 16384):
        x = torch.zeros(n, device="cuda", dtype=torch.float32)
        y = torch.zeros(n, device="cuda", dtype=torch.float32)
        for name, fn in (("add_", lambda: x.add_(1.0)), ("fill_", lambda: x.fill_(2.0)), ("copy_", lambda: y.copy_(x))):
            d = durations(fn)
            if d:
                out["kernels"]["%s_%dB" % (name, 4 * n)] = {"min_us": d[0], "median_us": d[len(d) // 2], "n": len(d)}
    out["kernel_min_us"] = min(v["min_us"] for v in out["kernels"].values())
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
