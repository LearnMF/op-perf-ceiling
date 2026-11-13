"""Tiled-GEMM ceiling: the best time any output-tiled GEMM kernel can reach on this machine.

T_bound treats a GEMM as fused work at the device peaks.  A real GEMM kernel is a grid of output tiles
(Mt x Nt, optionally split over K), and the tiling is forced on it by the on-chip capacity; it costs:

  compute:tiles     padded flops (whole tiles are computed) at the share of the device the grid can keep
                    busy -- a CU holds ceil(tiles / CUs) tiles of work, so a grid that does not divide into
                    the CUs leaves some idle in the last wave, and a small grid leaves most of them idle;
  memory:hbm        the required bytes at the bandwidth measured for that many resident CTAs (machine
                    curve `peaks.hbm_read_vs_ctas`) -- a skinny GEMM with few tiles cannot stream at peak;
  memory:l2 / lds   A re-read once per tile column, B once per tile row, through L2 and staged in LDS;
  serial:kloop      per wave of CTAs: the first operand load (HBM latency), then per K step the larger of
                    (one dependent MMAC + one barrier) and (the operand load latency / prefetch depth, with
                    prefetch depth = LDS stages - 1: a stage cannot be consumed before it has arrived), then
                    the store of C (L2 latency) -- what a kernel with few K steps per CTA cannot hide;
  fixed:launch      one dispatch.

The candidate space (tile sizes, split-K, threads per CTA, LDS stages) is a design choice of this
module, listed below; a tiling is feasible when its LDS and register footprint fits the machine file's
`device.lds_per_cu` / `device.max_vgpr_per_thread`.  T_split is the fastest feasible candidate: no
candidate is fitted to, or chosen from, a measured kernel.  Every term here is class `model`.
"""
from __future__ import annotations

import itertools
import math

from ..ops import Work

TILES = (16, 32, 64, 128, 256)
KTILES = (16, 32, 64, 128)
SPLITK = (1, 2, 4, 8, 16, 32)
THREADS = (64, 256, 512)
STAGES = (2, 3, 4)              # LDS buffers of the A / B tiles; prefetch depth = stages - 1
REG_BASE = 16                   # addressing / loop registers per thread (same assumption as the KDA plugin)
SIMDS_PER_CU = 4                # a CU's matrix peak needs at least one wave per SIMD
VGPR_MAX_DEFAULT = 256


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def candidates(m: int, n: int, k: int, e: int):
    """(Mt, Nt, Kt, split_k, threads) worth pricing for an [m,k] x [k,n] GEMM: tiles no larger than the
    next power of two above the dimension (a bigger one only adds padding)."""
    cap = lambda d: 1 << max(4, math.ceil(math.log2(max(d, 1))))
    for mt, nt, kt, s, thr, st in itertools.product(TILES, TILES, KTILES, SPLITK, THREADS, STAGES):
        if mt > cap(m) or nt > cap(n) or kt * s > max(k, kt):
            continue
        if thr // 64 > (mt // 16) * (nt // 16):
            continue                                    # more waves than 16x16 output fragments
        yield mt, nt, kt, s, thr, st


def footprint(mt: int, nt: int, kt: int, thr: int, e: int, stages: int) -> tuple:
    """(LDS bytes, VGPR per thread) of one CTA: `stages` copies of the A and B tiles in LDS; fp32
    accumulators, one stage of the global->LDS copy and REG_BASE in registers."""
    stage = (mt + nt) * kt * e
    lds = stages * stage
    vgpr = mt * nt / thr + stage / (thr * 4.0) + REG_BASE
    return lds, vgpr


def price(mach, m: int, n: int, k: int, e: int, out_e: int, peak_key: str, peak: float,
          mt: int, nt: int, kt: int, s: int, thr: int, stages: int) -> tuple:
    """(Work, footprint dict) of one tiling, or (None, reason) if it does not fit."""
    lds, vgpr = footprint(mt, nt, kt, thr, e, stages)
    lds_cu = float(mach.device.get("lds_per_cu") or mach.device.get("lds_per_block") or 0)
    vmax = float(mach.device.get("max_vgpr_per_thread") or VGPR_MAX_DEFAULT)
    if lds_cu and lds > lds_cu:
        return None, "lds"
    if vgpr > vmax:
        return None, "vgpr"
    gm, gn = _cdiv(m, mt), _cdiv(n, nt)
    ctas = gm * gn * s
    iters = _cdiv(k, kt * s)
    a_b, b_b = float(m * k * e), float(k * n * e)
    c_b = float(m * n * out_e)
    # split-K: every K slice adds its fp32 partial into C (atomics through L2); C is still written once
    part = float(m * n * 4 * s) if s > 1 else 0.0
    w = Work(flops=2.0 * m * n * k, bytes_rd=a_b + b_b, bytes_wr=c_b + (float(m * n * 4) if s > 1 else 0.0),
             bytes_l2=a_b * gn + b_b * gm + c_b + part,
             bytes_lds=2.0 * (a_b * gn + b_b * gm),
             occupancy={"ctas": ctas, "threads": thr, "lds_bytes": lds, "vgpr": vgpr})
    w.bytes_min = w.bytes_rd + w.bytes_wr
    con = mach.concurrency(w.occupancy) if mach.use_occupancy else {}
    cus = mach.cus or 1
    per_cu = _cdiv(ctas, cus)                               # tiles of work the busiest CU holds
    resident = max(1, int(con.get("ctas_per_cu") or 1))
    busy_waves = min(per_cu, resident) * (thr // 64)
    simd = min(1.0, busy_waves / SIMDS_PER_CU)
    padded = 2.0 * (gm * mt) * (gn * nt) * (iters * kt * s)
    w.extra_terms["compute:tiles"] = padded / (peak * simd * ctas / (cus * per_cu))
    lat = mach.latency_s
    hbm_lat, l2_lat = lat.get("gmem_hbm_ws") or 0.0, lat.get("gmem_l2_ws") or 0.0
    # An operand tile read by a single CTA row / column is streamed from HBM; one shared by several CTAs is
    # (at best) an L2 hit for all but the first.  A K step cannot finish before its stage has arrived, and
    # only stages - 1 loads can be in flight ahead of the one being consumed.
    load_lat = hbm_lat if (gm == 1 or gn == 1) else l2_lat
    step = max((lat.get("mmac_bf16_dependent") or 0.0) + (mach.barrier_s(thr) or 0.0),
               load_lat / (stages - 1))
    waves = _cdiv(ctas, cus * resident)
    chain = hbm_lat + iters * step + l2_lat
    if chain:
        w.extra_terms["serial:kloop"] = waves * chain
    return w, {"Mt": mt, "Nt": nt, "Kt": kt, "split_k": s, "threads": thr, "stages": stages, "ctas": ctas,
               "k_iters": iters,
               "lds_kb": lds / 1024, "vgpr": round(vgpr, 1), "ctas_per_cu": resident, "waves": waves}


def search(mach, m: int, n: int, k: int, e: int, out_e: int, peak_key: str, peak: float, top: int = 3) -> dict:
    from .predict import _bound
    best, rows, infeasible = None, [], {"lds": 0, "vgpr": 0}
    for cand in candidates(m, n, k, e):
        w, fp = price(mach, m, n, k, e, out_e, peak_key, peak, *cand)
        if w is None:
            infeasible[fp] += 1
            continue
        t, krows, _, _ = _bound(mach, [("gemm", w)], peak_key, peak)
        rows.append((t, fp, krows[0]))
    if not rows:
        raise SystemExit("no tiling of %dx%dx%d fits %s" % (m, n, k, mach.id))
    rows.sort(key=lambda r: r[0])
    t, fp, kr = rows[0]
    return {"T_us": t * 1e6, "tiling": fp, "binding": kr["binding"], "terms_us": kr["terms_us"],
            "terms_class": kr["terms_class"], "candidates": len(rows) + sum(infeasible.values()),
            "feasible": len(rows), "infeasible": infeasible,
            "runners_up": [{"T_us": r[0] * 1e6, "tiling": r[1], "binding": r[2]["binding"]} for r in rows[1:top]]}
