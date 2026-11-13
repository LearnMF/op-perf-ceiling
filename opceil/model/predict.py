"""Shape-only prediction of an operator's performance ceiling.

Same bound structure as kra.model.bounds in kernel-roofline-analysis-tools, but every term is computed from an operator
formula instead of from a measured window, so no run on hardware is needed:

  T_flop   required flops at the compute peak of the dtype
  T_mem    required bytes (inputs read once, outputs written once) at attainable HBM bw
  T_other  per-resource required work (e.g. transcendentals) at that resource's peak
  T_fixed  kernels x dispatch cost -- the floor small shapes hit

Per kernel the bound is max(all of them): every term is a lower bound on time, and the
largest binds.  Kernels that run back to back on one stream add.  The binding term of
the largest share is the bottleneck; the report also states each other resource's
utilisation at T_theory, which is what says how much a different split could recover.

T_theory is a *lower bound on time*, not an achievable runtime: see docs/devlog/requirements-qa.md.
"""
from __future__ import annotations

import json

from ..ops import DTYPE_PEAK, Work, get

RESOURCE_UNITS = {"TFLOPS": 1e12, "TOPS": 1e12, "Gop/s": 1e9, "GB/s": 1e9}


class TheoryMachine:
    """Peaks and fixed costs of one device, in SI units per second.

    `peaks="theoretical"` uses the datasheet number where the machine file has one and
    falls back to the measured value otherwise; `peaks="measured"` uses the calibrated
    microbenchmark value.  Both are reported so a prediction always says which it used.
    """

    def __init__(self, m: dict, peaks: str = "measured", occupancy: bool = True, tiling: bool = True):
        self.raw = m
        self.use_occupancy = occupancy
        self.use_tiling = tiling
        self.which = peaks
        self.id = m.get("id") or m.get("device", {}).get("name", "?")
        self.device = m.get("device", {})
        self.peaks = {}
        # T_bound needs a rate no kernel can exceed.  A measured peak is the best a microbenchmark reached,
        # i.e. a LOWER estimate of the true peak (a vendor GEMM streamed 1366 GB/s on a card whose HBM
        # microbenchmark peaked at 1339), so the provable terms use the datasheet value where the machine
        # file has one, whatever --peaks says; keys without one fall back to the measured value and are listed.
        self.peaks_bound, self.bound_fallback = {}, []
        for k, v in m.get("peaks", {}).items():
            if not isinstance(v, dict):
                continue
            unit = v.get("unit", "")
            scale = next((s for u, s in RESOURCE_UNITS.items() if unit.startswith(u)), None)
            if scale is None or v.get("value") is None and v.get("theoretical") is None:
                continue
            meas, theo = v.get("value"), v.get("theoretical")
            val = theo if (peaks == "theoretical" and theo is not None) else (meas if meas is not None else theo)
            self.peaks[k] = float(val) * scale
            if theo is not None:
                self.peaks_bound[k] = max(float(theo), float(meas or 0.0)) * scale
            else:
                self.peaks_bound[k] = float(meas) * scale
                self.bound_fallback.append(k)
        hbm_keys = ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy")
        self.bw = max((self.peaks[k] for k in hbm_keys if k in self.peaks), default=0.0)
        self.bw_bound = max((self.peaks_bound[k] for k in hbm_keys if k in self.peaks_bound), default=0.0)
        launch = m.get("launch", {}).get("launch_stream", {})
        # a kernel costs at least one dispatch interval and at least the shortest device duration measured
        # for a trivial kernel (`launch.kernel_min`, bench/kernel_floor.py), whichever is larger
        self.launch_us = max(float(launch.get("best_us", 0.0)),
                             float(m.get("launch", {}).get("kernel_min", {}).get("best_us", 0.0)))
        self.ridge = m.get("ridge_points_flop_per_byte", {})
        self.cus = int(self.device.get("cu_count", 0))
        self.clk = float(self.device.get("clock_khz", 0)) * 1e3
        if self.cus and self.clk:
            # one lane-op per lane per cycle; packed fp32 pairs are the plugin's business
            self.peaks.setdefault("valu_lane", self.cus * int(self.device.get("warp_size", 64)) * self.clk)
        self.structural = {k: v for k, v in m.get("structural", {}).items() if not k.startswith("_")}
        self.latency_s = {k: float(v["ns"]) * 1e-9 for k, v in m.get("latency", {}).items()
                          if isinstance(v, dict) and "ns" in v}
        # measured attainable bandwidth as a function of how many CTAs are resident (streaming
        # microbenchmark): the ceiling a kernel with that much parallelism can reach.
        self.bw_curves = {k: {int(c): float(b) for c, b in v["curve"].items()}
                          for k, v in m.get("peaks", {}).items()
                          if isinstance(v, dict) and isinstance(v.get("curve"), dict)}

    def bw_at_ctas(self, key: str, ctas: int) -> float | None:
        """Attainable bandwidth of `key` with `ctas` CTAs resident, interpolated on the measured curve;
        None when the machine file has no curve for it.

        The curve gives the SHAPE (how fast bandwidth saturates with resident CTAs); its magnitude is scaled
        so that saturation equals the streaming peak (`peaks.hbm_read`): the curve's own microbenchmark is
        one kernel, and a better one reaches the peak with the same parallelism."""
        cur = self.bw_curves.get(key)
        if not cur:
            return None
        xs = sorted(cur)
        top = max(cur.values())
        scale = (self.peaks.get("hbm_read", top * 1e9) / 1e9) / top if top else 1.0
        if ctas >= xs[-1]:
            return cur[xs[-1]] * scale * 1e9
        if ctas <= xs[0]:
            return cur[xs[0]] * scale * 1e9 * ctas / xs[0]
        hi = next(x for x in xs if x >= ctas)
        lo = max(x for x in xs if x <= ctas)
        f = 0.0 if hi == lo else (ctas - lo) / (hi - lo)
        return (cur[lo] + f * (cur[hi] - cur[lo])) * scale * 1e9

    def concurrency(self, occ: dict) -> dict:
        """What one kernel can keep in flight on this machine, from the hardware limits only.

        Resident waves per CU is the smallest of what LDS, the register file, the CU's wave slots and
        the grid allow.  Whatever of the register file those waves do not hold live is the deepest
        prefetch they could run, which is an *upper* bound on the bytes in flight -- so the latency it
        gives is a lower bound on time.  Register-hungry, low-residency kernels get a smaller one."""
        dev = self.device
        warp = int(dev.get("warp_size", 64) or 64)
        thr = int(occ.get("threads") or warp)
        per_cta = max(1, thr // warp)
        regs_per_cu = float(dev.get("regs_per_block") or 0)      # dwords of the CU register file
        lds_per_cu = float(dev.get("lds_per_cu") or dev.get("lds_per_block") or 0)
        vgpr = max(1.0, float(occ.get("vgpr") or 1.0))
        lim, why = {}, []
        if lds_per_cu and occ.get("lds_bytes"):
            lim["lds"] = int(lds_per_cu // occ["lds_bytes"]) * per_cta
        if regs_per_cu:
            lim["vgpr"] = int(regs_per_cu // (vgpr * warp))
        if dev.get("max_threads_per_cu"):
            lim["slots"] = int(float(dev["max_threads_per_cu"]) // warp)
        if self.cus and occ.get("ctas"):
            lim["grid"] = max(1, -(-int(occ["ctas"]) // self.cus)) * per_cta
        if not lim or not regs_per_cu:
            return {}
        waves = max(1, min(lim.values()))
        binding = min(lim, key=lim.get)
        # Deepest prefetch those waves could run: what the register file leaves them, and never more
        # than the architectural per-thread register limit a single wave can address.
        free = min(regs_per_cu / (warp * waves), float(dev.get("max_vgpr_per_thread") or 256)) - vgpr
        if free < 1.0:
            free, why = 1.0, ["registers are fully live: one dword per lane assumed in flight"]
        return {"waves_per_cu": waves, "occupancy_limit": binding, "ctas_per_cu": waves / per_cta,
                "wave_slots_pct": 100.0 * waves / max(1.0, lim.get("slots", waves)),
                "cus_used": int(min(int(occ.get("ctas") or 0) or self.cus, self.cus)) if self.cus else 0,
                "ctas_resident": int(min(int(occ.get("ctas") or 0) or 10 ** 9,
                                         self.cus * max(1, waves // per_cta))),
                "in_flight_bytes_per_cu": waves * warp * free * 4.0,
                "free_dwords_per_lane": free, "notes": why}

    def barrier_s(self, threads: int) -> float | None:
        """Measured latency of one block-wide barrier at the nearest calibrated block size, or None."""
        cand = {}
        for k, v in (self.raw.get("latency", {}).get("barrier") or {}).items():
            if k.startswith("lds_barrier") and isinstance(v, dict) and "ns" in v:
                cand[int(k.rsplit("_", 1)[1])] = float(v["ns"]) * 1e-9
        if not cand:
            return None
        ks = sorted(cand)
        return cand[next((k for k in ks if k >= threads), ks[-1])]

    def dep_step_s(self, threads: int) -> float | None:
        """Floor on one dependent step of a scan, from measured latencies only: the step's product
        depends on the previous step's, so at least one dependent MMAC latency is exposed, and the
        state crossing LDS between steps costs at least one barrier.  One of each is the *weakest*
        statement the dataflow allows, so it is a lower bound on any step; a machine-and-operator
        specific `recurrence_step_us` (when calibrated) is larger and wins the max."""
        m = self.latency_s.get("mmac_bf16_dependent")
        b = self.barrier_s(threads)
        if m is None and b is None:
            return None
        return (m or 0.0) + (b or 0.0)

    def step_s(self, key: str, ctas: int) -> float | None:
        """Latency of one dependent recurrence step, or None if the machine file lacks it."""
        tab = self.structural.get("recurrence_step_us", {}).get(key)
        if not tab:
            return None
        per_cu = max(1, -(-ctas // self.cus)) if self.cus else 1
        return float(tab.get("%dcu" % per_cu, tab[max(tab)])) * 1e-6

    @classmethod
    def load(cls, path: str, peaks: str = "measured", occupancy: bool = True) -> "TheoryMachine":
        return cls(json.loads(open(path, encoding="utf-8").read()), peaks, occupancy)

    def bound_fallback_for(self, keys) -> list:
        """Which of these peaks the provable terms had to take from a measurement (no datasheet value)."""
        return sorted(k for k in keys if k in self.bound_fallback)

    def compute_peak(self, dtype: str) -> tuple:
        key = DTYPE_PEAK.get(dtype)
        if key is None:
            raise SystemExit("dtype %r has no compute peak mapping" % dtype)
        if key not in self.peaks:
            raise SystemExit(
                "machine %s has no %s peak, so %s cannot be predicted on it; add the peak "
                "to the machine file (measured or datasheet) first." % (self.id, key, dtype))
        return key, self.peaks[key]


# How far each term can be trusted.  A term is only as strong as its weakest input:
#   bound       required work fixed by the algorithm / the plugin's dataflow, over a peak no kernel exceeds
#               (or a generic measured latency every step of a declared dependency chain must expose)
#   model       depends on an assumption about how the kernel is built or scheduled: occupancy, latency
#               hiding, barrier exposure, LDS staging of a chosen tiling, separate read/write peaks
#   calibrated  uses a machine constant measured with an operator-shaped microbenchmark
#               (structural.mmac_tile_eff / vmem_cycles_per_kb / recurrence_step_us)
TERM_CLASSES = ("bound", "model", "calibrated")
_RANK = {c: i for i, c in enumerate(TERM_CLASSES)}


def weakest(*classes: str) -> str:
    """The class of a quantity built from several terms: the weakest of them."""
    return max(classes or ("bound",), key=lambda c: _RANK[c])


def _terms(mach: TheoryMachine, w: Work, peak_key: str, peak: float, strict: bool = False,
           cls: dict | None = None) -> dict:
    """Per-resource lower-bound terms of one kernel, in seconds.

    `strict` keeps only the terms of class `bound`, each at its provable value (no tile efficiency, no
    occupancy scaling, the best HBM stream bandwidth, the generic dependent-step floor): their max is a
    lower bound no implementation can beat under the machine file's peaks.  `cls`, when given, is filled
    with the class of every term returned."""
    t, c = {}, {}
    con = {} if strict else (mach.concurrency(w.occupancy) if (w.occupancy and mach.use_occupancy) else {})
    # A kernel with fewer tiles of work than the device has CUs can only occupy that many CUs, so the
    # per-CU resources (matrix and vector pipes, LDS, memory issue) are available to it in that
    # fraction only.  Device-level resources (HBM, L2) are not scaled this way; HBM gets the measured
    # bandwidth-vs-CTAs curve instead.
    cu_frac = min(1.0, con["cus_used"] / mach.cus) if con.get("cus_used") and mach.cus else 1.0
    scaled = "model" if cu_frac < 1.0 else "bound"
    pk = mach.peaks_bound if strict else mach.peaks
    if strict:
        peak = mach.peaks_bound.get(peak_key, peak)
    if w.flops:
        use_eff = (not strict) and bool(w.compute_eff) and w.compute_eff in mach.structural
        eff = float(mach.structural[w.compute_eff]) if use_eff else 1.0
        t["compute:" + peak_key] = w.flops / (peak * eff * cu_frac)
        c["compute:" + peak_key] = weakest(scaled, "calibrated" if use_eff else "bound")
    if w.bytes_min or w.bytes_rd:
        if not mach.bw:
            raise SystemExit("machine %s has no HBM bandwidth peak" % mach.id)
        if (not strict) and w.bytes_rd is not None and "hbm_read" in mach.peaks and "hbm_write" in mach.peaks:
            # A kernel with too few resident CTAs cannot reach the streaming peak even in principle:
            # charge its reads at the bandwidth measured at that CTA count.  Writes are left at the
            # peak, because no write-vs-CTA curve was measured -- under-charging keeps the bound below.
            bw_rd = mach.peaks["hbm_read"]
            if con.get("ctas_resident"):
                at = mach.bw_at_ctas("hbm_read_vs_ctas", con["ctas_resident"])
                if at:
                    bw_rd = min(bw_rd, at)
            t["memory:hbm"] = w.bytes_rd / bw_rd + (w.bytes_wr or 0.0) / mach.peaks["hbm_write"]
            # reads and writes serialised at their own peaks, and the CTA curve: assumptions
            c["memory:hbm"] = "model"
        else:
            t["memory:hbm"] = (w.bytes_min or (w.bytes_rd or 0.0) + (w.bytes_wr or 0.0)) / (
                mach.bw_bound if strict else mach.bw)
            c["memory:hbm"] = "bound"
        # multi-level roofline: the same required-bytes argument one level in.  Every HBM byte
        # also crosses L2, so bytes_l2 defaults to the HBM traffic; LDS traffic has to be declared
        # by the operator (staged tiles) or the level is left out.
        lvl = {"l2": (w.bytes_l2 if w.bytes_l2 is not None else w.bytes_min, "l2_read", "bound"),
               "lds": (w.bytes_lds, "lds_read", "model")}
        for name, (b, key, k) in lvl.items():
            if strict and k != "bound":
                continue
            if b and key in pk:
                t["memory:" + name] = b / (pk[key] * (cu_frac if name == "lds" else 1.0))
                c["memory:" + name] = k
        vm = mach.structural.get("vmem_cycles_per_kb")
        if (not strict) and w.vmem_issue and vm and mach.cus and w.bytes_rd is not None:
            t["issue:vmem"] = (w.bytes_rd / 1024 * vm["read"] + (w.bytes_wr or 0.0) / 1024 * vm["write"]) \
                / (mach.cus * cu_frac * mach.clk)
            c["issue:vmem"] = "calibrated"
    if con.get("in_flight_bytes_per_cu") and mach.cus and w.bytes_rd and mach.latency_s.get("gmem_hbm_ws"):
        # Little's law the other way round: the bytes a CU must read, at the rate its own in-flight
        # window and the measured memory latency allow.  Independent of bandwidth: it is what the
        # kernel's own parallelism cannot hide.
        t["latency:hbm"] = (w.bytes_rd / mach.cus) * mach.latency_s["gmem_hbm_ws"] \
            / con["in_flight_bytes_per_cu"]
        c["latency:hbm"] = "model"
    for res, amount in w.other.items():
        est = res in w.other_model
        if strict and est:
            continue
        if res in pk:
            t["other:" + res] = amount / (pk[res] * cu_frac)
            c["other:" + res] = weakest(scaled, "model" if est else "bound")
    if w.barriers and con.get("ctas_per_cu"):
        # Every value a kernel passes between its ops through LDS (or spills and reads back) costs the
        # whole block a barrier before the consumer may read it, once per tile of work.  The waves of
        # one CTA cannot overlap their own barrier; other CTAs resident on the same CU can, so the
        # exposed time is the per-CU tile batches, not every tile.
        n, bthr = w.barriers
        lat = mach.barrier_s(int(bthr))
        if lat and mach.cus:
            tiles_per_cu = max(1.0, -(-int(w.occupancy["ctas"]) // mach.cus))
            t["sync:barrier"] = tiles_per_cu / max(1.0, con["ctas_per_cu"]) * n * lat
            c["sync:barrier"] = "model"
    if w.serial:
        thr = int((w.occupancy or {}).get("threads") or mach.device.get("warp_size", 64) or 64)
        floor = mach.dep_step_s(thr)
        ch, missing, generic, calibrated = 0.0, [], [], False
        for steps, key, ctas in w.serial:
            s = None if strict else mach.step_s(key, ctas)
            if s is None and floor is None:
                missing.append(key)
                continue
            if s is None:
                generic.append(key)
            elif s > (floor or 0.0):
                calibrated = True
            ch += steps * max(s or 0.0, floor or 0.0)
        if not strict:
            if missing:
                w.notes.append("machine %s has no recurrence_step_us for %s and no measured dependent-MMAC "
                               "or barrier latency: the serial chain is NOT in this bound"
                               % (mach.id, ",".join(sorted(set(missing)))))
            if generic:
                w.notes.append("machine %s has no recurrence_step_us for %s: the serial chain uses the "
                               "generic floor of one dependent MMAC + one barrier per step (%.0f ns), which "
                               "is looser than a calibrated step" % (mach.id, ",".join(sorted(set(generic))),
                                                                     (floor or 0.0) * 1e9))
        if ch:
            t["serial:chain"] = ch
            c["serial:chain"] = "calibrated" if calibrated else "bound"
    t["fixed:launch"] = w.kernels * mach.launch_us * 1e-6
    c["fixed:launch"] = "bound"
    if not strict:
        for k, v in w.extra_terms.items():
            t[k] = v
            c[k] = "model"
    if cls is not None:
        cls.update(c)
    return t


def _bound(mach: TheoryMachine, parts: list, peak_key: str, peak: float, strict: bool = False) -> tuple:
    """Kernels back to back on one stream: each bounded by its own binding term, summed."""
    rows, by_binding, terms = [], {}, {}
    for name, pw in parts:
        c = {}
        t = _terms(mach, pw, peak_key, peak, strict=strict, cls=c)
        b = max(t, key=t.get)
        rows.append({"kernel": name, "binding": b, "binding_class": c[b], "T_us": t[b] * 1e6,
                     "terms_us": {k: v * 1e6 for k, v in t.items()}, "terms_class": c})
        by_binding[b] = by_binding.get(b, 0.0) + t[b]
        for k, v in t.items():
            terms[k] = terms.get(k, 0.0) + v
    return sum(by_binding.values()), rows, by_binding, terms


def predict(mach: TheoryMachine, op_name: str, shape: dict, dtype: str, phase: str,
            work: Work | None = None) -> dict:
    """`work` given: price that work under `op_name` instead of asking the plugin (front ends use this)."""
    w: Work = work if work is not None else get(op_name).work(shape, dtype, phase)
    peak_key, peak = mach.compute_peak(dtype)

    # A sequence of kernels on one stream: each is bounded by its own binding term and
    # the bounds add.  A single kernel is the one-part case.
    t_theory, rows, by_binding, terms = _bound(mach, w.parts or [(op_name, w)], peak_key, peak)
    # The same work with only the provable terms: what no implementation can beat under these peaks.
    t_bound, brows, bbind, _ = _bound(mach, w.parts or [(op_name, w)], peak_key, peak, strict=True)
    # Only the binding term of each kernel sets T_bound, so only its peak has to be a datasheet value; a
    # non-binding term priced at a (lower) measured peak is overstated but does not move the bound.
    used = set()
    for row in brows:
        b = row["binding"]
        used |= ({peak_key} if b.startswith("compute:") else
                 {"hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy"} if b == "memory:hbm" else
                 {"l2_read"} if b == "memory:l2" else {b.split(":", 1)[1]} if b.startswith("other:") else set())
    hbm = {"hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy"}
    fallback = mach.bound_fallback_for(used - hbm) + (
        ["hbm"] if used & hbm and not (used & hbm) - set(mach.bound_fallback) else [])
    if fallback:
        w.notes.append("T_bound uses MEASURED peaks for %s (the machine file has no datasheet value): a kernel can "
                       "beat it by however far the calibration microbenchmark fell short of the true peak"
                       % ", ".join(fallback))
    term_class = {}
    for row in rows:
        for k, c in row["terms_class"].items():
            term_class[k] = weakest(term_class.get(k, "bound"), c)
    theory_class = weakest(*(row["binding_class"] for row in rows))
    splits = {}
    for sname, sparts in w.splits.items():
        ts, srows, sbb, _ = _bound(mach, sparts, peak_key, peak)
        for _, pw in sparts:
            w.notes += [n for n in pw.notes if n not in w.notes]
        splits[sname] = {"T_split_us": ts * 1e6, "O_handoff": ts / t_theory if t_theory else None,
                         "kernels": srows,
                         "bytes": sum((pw.bytes_rd or 0) + (pw.bytes_wr or 0) or pw.bytes_min for _, pw in sparts),
                         "binding_share": {k: v / ts for k, v in sbb.items()} if ts else {}}
    if work is None and hasattr(get(op_name), "gemm_parts") and mach.use_tiling:
        from . import tiling
        tiled = [(name,) + (tiling.search(mach, m_, n_, k_, e_, oe_, peak_key, peak),)
                 for name, m_, n_, k_, e_, oe_ in get(op_name).gemm_parts(shape, dtype, phase)]
        ts = sum(r["T_us"] for _, r in tiled) * 1e-6
        share = {}
        for _, r in tiled:
            share[r["binding"]] = share.get(r["binding"], 0.0) + r["T_us"] * 1e-6
        splits["tiled"] = {"T_split_us": ts * 1e6, "O_handoff": ts / t_theory if t_theory else None,
                           "kernels": [{"kernel": name, "binding": r["binding"], "T_us": r["T_us"],
                                        "tiling": r["tiling"], "terms_us": r["terms_us"],
                                        "terms_class": r["terms_class"]} for name, r in tiled],
                           "bytes": w.bytes_min,
                           "binding_share": {k: v / ts for k, v in share.items()} if ts else {},
                           "class": "model"}
    binding = max(by_binding, key=by_binding.get)
    oi = w.flops / w.bytes_min if w.bytes_min else float("inf")
    ridge = peak / mach.bw if mach.bw else float("inf")
    # hierarchical roofline: where the operator sits against each level's roof.  The traffic of a
    # level the operator does not bound is left out rather than guessed.
    levels = {}
    for name, key, b in (("hbm", "hbm_read", w.bytes_min),
                         ("l2", "l2_read", w.bytes_l2 if w.bytes_l2 is not None else w.bytes_min),
                         ("lds", "lds_read", w.bytes_lds)):
        bw = mach.peaks.get(key)
        if not bw or not b:
            continue
        levels[name] = {"bw_GBps": bw / 1e9, "bytes": b, "intensity_flop_per_byte": w.flops / b,
                        "ridge_flop_per_byte": peak / bw, "T_us": b / bw * 1e6,
                        "bound": "compute" if w.flops / b > peak / bw else "memory"}

    out = {
        "schema": "opceil.predict/1",
        "machine": {"id": mach.id, "peaks": mach.which,
                    "compute_peak_tflops": peak / 1e12, "hbm_bw_GBps": mach.bw / 1e9,
                    "launch_us": mach.launch_us},
        "op": {"name": op_name, "phase": phase, "dtype": dtype, "shape": shape,
               "kernels": w.kernels},
        "work": {"flops": w.flops, "bytes_min": w.bytes_min, "working_set": w.working_set,
                 "other": w.other},
        "arithmetic_intensity_flop_per_byte": oi,
        "levels": levels,
        "ridge_point_flop_per_byte": ridge,
        "terms_us": {k: v * 1e6 for k, v in terms.items()},
        "kernels": rows,
        "binding_share": {k: v / t_theory for k, v in by_binding.items()} if t_theory else {},
        "binding": binding,
        "bottleneck": binding.split(":")[0],
        "T_theory_us": t_theory * 1e6,
        "T_theory_class": theory_class,
        "T_bound_us": t_bound * 1e6,
        "T_bound_binding": max(bbind, key=bbind.get),
        "T_bound_peaks": "datasheet" if not fallback else "datasheet, measured for " + ",".join(fallback),
        "term_class": term_class,
        "T_split": splits,
        "ceiling": {
            "mfu_pct": 100.0 * (w.flops / peak) / t_theory if t_theory else 0.0,
            "bw_util_pct": 100.0 * (w.bytes_min / mach.bw) / t_theory if (t_theory and mach.bw) else 0.0,
        },
        "notes": list(w.notes),
        "disclaimer": "T_bound is a lower bound on time under this machine's peaks (provable terms only). "
                      "T_theory adds model-estimate and calibrated terms, so it is tighter but rests on their "
                      "assumptions; T_split is the best split inside the plugin's search space under the "
                      "capacity estimate.  None of them is a runtime any implementation is guaranteed to reach.",
    }
    out["notes"].insert(0, _why(out))
    return out


def _why(r: dict) -> str:
    oi, ridge = r["arithmetic_intensity_flop_per_byte"], r["ridge_point_flop_per_byte"]
    side = "above" if oi > ridge else "below"
    return ("arithmetic intensity %.1f flop/byte is %s the ridge point %.1f, and %s binds "
            "at %.2f us" % (oi, side, ridge, r["binding"], r["T_theory_us"]))


def markdown(r: dict) -> str:
    o, m = r["op"], r["machine"]
    L = ["# %s %s on %s (%s peaks)" % (o["name"], o["phase"], m["id"], m["peaks"]), "",
         "shape: `%s`  dtype: `%s`  kernels: %d" % (
             ",".join("%s=%s" % kv for kv in o["shape"].items()), o["dtype"], o["kernels"]), "",
         "| quantity | value |", "| --- | ---: |",
         "| required flops | %.3f GFLOP |" % (r["work"]["flops"] / 1e9),
         "| required HBM bytes | %.3f MB |" % (r["work"]["bytes_min"] / 1e6),
         "| arithmetic intensity | %.1f flop/byte |" % r["arithmetic_intensity_flop_per_byte"],
         "| ridge point | %.1f flop/byte |" % r["ridge_point_flop_per_byte"],
         "| **T_bound** (provable terms only) | **%.3f us** (%s) |" % (r["T_bound_us"], r["T_bound_binding"]),
         "| **T_theory** (+ model / calibrated terms) | **%.3f us** (binding term: %s) |" % (
             r["T_theory_us"], r["T_theory_class"]),
         "| bottleneck | %s (%s) |" % (r["bottleneck"], r["binding"]),
         "| ceiling MFU | %.1f%% |" % r["ceiling"]["mfu_pct"],
         "| ceiling HBM utilisation | %.1f%% |" % r["ceiling"]["bw_util_pct"], "",
         "## hierarchical roofline", "",
         "| level | attainable bw | required bytes | intensity | ridge | bytes/bw | side |",
         "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, lv in r.get("levels", {}).items():
        L.append("| %s | %.0f GB/s | %.1f MB | %.1f | %.1f | %.1f us | %s |" % (
            name, lv["bw_GBps"], lv["bytes"] / 1e6, lv["intensity_flop_per_byte"],
            lv["ridge_flop_per_byte"], lv["T_us"], lv["bound"]))
    tc = r.get("term_class", {})
    L += ["",
         "## per-kernel bound", "",
         "term class: `bound` = required work over a peak (provable); `model` = rests on an assumption about "
         "the implementation; `calibrated` = uses an operator-shaped measured constant.", "",
         "| kernel | binding | T_us | " +
         " | ".join("%s [%s]" % (k, tc.get(k, "?")) for k in sorted(r["terms_us"])) + " |",
         "| --- | --- | ---: |" + " ---: |" * len(r["terms_us"])]
    for row in r["kernels"]:
        L.append("| %s | %s | %.3f | " % (row["kernel"], row["binding"], row["T_us"]) +
                 " | ".join("%.3f" % row["terms_us"].get(k, 0.0) for k in sorted(r["terms_us"])) + " |")
    for sname, sp in r.get("T_split", {}).items():
        L += ["", "## T_split (model estimate): decomposition `%s` -- %.3f us (O_handoff = T_split / T_theory = %.2f, "
              "interface bytes %.1f MB = %.2fx required)" % (
                  sname, sp["T_split_us"], sp["O_handoff"], sp["bytes"] / 1e6,
                  sp["bytes"] / max(1.0, r["work"]["bytes_min"])), "",
              "| kernel | binding | T_us |", "| --- | --- | ---: |"]
        for k in sp["kernels"]:
            tl = k.get("tiling")
            L.append("| %s | %s | %.1f |" % (k["kernel"] + (
                " -- tile %dx%dx%d, split-K %d, %d threads, %d stages, %d CTAs, %d waves" % (
                    tl["Mt"], tl["Nt"], tl["Kt"], tl["split_k"], tl["threads"], tl["stages"], tl["ctas"],
                    tl["waves"]) if tl else ""), k["binding"], k["T_us"]))
    L += ["", "## notes", ""] + ["- " + n for n in r["notes"]]
    L += ["", "> " + r["disclaimer"], ""]
    return "\n".join(L)
