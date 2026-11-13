"""Scaled dot-product attention forward, O = softmax(Q K^T * scale [+ causal mask]) V.  L0 plugin.

This file is also the worked example of docs/plugin-guide.md: it answers only the questions an L0 plugin
must answer (required flops, required HBM bytes, unavoidable transcendentals) and declares nothing about
tiling, so the tool gives T_bound / T_theory and the bottleneck, and refuses --split-search.

shape: B (batch), H (query heads), S (query length), D (head dim); optional Hkv (KV heads, GQA; default H),
Skv (key length, default S), causal (0/1, default 0).
"""
from __future__ import annotations

from . import Op, Work, dtype_bytes


def _pairs(sq: int, sk: int, causal: bool) -> float:
    """(query, key) pairs that are actually computed.  Causal with sq <= sk: query i (aligned to the end of
    the keys) sees sk - sq + i + 1 keys."""
    if not causal:
        return float(sq) * sk
    off = sk - sq
    if off < 0:
        raise SystemExit("attention: causal with Skv < S is not modelled")
    return float(sq) * off + sq * (sq + 1) / 2.0


class Attention(Op):
    name = "attention"
    params = ("B", "H", "S", "D")
    phases = ("fwd",)

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        self.check(shape, phase)
        b, h, s, d = (int(shape[k]) for k in ("B", "H", "S", "D"))
        hkv, skv = int(shape.get("Hkv", h)), int(shape.get("Skv", s))
        causal = bool(int(shape.get("causal", 0)))
        if h % hkv:
            raise SystemExit("attention: H=%d is not a multiple of Hkv=%d" % (h, hkv))
        e = dtype_bytes(dtype)
        pairs = _pairs(s, skv, causal) * b * h
        # Q K^T and P V: one multiply-add per (pair, head-dim element) each.
        flops = 2.0 * 2.0 * pairs * d
        # Q read, K and V read once (shared by the H / Hkv query heads of a group), O written once, plus the
        # fp32 row log-sum-exp a training forward saves for the backward.
        rd = (b * h * s * d + 2.0 * b * hkv * skv * d) * e
        wr = b * h * s * d * e + b * h * s * 4.0
        w = Work(flops=flops, bytes_min=rd + wr, working_set=float(2 * d * e))
        w.other["sfu_exp2"] = pairs                 # one exp per computed score
        w.notes.append("%s attention, %.3g computed (query, key) pairs; bytes = Q + K + V read once, O + lse "
                       "written once (the fused flash-attention floor)" % ("causal" if causal else "full", pairs))
        return w


OPS = (Attention(),)
