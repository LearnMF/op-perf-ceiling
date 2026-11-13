"""Derive the L0 work of a PyTorch function automatically: run it on meta tensors and record every aten op.

No data, no device, no kernel: tensors on device="meta" carry only shape and dtype, so a whole model block can
be traced on a laptop.  Each recorded op gives

  flops   from torch's own flop formulas (torch.utils.flop_counter.flop_registry: mm / bmm / addmm / conv /
          scaled-dot-product attention ...).  Ops without a formula (elementwise, reductions, copies) are
          counted as 0 flops -- they are memory-bound, and their bytes are still counted;
  bytes   every input tensor read once + every output written once (views move no data and are skipped).

Two works come out of one trace:

  eager   one kernel per op, back to back: the floor of running the function op by op (what eager PyTorch
          does), so the gap eager -> fused is what fusion can at most recover;
  fused   the whole function as one kernel: total flops, and only the function's own inputs and outputs
          crossing HBM.  This is T_theory of the operator the function defines.

What this cannot give, by construction: the on-chip working set, dependency chains between steps, and the
feasible kernel splits -- those are exactly what a hand-written plugin adds (see docs/plugin-guide.md).
Custom ops registered with torch.library are recorded but have no flop formula (reported as unknown).  Kernels
launched directly from Python (most Triton code, e.g. FLA) bypass the dispatcher and cannot run on meta tensors
at all: trace the operator's pure-PyTorch reference implementation instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ops import Work


@dataclass
class OpRecord:
    op: str
    flops: float
    bytes_in: float
    bytes_out: float
    dtype: str
    out_shape: tuple


@dataclass
class Trace:
    ops: list = field(default_factory=list)
    input_bytes: float = 0.0
    output_bytes: float = 0.0
    unknown: list = field(default_factory=list)


def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


def _nbytes(t) -> float:
    """Bytes a kernel must touch for `t`: broadcast (stride-0) dimensions hold one element, not `size`."""
    n = 1
    for size, stride in zip(t.shape, t.stride()):
        if stride != 0:
            n *= size
    return float(n * t.element_size())


def trace(fn, *args, **kwargs) -> Trace:
    """Run `fn(*args, **kwargs)` (tensors on device="meta") and record its aten ops, forward and -- if `fn`
    calls .backward() -- backward.  Leaf inputs with requires_grad count their .grad as an output."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import flop_registry

    tr = Trace()
    skip = {torch.ops.aten.detach, torch.ops.aten.empty, torch.ops.aten.empty_like, torch.ops.aten.zeros,
            torch.ops.aten.zeros_like, torch.ops.aten.ones_like, torch.ops.aten.lift_fresh,
            torch.ops.aten.alias}

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), kw=None):
            kw = kw or {}
            out = func(*a, **kw)
            packet = func._overloadpacket
            if getattr(func, "is_view", False) or packet in skip:
                return out
            ins = list(_tensors(a)) + list(_tensors(kw))
            outs = list(_tensors(out))
            if not outs:
                return out
            fl = float(flop_registry[packet](*a, **kw, out_val=out)) if packet in flop_registry else 0.0
            name = str(packet).replace("aten.", "")
            if func.namespace not in ("aten", "prims"):
                tr.unknown.append(str(func))
            tr.ops.append(OpRecord(op=name, flops=fl, bytes_in=sum(_nbytes(t) for t in ins),
                                   bytes_out=sum(_nbytes(t) for t in outs), dtype=str(outs[0].dtype),
                                   out_shape=tuple(outs[0].shape)))
            return out

    leaves = [t for t in list(_tensors(args)) + list(_tensors(kwargs))]
    with Rec():
        res = fn(*args, **kwargs)
    tr.input_bytes = sum(_nbytes(t) for t in leaves)
    grads = [t.grad for t in leaves if t.requires_grad and t.grad is not None]
    tr.output_bytes = sum(_nbytes(t) for t in list(_tensors(res)) + grads)
    return tr


def to_work(tr: Trace, mode: str) -> Work:
    """`eager`: one kernel per recorded op, back to back.  `fused`: the whole trace as one kernel."""
    notes = ["derived from a torch meta trace: %d aten ops; flops only for ops with a torch flop formula, "
             "the rest counted as memory traffic" % len(tr.ops)]
    if tr.unknown:
        notes.append("ops outside aten (custom kernels) were recorded without a flop formula: %s"
                     % ", ".join(sorted(set(tr.unknown))))
    if mode == "eager":
        parts = [("%03d %s" % (i, o.op), Work(flops=o.flops, bytes_min=o.bytes_in + o.bytes_out))
                 for i, o in enumerate(tr.ops)]
        w = Work.sequence(parts)
    elif mode == "fused":
        w = Work(flops=sum(o.flops for o in tr.ops), bytes_min=tr.input_bytes + tr.output_bytes)
    else:
        raise ValueError(mode)
    w.notes += notes
    return w


def summary(tr: Trace) -> list:
    """Ops grouped by name: count, flops, bytes -- what dominates the eager work."""
    agg = {}
    for o in tr.ops:
        a = agg.setdefault(o.op, {"op": o.op, "count": 0, "flops": 0.0, "bytes": 0.0})
        a["count"] += 1
        a["flops"] += o.flops
        a["bytes"] += o.bytes_in + o.bytes_out
    return sorted(agg.values(), key=lambda a: -(a["bytes"] + a["flops"]))
