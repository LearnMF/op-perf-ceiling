"""TEMPLATE -- copy to opceil/ops/<name>.py, rename, fill in, register in opceil/ops/__init__.py:registry().

<One paragraph: what the operator computes, in math, and which library / reference implementation defines its
semantics (e.g. "FLA chunk_kda", "torch.nn.functional.scaled_dot_product_attention").>

shape: <every key --shape takes, with its meaning and default>.

Checklist (docs/plugin-guide.md section 2) -- answer each from the algorithm, never from a measured kernel:
  1. required flops ........ matrix multiply-adds x 2, at the compute peak of the dtype
  2. required HBM bytes .... inputs read once + outputs written once (+ tensors the backward must find saved)
  3. other unavoidable work  transcendentals (sfu_exp2), lane-ops that cannot run on the matrix unit
  4. serial structure ...... dependent steps that cannot overlap (scans / recurrences), or none
  5. (L1 only) capacity graph: ops, tensors between them, tile bytes per tile of work -- see kda.py
"""
from __future__ import annotations

from . import Op, Work, dtype_bytes


class Template(Op):
    name = "template"                 # the --op name
    params = ("M", "N")               # required --shape keys; a tuple entry ("H", "heads") means one of them
    phases = ("fwd",)                 # add "bwd", "fwd+bwd" only if the backward is modelled too

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        self.check(shape, phase)      # refuses missing keys / unmodelled phases instead of guessing
        m, n = int(shape["M"]), int(shape["N"])
        e = dtype_bytes(dtype)
        flops = 0.0                   # 1. required flops
        bytes_min = float(m * n * e)  # 2. required HBM bytes
        w = Work(flops=flops, bytes_min=bytes_min)
        # 3. w.other["sfu_exp2"] = <exp count>;  if a count rests on an implementation assumption, also
        #    w.other_model = ("<key>",) so it is kept out of the provable T_bound.
        # 4. w.serial = [(steps, "<machine structural.recurrence_step_us key>", ctas)]
        w.notes.append("say here what the bytes assume (e.g. 'each operand crosses HBM once')")
        return w


OPS = ()   # (Template(),) once it is real -- the template itself is not registered
