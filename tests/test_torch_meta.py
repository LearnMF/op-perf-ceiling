"""The torch meta front end derives the same L0 work a hand-written plugin states."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from opceil.frontends.torch_meta import to_work, trace  # noqa: E402
from opceil.model.predict import TheoryMachine, predict  # noqa: E402
from opceil.ops import get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MACH = TheoryMachine.load(str(ROOT / "machines" / "gfx936-bw1000.json"))
META = dict(device="meta", dtype=torch.bfloat16)


def test_gemm_fused_work_matches_the_plugin():
    m, n, k = 4096, 2048, 6144
    tr = trace(lambda a, b: a @ b, torch.empty(m, k, **META), torch.empty(k, n, **META))
    w, ref = to_work(tr, "fused"), get("gemm").work({"M": m, "N": n, "K": k}, "bf16", "fwd")
    assert w.flops == ref.flops and w.bytes_min == ref.bytes_min


def test_gemm_backward_is_traced():
    m, n, k = 1024, 2048, 512
    a = torch.empty(m, k, requires_grad=True, **META)
    b = torch.empty(k, n, requires_grad=True, **META)

    def f(a, b):
        y = a @ b
        y.backward(torch.empty_like(y))
        return y
    tr = trace(f, a, b)
    assert sum(o.flops for o in tr.ops) == 3 * 2 * m * n * k
    assert sum(1 for o in tr.ops if o.op == "mm") == 3


def test_attention_flops_match_the_plugin():
    b, h, s, d = 2, 8, 1024, 64
    q, k, v = (torch.empty(b, h, s, d, **META) for _ in range(3))
    tr = trace(F.scaled_dot_product_attention, q, k, v)
    ref = get("attention").work({"B": b, "H": h, "S": s, "D": d}, "bf16", "fwd")
    assert to_work(tr, "fused").flops == pytest.approx(ref.flops)


def test_broadcast_inputs_are_not_counted_at_their_virtual_size():
    x = torch.empty(4096, 4096, **META)
    g = torch.empty(4096, **META)
    tr = trace(lambda x, g: x * g, x, g)
    assert tr.ops[0].bytes_in == (4096 * 4096 + 4096) * 2


def test_eager_is_never_below_fused_and_example_script_runs():
    from opceil.frontends.tcli import run
    r = run(MACH, str(ROOT / "examples" / "torch" / "swiglu_mlp.py"), "bf16")
    assert r["eager"]["T_bound_us"] >= r["fused"]["T_bound_us"] > 0
    assert r["fused"]["work"]["flops"] == pytest.approx(9 * 2 * 4096 * 4096 * 11008)


def test_eager_and_fused_price_through_predict():
    m = 2048
    tr = trace(lambda x: F.silu(x) * x, torch.empty(m, m, **META))
    e = predict(MACH, "t", {}, "bf16", "fwd", work=to_work(tr, "eager"))
    f = predict(MACH, "t", {}, "bf16", "fwd", work=to_work(tr, "fused"))
    assert e["op"]["kernels"] == 2 and f["op"]["kernels"] == 1
    assert e["T_bound_us"] > f["T_bound_us"]
