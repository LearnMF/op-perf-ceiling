"""SwiGLU MLP block (LLaMA-style), forward + backward, traced on meta tensors.

    python -m opceil trace --machine machines/gfx936-bw1000.json --script examples/torch/swiglu_mlp.py
"""
import torch
import torch.nn.functional as F

TOKENS, HIDDEN, INTER = 4096, 4096, 11008


def mlp(x, w_gate, w_up, w_down):
    y = (F.silu(x @ w_gate) * (x @ w_up)) @ w_down
    y.float().pow(2).mean().backward()
    return y


def build():
    kw = dict(device="meta", dtype=torch.bfloat16, requires_grad=True)
    x = torch.empty(TOKENS, HIDDEN, **kw)
    return mlp, (x, torch.empty(HIDDEN, INTER, **kw), torch.empty(HIDDEN, INTER, **kw),
                 torch.empty(INTER, HIDDEN, **kw))
