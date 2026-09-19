"""
Model-agnostic DRAM-layout fix for decode-time GEMV: nn.Linear.forward is
`F.linear(input, self.weight, self.bias)`, which computes `input @ self.weight.T`
-- reading `weight` through an implicit transpose against its real
(out_features, in_features) checkpoint layout. On the NPU simulator this
fragments each DMA burst down to TILE_K elements instead of a full contiguous
row, which measured at Llama2-7B's q_proj/gate_proj GEMV shapes (M=1) as a
~2-2.5x drop in achieved DRAM bandwidth and a collapse in DRAM row-buffer hit
rate (33-98% -> 17-34%). See the conversation tests/Llama/test_llama2_7B.py's
TransposedLinear came out of, and tests/GPT/test_GPT_NeoX_20B.py for a second
model this was ported to.

Usage: for any model built on the meta device, call
`replace_linear_with_transposed_(model_or_submodule)` once, then have your
streamed checkpoint loader transpose (`.t().contiguous()`) each
TransposedLinear.weight tensor right after reading it off disk, before it
reaches the device -- see StreamedCheckpointLoader.load_module_ in either
tests/Llama/test_llama2_7B.py or tests/GPT/test_GPT_NeoX_20B.py for the
pattern (not shared here, since each model's checkpoint loader is already its
own independent class).
"""

import torch


class TransposedLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear whose `weight` is stored (in_features,
    out_features) instead of nn.Linear's (out_features, in_features).

    Storing `weight` pre-transposed and computing `input @ self.weight + self.bias`
    (a contiguous-access matmul) restores the good access pattern. The tensor still
    has to be physically transposed once somewhere -- do it in your checkpoint
    loader, on host, outside any torch.compile'd region (doing it live inside a
    compiled forward instead measured as strictly worse: an extra ~5x-cycle,
    low-bandwidth transpose kernel that didn't even make the following matmul
    faster).

    Uses plain `matmul` + a separate bias-add (not `addmm`) so `forward` also
    works for the 3D (batch, seq, hidden) inputs real attention/MLP modules
    actually pass, which `addmm` (2D operands only) can't handle -- the
    DRAM-layout benefit comes from `weight`'s storage order, not from which op
    does the multiply.
    """

    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features, dtype=dtype, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linear_with_transposed_(module):
    """Recursively replaces every nn.Linear submodule of `module`, in place, with a
    (still-empty/meta) TransposedLinear of matching shape -- your checkpoint loader
    materializes its weight (pre-transposed) later. See TransposedLinear's docstring."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, child_name, TransposedLinear(
                child.in_features, child.out_features, bias=child.bias is not None,
                dtype=child.weight.dtype, device=child.weight.device,
            ))
        else:
            replace_linear_with_transposed_(child)
