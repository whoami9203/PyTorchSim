"""
GEMV (matrix-vector) testcase, shaped after Llama2-7B's decode-time linear
projections.

During autoregressive decode (batch=1, one new token per step), every
nn.Linear in the model multiplies its (out_features, in_features) weight
against a single activation *vector*, not a batch of rows -- i.e. a GEMV,
not a GEMM. Llama2-7B's hidden_size=4096 and intermediate_size=11008 give:

  * q_proj (and k_proj/v_proj/o_proj): weight (4096, 4096)
  * gate_proj (and up_proj):           weight (11008, 4096)

both applied as `y = x @ W.T` with `x` shaped (1, 4096) and no bias
(LlamaConfig's attention_bias/mlp_bias both default to False).

Unlike tests/test_dma_bandwidth.py and tests/test_double_buffering.py, this
is a functional-correctness check like tests/test_matmul.py -- it runs the
GEMV through torch.compile on the NPU simulator, diffs the result against a
plain CPU nn.Linear, and prints the actual output tensor so the result is
visible, not just a pass/fail. Needs pytorchsim_functional_mode: 1 in
TOGSIM_CONFIG (the default fallback config already sets this); under
pytorchsim_functional_mode: 0 (timing-only, e.g. eclab_cambricon.yml) the
NPU output isn't functionally computed and the correctness check will fail
regardless of anything being wrong -- see test_dma_bandwidth.py's note on
the same point.

Correctness is checked via relative L2 norm, not tests/test_matmul.py's
elementwise torch.allclose: verified empirically (in-simulator run, q_proj
shape) that CPU fp16 nn.Linear already matches a fp32 reference to within
0.008 absolute here, so it's a trustworthy reference, but the NPU output
differs from it by up to 0.375 on individual elements while the overall
relative L2 error is only ~0.09%. That gap is fp16-accumulation noise over
a K=4096/11008 reduction landing disproportionately on near-zero output
elements (large *relative* error there even though the result is correct to
within simulated-hardware fp16 precision) -- torch.allclose's atol+rtol*|ref|
check fails on exactly those elements regardless of the overall result being
right, so it isn't a meaningful pass/fail signal at this reduction size.

Usage (from repo root):
    python3 tests/test_gemv.py
"""

import torch


def test_result(name, out, cpu_out, rel_l2_threshold=0.02):
    out_cpu = out.cpu().float()
    ref = cpu_out.float()
    diff = (out_cpu - ref).abs()
    rel_l2 = diff.norm().item() / ref.norm().item()

    print(f"\n[{name}] output shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"[{name}] first 8 values: {out.flatten()[:8].tolist()}")
    print(f"[{name}] mean={out_cpu.mean().item():.6f} std={out_cpu.std().item():.6f}")
    print(f"[{name}] max abs diff vs CPU reference: {diff.max().item():.6f}, relative L2 error: {rel_l2:.4%}")

    if rel_l2 < rel_l2_threshold:
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)


def test_gemv(device, name, in_features, out_features, dtype=torch.float16):
    """y = x @ W.T for a single activation vector x (batch=1), matching a
    Llama2-7B decode-step projection: W has shape (out_features, in_features),
    no bias."""
    def gemv(x, weight):
        return torch.nn.functional.linear(x, weight)

    torch.manual_seed(0)
    x = torch.randn(1, in_features, dtype=dtype)
    weight = torch.randn(out_features, in_features, dtype=dtype)

    x_npu = x.to(device=device)
    weight_npu = weight.to(device=device)

    opt_fn = torch.compile(dynamic=False)(gemv)
    res = opt_fn(x_npu, weight_npu)
    ref = gemv(x, weight)

    print(f"\n=== GEMV: {name} -- weight ({out_features}, {in_features}), "
          f"x (1, {in_features}) ===")
    test_result(name, res, ref)
    return res


if __name__ == "__main__":
    device = torch.device("npu:0")

    # Llama2-7B q_proj (and k_proj/v_proj/o_proj share this shape)
    test_gemv(device, "Llama2-7B q_proj GEMV", in_features=4096, out_features=4096)

    # Llama2-7B gate_proj (up_proj shares this shape)
    test_gemv(device, "Llama2-7B gate_proj GEMV", in_features=4096, out_features=11008)
