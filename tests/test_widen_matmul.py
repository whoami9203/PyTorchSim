import torch

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
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

def test_widen_matmul_fp32(device, input_size=32, hidden_size=32, output_size=32):
    """int8 weight/activation loaded into scratchpad, widened on-chip to fp32,
    then matmul'd. This is the SmoothQuant-style W8A8 GEMM building block:
    the systolic array's compute datapath is float-only (no genuine int32
    accumulator), so int8 operands must be widened to fp32 -- not int32 --
    to get correct results here.
    """
    def custom_fn(a, b):
        return torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.manual_seed(0)
    input = torch.randint(-100, 100, (input_size, hidden_size), dtype=torch.int8)
    weight = torch.randint(-100, 100, (hidden_size, output_size), dtype=torch.int8)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_fn)
    res = opt_fn(x1, w1)
    y = custom_fn(x2, w2)
    test_result("Widen Matmul (int8->fp32) Forward", res, y, rtol=0, atol=0)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_widen_matmul_fp32(device, 32, 32, 32)
