"""
DMA/DRAM bandwidth sweep.

Two probe ops, at opposite ends of achievable utilization:

  --op add (default): a purely memory-bound elementwise add (read A, read B,
      write C). Simple, but the codegen does NOT double-buffer scratchpad
      tiles for vector ops, so the DMA engine stalls waiting on the vector
      unit between tiles -- `Core [0] : DMA active_cycles/idle_cycles` in the
      log typically shows only ~25% duty cycle, capping achieved bandwidth
      well below peak regardless of tensor size.

  --op matmul: a "weight-streaming" GEMM (small M/N, large K) shaped so it's
      still memory-bound, but gemm_combination_mapping's tiling IS explicitly
      double-buffered (see mlir_template.py), so the DMA engine stays ~100%
      active, overlapping tile N+1's load with tile N's compute. This alone
      lifts achieved bandwidth substantially over --op add at the same
      config.

Either probe is further capped by TOGSim's per-core DMA issue rate: Core.cc's
dma_cycle() calls `_dma.get_memory_access(core_cycle, icnt_injection_ports_per_core)`
once per core cycle, so at most `icnt_injection_ports_per_core` DRAM requests
can be injected per cycle no matter how wide the interconnect links are
(widening booksim's flit_size alone does *nothing* -- verified empirically).
With the stock eclab.yml (icnt_injection_ports_per_core: 1), a fully-busy DMA
engine saturates around core_freq_mhz * dram_req_size, well under the LPDDR5X
4-channel aggregate peak. configs/eclab_bw.yml raises
icnt_injection_ports_per_core to 4 with a matching booksim topology
(configs/booksim2_configs/fly_c1_m4_p4.icnt, k = num_cores*ports + dram_channels
= 1*4+4 = 8) to relieve that cap.

For highest achieved utilization, combine both: --op matmul with
TOGSIM_CONFIG pointed at eclab_bw.yml.

Usage (from repo root):
    source setup_env_var.sh dma_bw_sweep
    export TOGSIM_CONFIG=$(pwd)/configs/eclab_bw.yml   # optional: relieve the injection-port cap
    bash cleanup_results.sh
    python3 tests/test_dma_bandwidth.py --op matmul --sizes 8192,16384,65536,262144
"""

import argparse
import os
import re
from pathlib import Path

import torch

# Matches Dram.cc's print_stat() final line, e.g.:
#   [DRAM] channels 0..3 combined | 42.17 GB/s aggregate, 61.78% of utilization (avg. per channel) | 123456 reads, 65432 writes
# Deliberately distinct from the periodic "[DRAM] all N channels combined | ..."
# line emitted every dram_stats_print_period_cycles by Dram.cc::cycle() -- that
# one reports a windowed rate, not the whole-run aggregate we want here.
_FINAL_BW_RE = re.compile(
    r"\[DRAM\] channels 0\.\.\d+ combined \| "
    r"([\d.]+) GB/s aggregate, ([\d.]+)% of utilization[^|]*\| "
    r"(\d+) reads, (\d+) writes"
)


def bw_probe_add(a, b):
    return a + b


def bw_probe_matmul(a, b):
    return torch.matmul(a, b)


# Weight-streaming GEMM shape: M and N stay small/fixed (little compute,
# small activation/output DMA); K -- the swept "size" -- is the large
# reduction dim, so nearly all DMA traffic is the K x MATMUL_N weight matrix
# streaming through double-buffered tiles.
MATMUL_M = 1024
MATMUL_N = 1024


def _log_dir():
    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError(
            "TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?"
        )
    return Path(log_dir)


def _parse_final_bandwidth(log_path):
    text = log_path.read_text(errors="replace")
    matches = _FINAL_BW_RE.findall(text)
    if not matches:
        return None
    gbs, util_pct, reads, writes = matches[-1]
    return {
        "gbs": float(gbs),
        "util_pct": float(util_pct),
        "reads": int(reads),
        "writes": int(writes),
    }


def run_bandwidth_probe(device, n, dtype=torch.float16, op="add"):
    # Note: this deliberately does not check the output against a CPU
    # reference. Under the default eclab.yml (pytorchsim_timing_mode: 1,
    # pytorchsim_functional_mode: 0), NPU output values aren't functionally
    # computed -- only DMA/compute timing is modeled -- so an allclose check
    # here would just fail every time regardless of anything being wrong.
    # If you need correctness too, rerun with a functional-mode config.
    torch.manual_seed(0)
    if op == "add":
        a = torch.randn(n, n, dtype=dtype).to(device=device)
        b = torch.randn(n, n, dtype=dtype).to(device=device)
        fn = bw_probe_add
        bytes_moved = 3 * n * n * dtype.itemsize  # read A + read B + write C
    else:
        a = torch.randn(MATMUL_M, n, dtype=dtype).to(device=device)
        b = torch.randn(n, MATMUL_N, dtype=dtype).to(device=device)
        fn = bw_probe_matmul
        bytes_moved = (MATMUL_M * n + n * MATMUL_N + MATMUL_M * MATMUL_N) * dtype.itemsize

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("*.log"))

    opt_fn = torch.compile(dynamic=False)(fn)
    opt_fn(a, b)

    after = set(log_dir.glob("*.log"))
    new_files = sorted(after - before, key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not new_files:
        print(f"[size={n}] No new TOGSim log found in {log_dir} -- is timing mode enabled "
              f"(pytorchsim_timing_mode: 1) in your TOGSIM_CONFIG?")
        return None

    results = []
    for f in new_files:
        bw = _parse_final_bandwidth(f)
        if bw is None:
            print(f"[size={n}] {f.name}: no final DRAM bandwidth line found (kernel may be too "
                  f"small to hit DRAM, or ran in functional-only mode).")
            continue
        results.append((f, bw))

    if not results:
        return None

    # Normally exactly one new log per compiled kernel. If more than one
    # appeared, report all of them so nothing is hidden, and use the one with
    # the most read+write traffic as "the" result for the summary table.
    if len(results) > 1:
        print(f"[size={n}] {len(results)} new TOGSim logs (multiple kernel launches for this call):")
        for f, bw in results:
            print(f"    {f.name}: {bw['gbs']:.2f} GB/s, {bw['util_pct']:.2f}% util, "
                  f"{bw['reads']} reads, {bw['writes']} writes")

    f, bw = max(results, key=lambda item: item[1]["reads"] + item[1]["writes"])
    return {
        "size": n,
        "bytes_moved_mb": bytes_moved / (1024 * 1024),
        "log_file": f.name,
        **bw,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep DMA-bound kernel sizes and report achieved DRAM bandwidth")
    parser.add_argument("--op", type=str, default="matmul", choices=["add", "matmul"],
                         help="add: elementwise NxN (DMA/compute not overlapped, low utilization). "
                              "matmul: weight-streaming GEMM with double-buffered tiling (high utilization).")
    parser.add_argument("--sizes", type=str, default="1024,8192,16384",
                         help="Comma-separated sizes. For --op add, tensors are NxN. "
                              "For --op matmul, this is K (the swept reduction dim) with M/N fixed small.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    dtype = getattr(torch, args.dtype)
    device = torch.device("npu:0")

    rows = []
    for n in sizes:
        row = run_bandwidth_probe(device, n, dtype=dtype, op=args.op)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results collected.")
        return

    size_label = "N" if args.op == "add" else "K"
    print()
    header = f"{size_label:>8} | {'bytes moved (MB)':>16} | {'GB/s':>10} | {'util %':>7} | {'reads':>10} | {'writes':>10} | log"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['size']:>8} | {r['bytes_moved_mb']:>16.2f} | {r['gbs']:>10.2f} | "
              f"{r['util_pct']:>7.2f} | {r['reads']:>10} | {r['writes']:>10} | {r['log_file']}")


if __name__ == "__main__":
    main()
