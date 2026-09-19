"""
GEMV (M=1) N/K sweep: isolates whether N or K (or just total weight size)
drives the low DRAM utilization seen on Llama2-7B's real GEMV shapes.

Context (see tests/test_gemv_vs_gemm_m8.py): a controlled M={1,8,32} sweep
at N=32, K=131072 found M=1 achieved the *highest* DRAM utilization of the
three (91.9%) -- the opposite of what was expected -- which ruled out
"M padding" as the explanation for Llama2-7B q_proj/gate_proj GEMV's low
37-47% utilization. That same script noticed the M-sweep's (N=32, K=131072)
shape needs only 4 K-tile-iterations (MOVIN/2), vs. 16 for q_proj
(N=4096, K=4096) and 32 for gate_proj (N=11008, K=4096) -- and that in all
three cases, weight bytes moved per iteration came out to roughly the same
~2-2.7MB (bounded by the half-scratchpad cap from gemm_combination_mapping),
suggesting iteration count may just track total weight size (K*N), not N or
K individually, and the earlier "shape" framing (small-N-few-iterations vs.
large-N-many-iterations) was confounded with total size (the good M-sweep
case moved only 8MB total vs. 32-86MB for the real GEMV shapes).

This script fixes M=1 and runs two independent sweeps to separate these
candidate explanations:

  Sweep A -- vary N, hold K=4096 fixed (matches Llama2-7B's K):
    N in {32, 128, 512, 2048, 4096, 11008}
    Larger N alone increases total weight size (N*K) AND (since TILE_N=N in
    every case observed so far, i.e. no N-tiling) forces a smaller TILE_K
    to stay under the half-scratchpad cap, which increases iteration count.

  Sweep B -- vary K, hold N=4096 fixed (matches Llama2-7B q_proj's N,
  i.e. one of the "bad" shapes):
    K in {1024, 4096, 16384, 65536, 131072}
    Larger K alone increases total weight size and iteration count too
    (TILE_K stays roughly fixed once N is fixed, so iterations = K/TILE_K
    grows with K) -- but K=131072 at N=32 was the *good* case, so this
    tests whether making K bigger at the *bad* N=4096 helps, hurts, or
    doesn't matter, which should help tell "total size drives iteration
    count drives utilization" apart from "large K provides more
    steady-state streaming time to amortize a fixed startup cost" (which
    would predict recovery at large K even with a small/bad TILE_K).

For every point, this script reports N, K, total weight bytes, K-tile
iteration count, weight bytes moved per iteration, and achieved DRAM
utilization -- so the actual correlation (if any) can be read off the
table instead of assumed. It does not draw the conclusion for you and
does not assert -- this is a diagnostic sweep, not a pass/fail test.

Usage (from repo root):
    source setup_env_var.sh gemv_nk_sweep
    bash cleanup_results.sh
    python3 tests/test_gemv_nk_sweep.py
"""

import os
import re
from pathlib import Path

import torch

_DRAM_FINAL_RE = re.compile(
    r"\[DRAM\] channels 0\.\.\d+ combined \| "
    r"([\d.]+) GB/s aggregate, ([\d.]+)% of utilization[^|]*\| "
    r"(\d+) reads, (\d+) writes"
)
_INST_COUNT_RE = re.compile(r"Core \[0\] : (\w+)\s+inst_count: (\d+)")

M = 1
PRECISION_BYTES = 2  # fp16


def _log_dir():
    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError("TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?")
    return Path(log_dir)


def _run_gemv_probe(N, K, label):
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("*.log"))

    device = torch.device("npu:0")
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16).to(device=device)
    b = torch.randn(K, N, dtype=torch.float16).to(device=device)
    opt_fn = torch.compile(dynamic=False)(torch.matmul)
    opt_fn(a, b)

    after = set(log_dir.glob("*.log"))
    new_files = sorted(after - before, key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not new_files:
        raise RuntimeError(f"[{label}] no new TOGSim log appeared in {log_dir}")

    best = None
    for f in new_files:
        text = f.read_text(errors="replace")
        dram_matches = _DRAM_FINAL_RE.findall(text)
        if not dram_matches:
            continue
        gbs, util_pct, reads, writes = dram_matches[-1]
        total_traffic = int(reads) + int(writes)
        if best is None or total_traffic > best["_traffic"]:
            inst_counts = {opcode: int(count) for opcode, count in _INST_COUNT_RE.findall(text)}
            best = {
                "_traffic": total_traffic,
                "log_file": f.name,
                "dram_gbs": float(gbs),
                "dram_util_pct": float(util_pct),
                "movin": inst_counts.get("MOVIN"),
                "gemm": inst_counts.get("COMP"),
            }

    if best is None:
        raise RuntimeError(f"[{label}] no final DRAM stat line found in {[f.name for f in new_files]}")

    total_weight_bytes = N * K * PRECISION_BYTES
    # 2 MOVIN (X tile + W tile) per K-tile-iteration -- see gemm_combination_mapping /
    # the GEMM_TEMPLATE's accumulation_loop in mlir_gemm_template.py.
    k_iters = best["movin"] // 2 if best["movin"] else None
    bytes_per_iter = (total_weight_bytes / k_iters) if k_iters else None

    result = {
        "N": N, "K": K,
        "total_weight_mb": total_weight_bytes / (1024 * 1024),
        "k_iters": k_iters,
        "bytes_per_iter_mb": (bytes_per_iter / (1024 * 1024)) if bytes_per_iter else None,
        "dram_util_pct": best["dram_util_pct"],
        "dram_gbs": best["dram_gbs"],
        "gemm_insts": best["gemm"],
        "log_file": best["log_file"],
    }
    print(f"[{label}] N={N} K={K}: weight={result['total_weight_mb']:.1f}MB, "
          f"k_iters={k_iters}, bytes/iter={result['bytes_per_iter_mb']:.2f}MB, "
          f"DRAM util={result['dram_util_pct']:.2f}% ({result['dram_gbs']:.2f} GB/s), "
          f"GEMM insts={result['gemm_insts']}")
    return result


def _print_table(title, rows, vary_col):
    print(f"\n{title}")
    header = (f"{vary_col:>8} | {'weight MB':>10} | {'k_iters':>8} | {'MB/iter':>8} | "
              f"{'DRAM util %':>11} | {'GB/s':>7} | {'GEMM insts':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        vary_val = r[vary_col]
        print(f"{vary_val:>8} | {r['total_weight_mb']:>10.1f} | {r['k_iters']:>8} | "
              f"{r['bytes_per_iter_mb']:>8.2f} | {r['dram_util_pct']:>11.2f} | "
              f"{r['dram_gbs']:>7.2f} | {r['gemm_insts']:>10}")


def main():
    fixed_K = 4096
    n_values = [32, 128, 512, 2048, 4096, 11008]
    print(f"\n=== Sweep A: vary N, fixed K={fixed_K}, M={M} ===")
    sweep_a = [_run_gemv_probe(n, fixed_K, f"sweep-A N={n}") for n in n_values]

    fixed_N = 4096
    k_values = [1024, 4096, 16384, 65536, 131072]
    print(f"\n=== Sweep B: vary K, fixed N={fixed_N}, M={M} ===")
    sweep_b = [_run_gemv_probe(fixed_N, k, f"sweep-B K={k}") for k in k_values]

    _print_table(f"Sweep A results (N varies, K={fixed_K} fixed):", sweep_a, "N")
    _print_table(f"Sweep B results (K varies, N={fixed_N} fixed):", sweep_b, "K")

    print("\nRead the DRAM util % and MB/iter columns across each sweep:")
    print("  - If util % tracks 'weight MB' (total size) similarly in both sweeps,")
    print("    regardless of whether N or K grew, total data volume is the likely driver.")
    print("  - If Sweep A (N growing) degrades utilization but Sweep B (K growing, N=4096")
    print("    fixed) does NOT recover/degrade the same way, N specifically (not just size)")
    print("    matters -- e.g. via the smaller TILE_K it forces.")
    print("  - If MB/iter stays roughly constant across a sweep while util % still moves,")
    print("    iteration count alone doesn't explain it either -- something else does.")
    print("This script does not draw that conclusion for you -- read the table.")


if __name__ == "__main__":
    main()
