"""
Controlled M-sweep: isolates whether GEMV's (M=1) low DRAM utilization is
caused by the M dimension being *padded* (1 real row hiding inside an
8-row tile) as opposed to just being *small* (a real, fully-populated
8-row tile).

Context (see tests/test_double_buffering.py and tests/test_gemv.py):
  - gemm_combination_mapping (PyTorchSimFrontend/mlir/mlir_template.py) pads
    M up to a multiple of 8 whenever M <= vector_lane (32). For M=1 (a real
    GEMV, e.g. Llama2-7B decode) that means TILE_M=8 with only 1 real row
    and 7 rows of padding. For M=8 exactly, M_padded == M == 8 too -- same
    TILE_M, but now all 8 rows are real data, no padding.
  - A DMA-bound M=32 matmul (tests/test_double_buffering.py's
    test_dma_bound_matmul_saturates_dram_more_than_non_double_buffered_add)
    measured 81.7% DRAM utilization with every GEMM instruction costing the
    same nonzero compute (active_cycles == GEMM_inst_count * 16 exactly).
    A Llama2-7B GEMV (M=1, N=4096/11008, K=4096) measured only 37-47% DRAM
    utilization with the *average* GEMM instruction costing far less
    (active_cycles << GEMM_inst_count * 16).
  - It's not yet established whether that gap is specifically an artifact of
    M=1 being *padded* (7/8 of the tile is wasted/no-op), or would show up
    for any small-M matmul regardless of padding, or is something else
    entirely (see the "uncertain" caveats below).

This script holds N=32 and K=131072 fixed (the same DMA-bound shape family
as test_double_buffering.py) and sweeps only M in {1, 8, 32}:
  - M=1:  real GEMV, TILE_M=8, 1/8 of the tile is real data (7/8 padding).
  - M=8:  TILE_M=8 too (M_padded == M == 8), but *all* 8 rows are real data.
  - M=32: already-verified high-DRAM-utilization baseline (TILE_M=32==M).

Since M=1 and M=8 both compile down to the identical TILE_M=8 tile shape,
any measured difference between them isolates the "real vs. padding rows"
effect specifically -- everything else about the tiling (TILE_N, TILE_K,
GEMM instruction count formula, DMA transfer sizes) should be identical.

This is a diagnostic/investigative script, not a pass/fail test: it prints
a comparison table and does not assert, since (per the caveats above) it
isn't yet established what the "expected" numbers should be.

Usage (from repo root):
    source setup_env_var.sh gemv_vs_gemm_m8
    bash cleanup_results.sh
    python3 tests/test_gemv_vs_gemm_m8.py
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
_INST_COUNT_RE = re.compile(r"Core \[0\] : (\w+)\s+inst_count: (\d+)(?: \(GEMM: (\d+), Vector: (\d+)\))?")
_SA_UTIL_FINAL_RE = re.compile(
    r"Core \[0\] : Systolic array \[0\] utilization\(%\): ([\d.]+), active_cycles: (\d+), idle_cycles: (\d+)"
)

N = 32
K = 131072


def _log_dir():
    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError("TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?")
    return Path(log_dir)


def _run_matmul_probe(M, label):
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

    # Take the log with the most DRAM traffic if more than one kernel launched.
    best = None
    for f in new_files:
        text = f.read_text(errors="replace")
        dram_matches = _DRAM_FINAL_RE.findall(text)
        if not dram_matches:
            continue
        gbs, util_pct, reads, writes = dram_matches[-1]
        total_traffic = int(reads) + int(writes)
        if best is None or total_traffic > best["_traffic"]:
            # Last occurrence of each per-run stat is the end-of-run summary
            # (Core.cc's print_stat(), which always runs after every
            # periodic print_current_stats() window).
            inst_counts = {}
            for opcode, total, gemm, vector in _INST_COUNT_RE.findall(text):
                inst_counts[opcode] = int(total)
            sa_matches = _SA_UTIL_FINAL_RE.findall(text)
            sa_util_pct, sa_active, sa_idle = sa_matches[-1] if sa_matches else (None, None, None)
            best = {
                "_traffic": total_traffic,
                "log_file": f.name,
                "dram_gbs": float(gbs),
                "dram_util_pct": float(util_pct),
                "reads": int(reads),
                "writes": int(writes),
                "movin": inst_counts.get("MOVIN"),
                "gemm": inst_counts.get("COMP"),
                "bar": inst_counts.get("BAR"),
                "sa_util_pct": float(sa_util_pct) if sa_util_pct else None,
                "sa_active_cycles": int(sa_active) if sa_active else None,
                "sa_idle_cycles": int(sa_idle) if sa_idle else None,
            }

    if best is None:
        raise RuntimeError(f"[{label}] no final DRAM/stat lines found in {[f.name for f in new_files]}")

    total_cycles = (best["sa_active_cycles"] or 0) + (best["sa_idle_cycles"] or 0)
    cycles_per_gemm_inst = (best["sa_active_cycles"] / best["gemm"]) if best["gemm"] else None

    print(f"\n=== M={M} ({label}) -- N={N}, K={K} ===")
    print(f"  log={best['log_file']}")
    print(f"  DRAM: {best['dram_gbs']:.2f} GB/s, {best['dram_util_pct']:.2f}% utilization "
          f"({best['reads']} reads, {best['writes']} writes)")
    print(f"  instructions: MOVIN={best['movin']}, GEMM={best['gemm']}, BAR={best['bar']}")
    print(f"  systolic array: utilization={best['sa_util_pct']:.2f}%, "
          f"active_cycles={best['sa_active_cycles']}, total_cycles={total_cycles}")
    print(f"  active_cycles / GEMM_inst_count = {cycles_per_gemm_inst:.3f} cycles/instruction "
          f"(M=32 baseline measured exactly 16.0)")

    best["M"] = M
    best["total_cycles"] = total_cycles
    best["cycles_per_gemm_inst"] = cycles_per_gemm_inst
    return best


def main():
    results = []
    for M, label in [(1, "real GEMV, TILE_M=8 w/ 7/8 padding"),
                      (8, "real 8-row matmul, TILE_M=8 fully real"),
                      (32, "already-verified DMA-bound baseline")]:
        results.append(_run_matmul_probe(M, label))

    print("\n" + "=" * 100)
    print(f"{'M':>4} | {'DRAM util %':>11} | {'GEMM insts':>10} | {'SA util %':>9} | "
          f"{'active_cyc':>10} | {'cyc/GEMM inst':>13} | {'total_cycles':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['M']:>4} | {r['dram_util_pct']:>11.2f} | {r['gemm']:>10} | {r['sa_util_pct']:>9.2f} | "
              f"{r['sa_active_cycles']:>10} | {r['cycles_per_gemm_inst']:>13.3f} | {r['total_cycles']:>12}")
    print("=" * 100)
    print("\nIf M=1 and M=8 differ sharply while M=8 and M=32 look similar, that points at "
          "padding (1-real-row-in-an-8-row-tile) as the driver, not M being small per se.\n"
          "If M=1 and M=8 look similar to each other (and both differ from M=32), the story "
          "is more likely 'M this small is the problem' regardless of padding.\n"
          "This script does not draw that conclusion for you -- read the table.")


if __name__ == "__main__":
    main()
