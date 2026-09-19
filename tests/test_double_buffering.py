"""
Double buffering detection test.

Double buffering means codegen never lets a scratchpad tile use more than
HALF of the scratchpad: the other half is reserved so DMA can load tile N+1
while compute is still consuming tile N. If PyTorchSim did NOT implement
this, its tile search would be free to use the full scratchpad for a single
big tile whenever that tile fits and gives better reuse.

This test checks the mechanism directly and cheaply (no simulation needed):
MLIRTemplateKernel.gemm_combination_mapping (PyTorchSimFrontend/mlir/
mlir_template.py) is the tile-shape search used for matmul codegen. Reading
it shows:

    max_spad_size = spad_size // 2      # double buffer
    max_spad_per_lane = spad_size_per_lane // 2   # double buffer
    ...
    check_spad_size = (used_spad_size < max_spad_size and
                        used_spad_size_per_lane < max_spad_per_lane)

i.e. every candidate tile is rejected unless it fits in half the scratchpad.
conv_combination_mapping/conv_multi_tile_mapping/conv_single_batch_mapping
and flash_sdpa_mapping have the identical "# double buffer" halving; plain
elementwise/vector codegen does not go through any of these functions (see
tests/test_dma_bandwidth.py's docstring) and is NOT expected to double
buffer.

Rather than just reading the source, this test proves the halving is
*enforced behaviorally*: it picks an M/N/K shape whose natural single-shot
(untiled) tile fits the FULL scratchpad but not HALF of it, calls the real
mapping search, and checks that shape is excluded from the returned
candidates while every candidate that IS returned stays under the half-spad
cap. If double buffering were removed (cap changed from spad_size // 2 to
spad_size), that oversized tile would become the top (largest-reuse)
candidate and this test would fail.

Usage -- fast static check (no simulation/TOGSim run required):
    export TORCHSIM_DIR=$(pwd)
    export TOGSIM_CONFIG=$(pwd)/configs/eclab_cambricon.yml   # any config works; picks scratchpad size
    python3 tests/test_double_buffering.py

Usage -- add the slower, DMA-bound end-to-end TOGSim test (~10-20s; runs two
real simulations and checks the DRAM bandwidth reported by Dram.cc against
the values verified in test_dma_bound_matmul_saturates_dram_more_than_non_double_buffered_add's
docstring):
    source setup_env_var.sh double_buffering_test
    bash cleanup_results.sh
    python3 tests/test_double_buffering.py --e2e
"""

import argparse
import os
import re
import sys
from pathlib import Path


def _load_mapping_probe():
    """Import PyTorchSimFrontend.mlir.mlir_template and build a lightweight
    object that can call MLIRTemplateKernel.gemm_combination_mapping without
    constructing a full compiler kernel (it only touches self.spad_info,
    self.vector_lane, self.num_cores -- all supplied by BaseMLIRHardwareInfo).

    `import torch` + touching the npu device must happen first: importing
    mlir_template directly triggers torch's npu backend autoload, which
    itself imports mlir_template -> circular import if mlir_template isn't
    already fully loaded via the normal torch device-backend init path.
    """
    import torch
    torch.device("npu:0")

    from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
    from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo

    class _GemmMappingProbe(BaseMLIRHardwareInfo):
        get_spad_size_per_lane = MLIRTemplateKernel.get_spad_size_per_lane
        gemm_combination_mapping = MLIRTemplateKernel.gemm_combination_mapping

    return _GemmMappingProbe()


def _untiled_spad_usage(M, N, K, precision_bytes):
    # Mirrors gemm_combination_mapping's own used_spad_size formula
    # (n_extra_node=n_prologue_node=0) for the single "one big tile, no
    # subdivision" shape -- i.e. what a mapper *without* double buffering
    # could pick if that shape fits in the full scratchpad.
    return (M * K + K * N + M * N) * precision_bytes


def test_gemm_mapping_caps_tiles_at_half_scratchpad():
    """Every tile gemm_combination_mapping returns must fit in <= half the
    scratchpad (both the aggregate cap and the stricter per-lane cap)."""
    probe = _load_mapping_probe()
    spad_total = probe.spad_info["spad_size"] * probe.vector_lane
    spad_per_lane = probe.spad_info["spad_size"]
    precision_bytes = 2

    M, N, K = 1024, 1024, 1024
    candidates = probe.gemm_combination_mapping(M, N, K, precision_bytes=precision_bytes)
    assert candidates, "gemm_combination_mapping returned no candidates -- can't check double buffering"

    for tile_M, tile_N, tile_K in candidates:
        used = (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N) * precision_bytes
        used_per_lane = (
            probe.get_spad_size_per_lane(tile_K, tile_N)
            + probe.get_spad_size_per_lane(tile_M, tile_K)
            + probe.get_spad_size_per_lane(tile_M, tile_N)
        ) * precision_bytes
        assert used < spad_total // 2, (
            f"tile {(tile_M, tile_N, tile_K)} uses {used} bytes, >= half of the "
            f"{spad_total}-byte scratchpad ({spad_total // 2}) -- double buffering "
            f"does not appear to be enforced."
        )
        assert used_per_lane < spad_per_lane // 2, (
            f"tile {(tile_M, tile_N, tile_K)} uses {used_per_lane} bytes/lane, >= half "
            f"of the {spad_per_lane}-byte-per-lane scratchpad ({spad_per_lane // 2})."
        )
    print(f"[OK] all {len(candidates)} gemm_combination_mapping candidates for M=N=K={M} "
          f"stay under half the {spad_total}-byte scratchpad.")


def test_gemm_mapping_excludes_tile_that_only_fits_full_scratchpad():
    """Pick a shape whose single, untiled allocation fits the FULL scratchpad
    but not HALF of it. If double buffering is implemented, the mapping
    search must exclude that shape (it violates the half-spad cap) even
    though it would otherwise be the best (most-reuse, fewest-DMA-calls)
    choice. If double buffering were absent (cap == full spad instead of
    half), this exact shape would be accepted -- and typically preferred,
    since the search maximizes tile size / reuse."""
    probe = _load_mapping_probe()
    spad_total = probe.spad_info["spad_size"] * probe.vector_lane
    precision_bytes = 2

    M = N = K = 1024
    untiled_usage = _untiled_spad_usage(M, N, K, precision_bytes)
    assert untiled_usage < spad_total, (
        f"test shape unusable on this scratchpad: untiled usage {untiled_usage} bytes "
        f">= full scratchpad {spad_total} bytes; pick a smaller M/N/K for this hardware config."
    )
    assert untiled_usage >= spad_total // 2, (
        f"test shape too small to be diagnostic: untiled usage {untiled_usage} bytes "
        f"already fits half the scratchpad ({spad_total // 2}); pick a larger M/N/K."
    )

    candidates = probe.gemm_combination_mapping(M, N, K, precision_bytes=precision_bytes)
    assert (M, N, K) not in candidates, (
        f"gemm_combination_mapping accepted the untiled ({M},{N},{K}) shape, which needs "
        f"{untiled_usage} bytes -- more than half the {spad_total}-byte scratchpad "
        f"({spad_total // 2}) but less than the full scratchpad. This means tiling is only "
        f"capped by the FULL scratchpad size, not half of it -- double buffering does NOT "
        f"appear to be implemented (or has regressed)."
    )
    print(f"[OK] untiled ({M},{N},{K}) shape needs {untiled_usage} bytes -- fits the full "
          f"{spad_total}-byte scratchpad but not half of it -- and was correctly excluded "
          f"from gemm_combination_mapping's candidates. Double buffering (half-scratchpad "
          f"cap) is active.")


# ---------------------------------------------------------------------------
# Slower, end-to-end DMA-bound companion (real TOGSim run, asserted).
#
# The two static tests above prove the tiling *mechanism* caps tiles at half
# the scratchpad. This section proves that cap actually buys overlap in the
# timing model, by running a genuinely DMA-bound kernel through TOGSim and
# reading Dram.cc's own end-of-run aggregate bandwidth line:
#
#   [DRAM] channels 0..3 combined | 62.72 GB/s aggregate, 81.66% of
#   utilization (avg. per channel) | 524288 reads, 64 writes
#
# DRAM channel utilization (not Core.cc's per-core "DMA active_cycles" stat)
# is the metric used here: an earlier attempt using Core-level DMA duty
# cycle on a large (M=N=1024, K=65536) matmul under eclab_cambricon.yml's
# default single-systolic-array core measured only ~4.6% duty cycle -- the
# run was compute-bound (systolic array was the bottleneck, not DMA), which
# swamps any double-buffering signal. DRAM channel utilization instead
# reports what the memory system itself saw, independent of whether the
# *core* was ever waiting on it, so it stays meaningful even when a given
# op/shape isn't perfectly DMA-bound.
#
# To get *into* the DMA-bound regime reliably (verified empirically against
# this repo's default TOGSIM_CONFIG, configs/eclab_cambricon.yml,
# icnt_injection_ports_per_core: 8): keep the GEMM's M and N pinned to the
# vector width (32) -- trivial compute per tile -- and stream a large K, so
# nearly all traffic is the weight matrix streaming through double-buffered
# tiles:
#     M=32, N=32, K=131072, fp16 -> measured 81-85% DRAM utilization.
# The contrasting non-double-buffered probe is a same-order-of-traffic
# elementwise add, which is NOT DMA-bound at a comparable scale because its
# codegen doesn't overlap load and compute between tiles:
#     1024x1024 fp16 add -> measured ~24% DRAM utilization.
# ---------------------------------------------------------------------------

_DRAM_FINAL_BW_RE = re.compile(
    r"\[DRAM\] channels 0\.\.\d+ combined \| "
    r"([\d.]+) GB/s aggregate, ([\d.]+)% of utilization[^|]*\| "
    r"(\d+) reads, (\d+) writes"
)

MATMUL_DMA_BOUND_M = 32
MATMUL_DMA_BOUND_N = 32
MATMUL_DMA_BOUND_K = 131072

ADD_DMA_BOUND_N = 1024

MATMUL_MIN_DRAM_UTIL_PCT = 55.0   # measured ~81-85%; margin for machine/config variance
ADD_MAX_DRAM_UTIL_PCT = 45.0      # measured ~24%
MIN_UTIL_RATIO = 2.0              # measured ~3.4x


def _run_and_get_final_dram_util(fn, *inputs, label):
    import torch

    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError("TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("*.log"))

    opt_fn = torch.compile(dynamic=False)(fn)
    opt_fn(*inputs)

    after = set(log_dir.glob("*.log"))
    new_files = sorted(after - before, key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not new_files:
        raise RuntimeError(
            f"[{label}] no new TOGSim log appeared in {log_dir} -- is timing mode enabled "
            f"(pytorchsim_timing_mode: 1) in your TOGSIM_CONFIG?"
        )

    best = None
    for f in new_files:
        matches = _DRAM_FINAL_BW_RE.findall(f.read_text(errors="replace"))
        if not matches:
            continue
        gbs, util_pct, reads, writes = matches[-1]
        stats = {
            "gbs": float(gbs), "util_pct": float(util_pct),
            "reads": int(reads), "writes": int(writes), "log_file": f.name,
        }
        if best is None or stats["reads"] + stats["writes"] > best["reads"] + best["writes"]:
            best = stats

    if best is None:
        raise RuntimeError(
            f"[{label}] no final '[DRAM] channels 0..N combined | ...' line found in "
            f"{[f.name for f in new_files]}"
        )
    print(f"[{label}] log={best['log_file']} dram_util={best['util_pct']:.2f}% "
          f"dram_bw={best['gbs']:.2f} GB/s ({best['reads']} reads, {best['writes']} writes)")
    return best


def test_dma_bound_matmul_saturates_dram_more_than_non_double_buffered_add():
    """End-to-end, DMA-bound proof: a weight-streaming GEMM shaped to be
    memory-bound (small fixed M/N, large streamed K) drives DRAM channel
    utilization far higher than a same-order-of-traffic elementwise add,
    because gemm_combination_mapping's half-scratchpad tiles let tile N+1's
    load overlap tile N's compute while the add's tiles cannot."""
    import torch

    device = torch.device("npu:0")
    torch.manual_seed(0)

    a_mm = torch.randn(MATMUL_DMA_BOUND_M, MATMUL_DMA_BOUND_K, dtype=torch.float16).to(device=device)
    b_mm = torch.randn(MATMUL_DMA_BOUND_K, MATMUL_DMA_BOUND_N, dtype=torch.float16).to(device=device)
    matmul_stats = _run_and_get_final_dram_util(
        torch.matmul, a_mm, b_mm, label="matmul (DMA-bound, expected double-buffered)"
    )

    a_add = torch.randn(ADD_DMA_BOUND_N, ADD_DMA_BOUND_N, dtype=torch.float16).to(device=device)
    b_add = torch.randn(ADD_DMA_BOUND_N, ADD_DMA_BOUND_N, dtype=torch.float16).to(device=device)
    add_stats = _run_and_get_final_dram_util(
        lambda x, y: x + y, a_add, b_add, label="add (expected NOT double-buffered)"
    )

    ratio = (
        matmul_stats["util_pct"] / add_stats["util_pct"]
        if add_stats["util_pct"] > 0 else float("inf")
    )

    assert matmul_stats["util_pct"] > MATMUL_MIN_DRAM_UTIL_PCT, (
        f"DMA-bound matmul only reached {matmul_stats['util_pct']:.2f}% DRAM utilization "
        f"(expected > {MATMUL_MIN_DRAM_UTIL_PCT}%). Either this probe shape is no longer "
        f"DMA-bound for the current TOGSIM_CONFIG, or double buffering has regressed."
    )
    assert add_stats["util_pct"] < ADD_MAX_DRAM_UTIL_PCT, (
        f"Elementwise add reached {add_stats['util_pct']:.2f}% DRAM utilization "
        f"(expected < {ADD_MAX_DRAM_UTIL_PCT}%) -- surprisingly high for a codegen path "
        f"that isn't supposed to double buffer."
    )
    assert ratio > MIN_UTIL_RATIO, (
        f"matmul/add DRAM utilization ratio was only {ratio:.2f}x "
        f"(matmul {matmul_stats['util_pct']:.2f}% vs add {add_stats['util_pct']:.2f}%), "
        f"expected > {MIN_UTIL_RATIO}x."
    )
    print(f"\n[OK] DMA-bound matmul reached {matmul_stats['util_pct']:.2f}% DRAM utilization "
          f"vs add's {add_stats['util_pct']:.2f}% ({ratio:.2f}x) -- consistent with double "
          f"buffering keeping the DMA engine fed on the GEMM path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", action="store_true",
                         help="also run the slower, DMA-bound end-to-end TOGSim test "
                              "(needs TORCHSIM_LOG_PATH / TOGSIM_CONFIG from setup_env_var.sh)")
    args = parser.parse_args()

    test_gemm_mapping_caps_tiles_at_half_scratchpad()
    test_gemm_mapping_excludes_tile_that_only_fits_full_scratchpad()
    print("\nPASS: gemm_combination_mapping enforces a half-scratchpad cap -- "
          "double buffering is implemented for GEMM/matmul codegen.")

    if args.e2e:
        print("\nRunning DMA-bound end-to-end test...")
        test_dma_bound_matmul_saturates_dram_more_than_non_double_buffered_add()


if __name__ == "__main__":
    main()
