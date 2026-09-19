"""
down_proj GEMV (M=1, N=4096, K=11008) tile-size sweep.

Context (see the conversation this came out of, and tests/test_gemv_nk_sweep.py):
Llama2-7B's down_proj is nn.Linear(intermediate_size=11008, hidden_size=4096),
i.e. a GEMV with M=1, K=in_features=11008, N=out_features=4096. Even after
fixing the linear-vs-matmul weight-layout issue (TransposedLinear in
tests/Llama/test_llama2_7B.py), a real decode run still measured down_proj at
only 37.74% DRAM utilization (togsim_results/llama2_7B_decode_fp16_cam2/35.log)
-- clearly worse than q/k/v/o/gate/up_proj's ~93%.

Querying gemm_combination_mapping (PyTorchSimFrontend/mlir/mlir_template.py)
directly for this exact shape shows why: its heuristic search picks
(TILE_M=8, TILE_N=128, TILE_K=11008) -- TILE_N far short of the full N=4096 --
because larger TILE_N leaves less half-scratchpad budget for TILE_K, and the
search scores raw used-spad-size (tile_M*tile_K + tile_K*tile_N + tile_M*tile_N)
rather than "how many separate N-tiles does this produce" (N/TILE_N -- 32 of
them here, each needing its own MOVOUT and a full separate K-pass over the
same tiny M=1 activation vector). The candidate search also reports, for this
shape, the largest TILE_K that still fits under the half-scratchpad cap at
each TILE_N:

    TILE_N=32/64/128  -> max TILE_K=11008 (full K, one shot) <- heuristic picks 128
    TILE_N=256        -> max TILE_K=5504
    TILE_N=512        -> max TILE_K=2752
    TILE_N=1024       -> max TILE_K=1376
    TILE_N=2048/4096  -> max TILE_K=256

This script forces each of those (TILE_N, TILE_K) pairs -- including
TILE_N=N=4096, which the heuristic never picks on its own -- via
PyTorchSim's external tile-mapping override (codegen_mapping_strategy:
external-then-heuristic, configs/eclab_cambricon_external.yml), and compares
DRAM utilization, row-buffer hit rate, and MOVOUT count across all of them
for the identical (M,N,K)=(1,4096,11008) GEMV, plus a plain heuristic-default
run (no override) as a baseline/sanity check that the override path
reproduces the heuristic's own choice when told to.

The weight tensor is built directly (K, N)-contiguous, matching what
TransposedLinear hands the NPU at runtime (not nn.Linear's native (N,K) --
that's a different, already-diagnosed problem; this script isolates the tile
*shape* question on its own).

Usage (from repo root):
    source setup_env_var.sh down_proj_tile_sweep
    export TOGSIM_CONFIG=/workspace/legomerged/eclab_legosim/PyTorchSim/configs/eclab_cambricon_external.yml
    bash cleanup_results.sh
    python3 tests/test_down_proj_tile_sweep.py
"""

import json
import os
import re
import shutil
from pathlib import Path

import torch

TORCHINDUCTOR_CACHE_DIR = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "outputs/.torchinductor"))

M, N, K = 1, 4096, 11008
PRECISION_BYTES = 2  # fp16
GEMM_SHAPE_KEY = f"{M}_{N}_{K}"

MAPPING_FILE = Path("/workspace/legomerged/eclab_legosim/PyTorchSim/configs/tile_mapping_override.generated.json")

# (label, TILE_N, TILE_K) -- TILE_M is always 8 (M=1 padded to the vector_lane's
# minimum granularity; see gemm_combination_mapping). None/None means "no
# override, let the heuristic search pick" (the current, ~38%-utilization
# behavior). Each (TILE_N, TILE_K) pair other than the heuristic-default is
# the largest TILE_K gemm_combination_mapping's own candidate search reports
# as feasible under the half-scratchpad cap for that TILE_N, queried directly
# -- not guessed.
TILE_CONFIGS = [
    ("heuristic-default (no override)", None, None),
    ("TILE_N=128 (== heuristic's own choice, explicit)", 128, 11008),
    ("TILE_N=256", 256, 5504),
    ("TILE_N=512", 512, 2752),
    ("TILE_N=1024", 1024, 1376),
    ("TILE_N=2048", 2048, 256),
    ("TILE_N=4096 (== N, full width)", 4096, 256),
]
TILE_M = 8

_DRAM_FINAL_RE = re.compile(
    r"\[DRAM\] channels 0\.\.\d+ combined \| "
    r"([\d.]+) GB/s aggregate, ([\d.]+)% of utilization[^|]*\| "
    r"(\d+) reads, (\d+) writes"
)
_INST_COUNT_RE = re.compile(r"Core \[0\] : (\w+)\s+inst_count: (\d+)")
_ROW_STAT_RE = re.compile(r"^\s*(row_hits|row_misses|row_conflicts): (\d+)\s*$", re.MULTILINE)


def _write_mapping(tile_n, tile_k):
    if tile_n is None:
        MAPPING_FILE.write_text("{}")
        return
    data = {GEMM_SHAPE_KEY: {"TILE_M": TILE_M, "TILE_N": tile_n, "TILE_K": tile_k}}
    MAPPING_FILE.write_text(json.dumps(data))


def _log_dir():
    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError("TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?")
    return Path(log_dir)


def _rel_l2(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return (a - b).norm().item() / b.norm().item()


def _run_one(label, tile_n, tile_k, x, weight_kn, ref):
    _write_mapping(tile_n, tile_k)

    # Every test point calls torch.matmul with the exact same (M,N,K,dtype,device)
    # signature -- only the on-disk mapping file changes. Without this reset,
    # Dynamo's guard cache (keyed by torch.matmul's code object + those argument
    # properties, not by which torch.compile() wrapper instance called it) matches
    # every call after the first and just replays the FIRST compiled kernel,
    # silently ignoring every later tile-size override. Confirmed empirically:
    # omitting this reset produced 7 byte-identical results across 7 different
    # tile configs.
    torch._dynamo.reset()
    # dynamo.reset() alone isn't enough: Inductor's ON-DISK code cache
    # (TORCHINDUCTOR_CACHE_DIR, persists across dynamo resets by design) is
    # keyed by the FX graph's structure, which is identical for every tile
    # config here -- it doesn't know the external mapping JSON's *contents*
    # changed. Confirmed empirically too: with only dynamo.reset(), the
    # heuristic-default run got its own kernel, but every explicit TILE_N in
    # {128,256,512,1024,2048,4096} silently reused the SAME cached kernel
    # (the first explicit override's), all reporting identical results.
    if TORCHINDUCTOR_CACHE_DIR.exists():
        shutil.rmtree(TORCHINDUCTOR_CACHE_DIR)
    TORCHINDUCTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("*.log"))

    opt_fn = torch.compile(dynamic=False)(torch.matmul)
    out = opt_fn(x, weight_kn)
    rel_l2 = _rel_l2(out, ref)

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
        traffic = int(reads) + int(writes)
        if best is None or traffic > best["_traffic"]:
            inst = {op: int(c) for op, c in _INST_COUNT_RE.findall(text)}
            hits = misses = conflicts = 0
            for name, count in _ROW_STAT_RE.findall(text):
                if name == "row_hits":
                    hits += int(count)
                elif name == "row_misses":
                    misses += int(count)
                elif name == "row_conflicts":
                    conflicts += int(count)
            total_row = hits + misses + conflicts
            best = {
                "_traffic": traffic, "log_file": f.name,
                "dram_gbs": float(gbs), "dram_util_pct": float(util_pct),
                "movin": inst.get("MOVIN"), "movout": inst.get("MOVOUT"), "gemm": inst.get("COMP"),
                "row_hit_rate_pct": (100.0 * hits / total_row) if total_row else None,
            }

    if best is None:
        raise RuntimeError(f"[{label}] no DRAM stat line found in {[f.name for f in new_files]}")

    result = {
        "label": label, "tile_n": tile_n if tile_n is not None else "heuristic",
        "tile_k": tile_k if tile_k is not None else "heuristic",
        "rel_l2_pct": rel_l2 * 100,
        **best,
    }
    print(f"[{label}] log={best['log_file']} MOVIN={best['movin']} MOVOUT={best['movout']} "
          f"GEMM={best['gemm']} DRAM util={best['dram_util_pct']:.2f}% ({best['dram_gbs']:.2f} GB/s) "
          f"row_hit_rate={best['row_hit_rate_pct']:.2f}% rel_l2_err={result['rel_l2_pct']:.4f}%")
    return result


def main():
    device = torch.device("npu:0")
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float16)
    weight_kn = torch.randn(K, N, dtype=torch.float16)  # (K, N)-contiguous, matching TransposedLinear's layout
    ref = torch.matmul(x, weight_kn)

    x_dev = x.to(device=device)
    weight_kn_dev = weight_kn.to(device=device)

    results = []
    for label, tile_n, tile_k in TILE_CONFIGS:
        results.append(_run_one(label, tile_n, tile_k, x_dev, weight_kn_dev, ref))

    print("\n" + "=" * 108)
    print(f"{'TILE_N':>10} | {'TILE_K':>8} | {'MOVOUT':>7} | {'GEMM insts':>10} | "
          f"{'DRAM util %':>11} | {'row hit %':>9} | {'rel L2 err %':>12} | label")
    print("-" * 108)
    for r in results:
        print(f"{r['tile_n']!s:>10} | {r['tile_k']!s:>8} | {r['movout']!s:>7} | {r['gemm']!s:>10} | "
              f"{r['dram_util_pct']:>11.2f} | {r['row_hit_rate_pct']:>9.2f} | {r['rel_l2_pct']:>12.4f} | {r['label']}")
    print("=" * 108)

    heuristic = results[0]
    explicit_128 = results[1]
    full_n = results[-1]
    print(f"\nheuristic-default vs explicit TILE_N=128 (should match): "
          f"{heuristic['dram_util_pct']:.2f}% vs {explicit_128['dram_util_pct']:.2f}%")
    print(f"heuristic-default vs explicit TILE_N=N=4096: "
          f"{heuristic['dram_util_pct']:.2f}% vs {full_n['dram_util_pct']:.2f}% "
          f"({full_n['dram_util_pct'] / heuristic['dram_util_pct']:.2f}x)")


if __name__ == "__main__":
    main()
