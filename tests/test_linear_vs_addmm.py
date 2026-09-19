"""
linear(input, weight, bias) vs addmm(bias, input, weight) at Llama2-7B's
GEMV shapes -- correctness AND the DRAM-bandwidth gap between them.

Background (see tests/test_gemv.py and the conversation that led here):
Llama2-7B's decode-time projections (q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj) are all nn.Linear, and nn.Linear.forward is literally
`F.linear(input, self.weight, self.bias)`, computing `input @ weight.T +
bias` against `weight`'s real (out_features, in_features) = (N, K)
row-major checkpoint layout. Because of that implicit transpose, the NPU
kernel reads `weight` through a transposed/strided view -- each DMA burst
is bounded by TILE_K (not N) elements, which at Llama's shapes fragments
into many small, far-apart bursts and drives up DRAM row-buffer conflicts.

torch.addmm(bias, input, weight2) computes `bias + input @ weight2`
directly with no implicit transpose. If weight2 is a genuinely
pre-transposed, contiguous (K, N) tensor (`weight.t().contiguous()`, done
ONCE outside the timed path -- see the conversation's caveat about where
that copy has to be paid for in a real streamed-checkpoint pipeline), the
same logical GEMV becomes a contiguous-access matmul instead.

Measured previously at N=4096, K=4096, M=1 (eclab_cambricon.yml):
  linear:                 DRAM util 46.43%, row-buffer hit rate 33.75%
  addmm w/ pretransposed: DRAM util 93.21%, row-buffer hit rate 97.93%
This file reproduces that comparison end-to-end at both of Llama2-7B's
actual GEMV shapes (q_proj/k_proj/v_proj/o_proj: N=K=4096; gate_proj/
up_proj: N=11008, K=4096), and additionally checks that linear and addmm
(pretransposed) actually agree numerically -- the whole point is that
they're the *same* computation reaching the NPU through two different
op/layout paths, not two different computations that happen to look similar.

Note: real Llama2 sets attention_bias=mlp_bias=False for these layers, but
both addmm and linear require a bias argument, so this test uses a real
(nonzero) random bias throughout -- if you want the exact no-bias case,
pass bias=None to compare_linear_vs_addmm (addmm doesn't support
bias=None, so that path falls back to plain matmul for the addmm side).

Usage (from repo root):
    source setup_env_var.sh linear_vs_addmm
    export TOGSIM_CONFIG=$(pwd)/configs/eclab_cambricon.yml   # functional_mode:1, needed for correctness
    bash cleanup_results.sh
    python3 tests/test_linear_vs_addmm.py
"""

import os
import re
import sys
from pathlib import Path

import torch

_DRAM_FINAL_RE = re.compile(
    r"\[DRAM\] channels 0\.\.\d+ combined \| "
    r"([\d.]+) GB/s aggregate, ([\d.]+)% of utilization[^|]*\| "
    r"(\d+) reads, (\d+) writes"
)
_ROW_STAT_RE = re.compile(r"^\s*(row_hits|row_misses|row_conflicts): (\d+)\s*$", re.MULTILINE)


def _log_dir():
    log_dir = os.environ.get("TORCHSIM_LOG_PATH")
    if not log_dir:
        raise RuntimeError("TORCHSIM_LOG_PATH is not set. Did you `source setup_env_var.sh <trace_name>` first?")
    return Path(log_dir)


def _parse_dram_and_row_stats(text):
    dram_matches = _DRAM_FINAL_RE.findall(text)
    if not dram_matches:
        return None
    gbs, util_pct, reads, writes = dram_matches[-1]

    # Aggregate row_hits/row_misses/row_conflicts across all channels (each
    # printed once, under its own "--- channel N ---" block, at end of run).
    hits = misses = conflicts = 0
    for name, count in _ROW_STAT_RE.findall(text):
        if name == "row_hits":
            hits += int(count)
        elif name == "row_misses":
            misses += int(count)
        elif name == "row_conflicts":
            conflicts += int(count)
    total_row_events = hits + misses + conflicts

    return {
        "dram_gbs": float(gbs),
        "dram_util_pct": float(util_pct),
        "reads": int(reads),
        "writes": int(writes),
        "row_hit_rate_pct": (100.0 * hits / total_row_events) if total_row_events else None,
        "row_conflicts": conflicts,
    }


def _run_and_measure(fn, inputs, label):
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("*.log"))

    opt_fn = torch.compile(dynamic=False)(fn)
    out = opt_fn(*inputs)

    after = set(log_dir.glob("*.log"))
    new_files = sorted(after - before, key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not new_files:
        raise RuntimeError(f"[{label}] no new TOGSim log appeared in {log_dir}")

    best = None
    for f in new_files:
        stats = _parse_dram_and_row_stats(f.read_text(errors="replace"))
        if stats is not None and (best is None or stats["reads"] + stats["writes"] > best["_traffic"]):
            stats["_traffic"] = stats["reads"] + stats["writes"]
            stats["log_file"] = f.name
            best = stats

    if best is None:
        raise RuntimeError(f"[{label}] no DRAM stat lines found in {[f.name for f in new_files]}")

    print(f"[{label}] log={best['log_file']} DRAM util={best['dram_util_pct']:.2f}% "
          f"({best['dram_gbs']:.2f} GB/s), row_hit_rate="
          f"{best['row_hit_rate_pct']:.2f}%, row_conflicts={best['row_conflicts']}")
    return out, best


def _rel_l2(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return (a - b).norm().item() / b.norm().item()


def compare_linear_vs_addmm(device, name, in_features, out_features, dtype=torch.float16,
                             use_bias=True, rel_l2_threshold=0.02):
    torch.manual_seed(0)
    x = torch.randn(1, in_features, dtype=dtype)
    weight_nk = torch.randn(out_features, in_features, dtype=dtype)  # HF/Llama checkpoint layout (N, K)
    bias = torch.randn(out_features, dtype=dtype) if use_bias else torch.zeros(out_features, dtype=dtype)

    # The "fix": physically transpose+materialize ONCE, on host, before this ever reaches the NPU.
    weight_kn = weight_nk.t().contiguous()
    assert weight_kn.is_contiguous() and weight_kn.stride() == (out_features, 1)

    x_dev = x.to(device=device)
    weight_nk_dev = weight_nk.to(device=device)
    weight_kn_dev = weight_kn.to(device=device)
    bias_dev = bias.to(device=device)

    cpu_ref = torch.nn.functional.linear(x, weight_nk, bias)

    print(f"\n=== {name}: in_features(K)={in_features}, out_features(N)={out_features} ===")

    out_linear, stats_linear = _run_and_measure(
        torch.nn.functional.linear, (x_dev, weight_nk_dev, bias_dev),
        label=f"{name} linear(x, weight_nk, bias)"
    )
    out_addmm, stats_addmm = _run_and_measure(
        torch.addmm, (bias_dev, x_dev, weight_kn_dev),
        label=f"{name} addmm(bias, x, weight_kn)"
    )

    rel_l2_linear_vs_cpu = _rel_l2(out_linear, cpu_ref)
    rel_l2_addmm_vs_cpu = _rel_l2(out_addmm, cpu_ref)
    rel_l2_linear_vs_addmm = _rel_l2(out_linear, out_addmm)

    print(f"[{name}] relative L2 error vs CPU reference: linear={rel_l2_linear_vs_cpu:.4%}, "
          f"addmm={rel_l2_addmm_vs_cpu:.4%}")
    print(f"[{name}] relative L2 diff between linear and addmm outputs: {rel_l2_linear_vs_addmm:.4%}")

    ok = True
    for check_name, err in [("linear vs CPU", rel_l2_linear_vs_cpu),
                             ("addmm vs CPU", rel_l2_addmm_vs_cpu),
                             ("linear vs addmm", rel_l2_linear_vs_addmm)]:
        status = "OK" if err < rel_l2_threshold else "MISMATCH"
        if err >= rel_l2_threshold:
            ok = False
        print(f"  [{status}] {check_name}: {err:.4%} (threshold {rel_l2_threshold:.2%})")

    util_ratio = (stats_addmm["dram_util_pct"] / stats_linear["dram_util_pct"]
                  if stats_linear["dram_util_pct"] else float("inf"))
    print(f"[{name}] DRAM utilization: linear={stats_linear['dram_util_pct']:.2f}%, "
          f"addmm={stats_addmm['dram_util_pct']:.2f}% ({util_ratio:.2f}x)")
    print(f"[{name}] Row-buffer hit rate: linear={stats_linear['row_hit_rate_pct']:.2f}%, "
          f"addmm={stats_addmm['row_hit_rate_pct']:.2f}%")

    return {
        "name": name, "K": in_features, "N": out_features,
        "ok": ok,
        "linear": stats_linear, "addmm": stats_addmm,
    }


def main():
    device = torch.device("npu:0")

    results = []
    # Llama2-7B q_proj (and k_proj/v_proj/o_proj share this shape)
    results.append(compare_linear_vs_addmm(device, "q_proj", in_features=4096, out_features=4096))
    # Llama2-7B gate_proj (and up_proj share this shape)
    results.append(compare_linear_vs_addmm(device, "gate_proj", in_features=4096, out_features=11008))

    print("\n" + "=" * 92)
    print(f"{'shape':>10} | {'N':>6} | {'K':>6} | {'linear util%':>12} | {'addmm util%':>11} | "
          f"{'linear hit%':>11} | {'addmm hit%':>10} | correctness")
    print("-" * 92)
    all_ok = True
    for r in results:
        all_ok = all_ok and r["ok"]
        print(f"{r['name']:>10} | {r['N']:>6} | {r['K']:>6} | "
              f"{r['linear']['dram_util_pct']:>12.2f} | {r['addmm']['dram_util_pct']:>11.2f} | "
              f"{r['linear']['row_hit_rate_pct']:>11.2f} | {r['addmm']['row_hit_rate_pct']:>10.2f} | "
              f"{'OK' if r['ok'] else 'MISMATCH'}")
    print("=" * 92)

    if not all_ok:
        print("\nFAIL: linear and addmm(pretransposed) produced different results somewhere above.")
        sys.exit(1)
    print("\nPASS: linear and addmm(pretransposed) agree numerically at both Llama2-7B GEMV shapes; "
          "see the utilization/hit-rate columns above for the bandwidth gap between them.")


if __name__ == "__main__":
    main()
