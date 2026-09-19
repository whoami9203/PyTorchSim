"""
Convenience layer on top of PyTorchSim's external GEMM tile-mapping override
(codegen_mapping_strategy: external-then-heuristic + codegen_external_mapping_file
-- see select_tile() in PyTorchSimFrontend/mlir/mlir_gemm_template.py).

That mechanism's JSON is keyed by raw "M_N_K" strings (e.g. "1_4096_11008")
with TILE_M/TILE_N/TILE_K values -- you have to already know exactly what
M/N/K the tracer extracted for a given op, and there's nowhere to write down
*why* a given tile was chosen. This module lets you instead maintain a small,
named, human-readable YAML file (one entry per op you want to override, named
by in_features/out_features rather than raw N/K, with room for a comment) and
generates the real JSON mapping file from it.

YAML schema -- one top-level key per op you want to override, e.g.:

    down_proj:
      in_features: 11008    # K
      out_features: 4096    # N
      tile_m: 8
      tile_n: 4096
      tile_k: 256
      # m: 1                # optional, defaults to 1 (decode-time GEVM)

`in_features`/`out_features` match nn.Linear's convention (weight shape is
(out_features, in_features)); they become K/N via MLIRGemmTemplate.extract_info's
M, N, K = X.size(0), W.size(1), X.size(1) (X is (M,K), W is (K,N) --
i.e. the weight must reach the NPU already (K,N)-contiguous, e.g. via
tests/Llama/test_llama2_7B.py's TransposedLinear, for the M/N/K PyTorchSim
sees to match what you wrote here). An op left out of the YAML entirely falls
straight through to the normal heuristic search (gemm_combination_mapping) --
no entry needed for shapes that are already well-tiled.

Usage: call apply_tile_overrides(yaml_path, json_path) once, early -- it's
cheap and idempotent, so it's fine to call unconditionally on every run, the
same way tests/Llama/sim_llama2_7B.py and test_llama2_7B.py do. It's a no-op
useless-but-harmless bystander unless TOGSIM_CONFIG also points at a config
with:
    codegen_mapping_strategy: external-then-heuristic
    codegen_external_mapping_file: <the json_path you passed here>
(see configs/eclab_cambricon_tile_overrides.yml for a ready-made one).
"""

import json
from pathlib import Path

import yaml


def apply_tile_overrides(yaml_path, json_path, default_m=1):
    """Reads `yaml_path` (see this module's docstring for schema) and writes
    PyTorchSim's external-mapping JSON to `json_path`. Returns the mapping
    dict that was written, keyed by "{M}_{N}_{K}"."""
    yaml_path = Path(yaml_path)
    json_path = Path(json_path)
    spec = yaml.safe_load(yaml_path.read_text()) or {}

    mapping = {}
    for name, entry in spec.items():
        for field in ("in_features", "out_features", "tile_m", "tile_n", "tile_k"):
            if field not in entry:
                raise ValueError(f"tile override '{name}' in {yaml_path} is missing '{field}'")
        m = entry.get("m", default_m)
        key = f"{m}_{entry['out_features']}_{entry['in_features']}"
        mapping[key] = {
            "TILE_M": entry["tile_m"],
            "TILE_N": entry["tile_n"],
            "TILE_K": entry["tile_k"],
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(mapping, indent=2))
    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_path", help="Human-readable tile-override spec (see this module's docstring)")
    parser.add_argument("json_path", help="Where to write PyTorchSim's codegen_external_mapping_file JSON")
    args = parser.parse_args()
    written = apply_tile_overrides(args.yaml_path, args.json_path)
    print(f"Wrote {len(written)} tile override(s) to {args.json_path}:")
    for key, tile in written.items():
        print(f"  {key}: {tile}")
