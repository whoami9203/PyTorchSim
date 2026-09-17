import argparse
import os
import re
from typing import List, Tuple

RANGE_RE = re.compile(r"\bbase=(\d+)\b.*\bend=(\d+)\b")
TENSOR_RE = re.compile(
    r"^(?P<name>[^\t]+)\tbase=(?P<base>\d+)\tend=(?P<end>\d+)\tsize_bytes=(?P<size>\d+)"
    r"\tshape=\((?P<shape>[^)]*)\)"
)


def parse_ranges(path: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    with open(path, "r") as f:
        for line in f:
            match = RANGE_RE.search(line)
            if not match:
                continue
            base = int(match.group(1))
            end = int(match.group(2))
            if end < base:
                base, end = end, base
            ranges.append((base, end))
    return ranges


def extract_1d_tensors(path: str) -> List[Tuple[str, int, int, int]]:
    """Returns (name, base, end, size_bytes) for every tensor in `path` whose
    shape has exactly one dimension -- LayerNorm/RMSNorm weight and bias
    vectors. Kept as individual, unmerged rows (unlike merge_ranges() above)
    so SingleShotWeightGate can track each tensor's own address span rather
    than a coalesced range that may also cover an unrelated neighbor.
    """
    tensors: List[Tuple[str, int, int, int]] = []
    with open(path, "r") as f:
        for line in f:
            match = TENSOR_RE.match(line)
            if not match:
                continue
            dims = [d.strip() for d in match.group("shape").split(",") if d.strip() != ""]
            if len(dims) != 1:
                continue
            tensors.append((
                match.group("name"),
                int(match.group("base")),
                int(match.group("end")),
                int(match.group("size")),
            ))
    return tensors


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged: List[Tuple[int, int]] = [ranges[0]]
    for base, end in ranges[1:]:
        last_base, last_end = merged[-1]
        if base <= last_end:
            merged[-1] = (last_base, max(last_end, end))
            continue
        if base == last_end:
            merged[-1] = (last_base, end)
            continue
        merged.append((base, end))
    return merged


def default_paths() -> Tuple[str, str]:
    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_DIR and TOGSIM_SSD_TRACE_NAME must be set")
    base_dir = os.path.join(trace_dir, trace_name)
    input_path = os.path.join(base_dir, "model_weight_ranges.txt")
    output_path = os.path.join(base_dir, "model_weight_ranges_merged.txt")
    return input_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge model weight address ranges")
    parser.add_argument("--input", dest="input_path", default=None, help="Path to model_weight_ranges.txt")
    parser.add_argument("--output", dest="output_path", default=None, help="Output path for merged ranges")
    args = parser.parse_args()

    if args.input_path is None or args.output_path is None:
        input_path, output_path = default_paths()
    else:
        input_path = args.input_path
        output_path = args.output_path

    ranges = parse_ranges(input_path)
    merged = merge_ranges(ranges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for base, end in merged:
            f.write(f"{base}\t{end}\n")

    print(f"Parsed {len(ranges)} ranges, merged to {len(merged)} ranges.")
    print(f"Output: {output_path}")

    tensors_1d = extract_1d_tensors(input_path)
    output_1d_path = os.path.join(os.path.dirname(output_path), "model_weight_ranges_1d.txt")
    with open(output_1d_path, "w") as f:
        for name, base, end, size_bytes in tensors_1d:
            f.write(f"{name}\t{base}\t{end}\t{size_bytes}\n")

    print(f"Found {len(tensors_1d)} 1-D tensor(s), written to {output_1d_path}")


if __name__ == "__main__":
    main()
