#!/usr/bin/env python3
import argparse
import csv
import os
from bisect import bisect_right


def load_ranges(path):
    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            start = int(parts[0], 0)
            end = int(parts[1], 0)
            if end < start:
                start, end = end, start
            ranges.append((start, end))
    ranges.sort()
    return ranges


def addr_in_ranges(addr, ranges, starts):
    idx = bisect_right(starts, addr) - 1
    if idx < 0:
        return False
    start, end = ranges[idx]
    return addr <= end


def trace_has_matches(trace_path, ranges, starts):
    with open(trace_path, "r", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader, None)
        if header is None:
            return False
        try:
            addr_idx = header.index("addr")
        except ValueError:
            return False

        for row in reader:
            if len(row) <= addr_idx:
                continue
            try:
                addr = int(row[addr_idx], 0)
            except ValueError:
                continue
            if addr_in_ranges(addr, ranges, starts):
                return True
    return False


def filter_trace(trace_path, out_path, ranges):
    starts = [r[0] for r in ranges]
    if not trace_has_matches(trace_path, ranges, starts):
        return False

    with open(trace_path, "r", encoding="utf-8") as src, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is None:
            return False
        writer.writerow(header)
        try:
            addr_idx = header.index("addr")
        except ValueError:
            return False

        for row in reader:
            if len(row) <= addr_idx:
                continue
            try:
                addr = int(row[addr_idx], 0)
            except ValueError:
                continue
            if addr_in_ranges(addr, ranges, starts):
                writer.writerow(row)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract SSD trace rows whose addresses fall within merged weight ranges."
    )
    parser.add_argument(
        "--trace_name",
        default=None,
        help="trace name",
    )
    parser.add_argument(
        "--dir",
        default="/workspace/legomerged/eclab_legosim/PyTorchSim/ssd_traces",
        help="Directory containing model_weight_ranges_merged.txt and SSD trace CSV files",
    )
    parser.add_argument(
        "--ranges",
        default=None,
        help="Merged weight ranges file (default: model_weight_ranges_merged.txt)",
    )
    args = parser.parse_args()


    base_dir = os.path.join(args.dir, args.trace_name)
    base_dir = os.path.abspath(base_dir)

    if args.ranges is None:
        ranges_path = os.path.join(base_dir, "model_weight_ranges_merged.txt")

    if not os.path.isfile(ranges_path):
        raise SystemExit(f"Ranges file not found: {ranges_path}")

    ranges = load_ranges(ranges_path)
    if not ranges:
        raise SystemExit("No ranges loaded. Check the ranges file.")

    output_dir = os.path.join(base_dir, "model_weight_traces")
    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(base_dir):
        if not name.endswith(".csv"):
            continue
        if name == os.path.basename(ranges_path):
            continue
        trace_path = os.path.join(base_dir, name)
        if not os.path.isfile(trace_path):
            continue
        out_path = os.path.join(output_dir, name)
        filter_trace(trace_path, out_path, ranges)


if __name__ == "__main__":
    main()
