"""Build a tensor-name -> SSD-offset/length map for one Llama2-7B decoder layer.

SimpleSSD has no notion of tensors -- it only understands a flat byte range
of LBAs. This script decides, for one decoder layer, where each of its
parameter tensors is considered to live in that flat SSD address space, so
that:
  - the FTL can be pre-warmed (mapped) over exactly that range before any
    timed read, since page_mapping.cc's readInternal() bills zero NAND
    latency to LBAs that were never written (see conversation notes on
    SimpleSSD-Standalone's readInternal/FillRatio gotcha).
  - repeated loads of "the same" tensor land on the same offset, and
    adjacent tensors don't alias into the same page (each tensor's end is
    rounded up to --alignment, default the SSD's PAL PageSize of 16384B --
    see simplessd/config/sample.cfg's [pal] PageSize).

By default it uses a hardcoded Llama2-7B decoder-layer tensor list (standard
HF LlamaDecoderLayer parameter order/shapes for hidden=4096,
intermediate=11008; Llama has no linear-layer biases). Pass --from-trace to
use real captured names/sizes instead, from a model_weight_ranges.txt
produced by PyTorchSim/tests/Llama/test_llama2_7B.py's
_dump_module_weight_ranges() -- preferable when available, since it sidesteps
any drift between this script's hand-derived shapes and the real checkpoint
(tied embeddings, quantization, etc.).

Output is JSON: one entry per tensor with its SSD offset/size, plus
layer_span_bytes -- the aligned total size of the layer, usable directly as
config/sample.cfg's [generator] io_size for a single-layer read simulation.

--pytorchsim-offsets-tsv (requires --from-trace) additionally writes a
"<name> <dram_base> <dram_end> <ssd_offset> <ssd_length>" TSV for
simpleSSD-lego/SimpleSSD-Standalone/sim/legosim_pytorchsim_main.cc's live
bridge. That bridge matches each incoming DMA request by its real host DRAM
address (req.addr) against [dram_base, dram_end) -- NOT by name -- because
TOGSim's own addr_name is just a positional kernel-argument label ("arg0",
"arg1", ...) reused across unrelated kernels, not a stable tensor identity
(see PyTorchSim/TOGSim/src/DMA.cc's comment on why name-based classification
was tried and reverted there too). dram_base/dram_end come straight from
model_weight_ranges.txt's base=/end= fields, so this only makes sense
together with --from-trace.
"""

import argparse
import json
import re
import sys

DTYPE_BYTES = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "int8": 1,
    "fp8": 1,
}

# Standard HF LlamaDecoderLayer parameter order for the 7B config
# (hidden_size=4096, intermediate_size=11008). No biases: Llama uses
# RMSNorm and bias-free linear projections.
LLAMA2_7B_LAYER_TENSORS = [
    ("input_layernorm.weight", (4096,)),
    ("self_attn.q_proj.weight", (4096, 4096)),
    ("self_attn.k_proj.weight", (4096, 4096)),
    ("self_attn.v_proj.weight", (4096, 4096)),
    ("self_attn.o_proj.weight", (4096, 4096)),
    ("post_attention_layernorm.weight", (4096,)),
    ("mlp.gate_proj.weight", (11008, 4096)),
    ("mlp.up_proj.weight", (11008, 4096)),
    ("mlp.down_proj.weight", (4096, 11008)),
]

TRACE_LINE_RE = re.compile(
    r"^(?P<name>\S+)\tbase=(?P<base>\d+)\tend=(?P<end>\d+)\tsize_bytes=(?P<size>\d+)"
)


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def align_up(value, alignment):
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def load_tensors_from_trace(path, layer_idx, name_prefix=None):
    """Parse model_weight_ranges.txt lines written by _dump_module_weight_ranges():
       <name>\\tbase=<int>\\tend=<int>\\tsize_bytes=<int>\\tshape=(...)\\tdtype=<...>
    Keeps only lines whose name starts with `name_prefix` (default
    "model.layers.<layer_idx>." -- Llama's convention; pass --name-prefix
    explicitly for other architectures, e.g. GPT-NeoX's dumps use
    "gpt_neox.layers.<N>." -- see tests/GPT/test_GPT_NeoX_20B.py), strips
    that prefix for display, and returns a list of dicts (name, size_bytes,
    dram_base, dram_end) in file order (the real named_parameters() order the
    layer was dumped in). dram_base/dram_end are the real host DRAM addresses
    the tensor occupied at dump time -- see --pytorchsim-offsets-tsv.
    """
    prefix = name_prefix if name_prefix is not None else "model.layers.%d." % layer_idx
    tensors = []
    with open(path) as f:
        for line in f:
            m = TRACE_LINE_RE.match(line)
            if not m or not m.group("name").startswith(prefix):
                continue
            tensors.append(
                {
                    "name": m.group("name")[len(prefix):],
                    "size_bytes": int(m.group("size")),
                    "dram_base": int(m.group("base")),
                    "dram_end": int(m.group("end")),
                }
            )
    if not tensors:
        raise ValueError(
            "No tensors found with prefix '%s' in %s" % (prefix, path)
        )
    return tensors


def build_offset_map(tensors, layer_base_offset, alignment):
    """Packs tensors back-to-back starting at layer_base_offset, each tensor's
    end rounded up to `alignment` so no two tensors share a page/LBA-alignment
    unit. `tensors` is a list of dicts with at least name/size_bytes (dram_base/
    dram_end are carried through into the output entries when present).
    Returns (entries, layer_span_bytes).
    """
    entries = []
    offset = layer_base_offset
    for t in tensors:
        size_bytes = t["size_bytes"]
        aligned_size = align_up(size_bytes, alignment)
        entry = {
            "name": t["name"],
            "ssd_offset": offset,
            "size_bytes": size_bytes,
            "aligned_size_bytes": aligned_size,
        }
        if t.get("dram_base") is not None:
            entry["dram_base"] = t["dram_base"]
            entry["dram_end"] = t["dram_end"]
        entries.append(entry)
        offset += aligned_size
    return entries, offset - layer_base_offset


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Decoder layer index. Used for naming, for filtering --from-trace, "
        "and (with --layer-stride) to compute the base offset. Default 0.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=sorted(DTYPE_BYTES),
        help="Weight dtype. Only used when NOT reading real sizes via --from-trace.",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=16384,
        help="Byte alignment applied to each tensor's end offset. Default 16384 "
        "(SimpleSSD's default PAL PageSize -- see simplessd/config/sample.cfg's "
        "[pal] PageSize). Must be a multiple of the SSD's LBA size (512).",
    )
    parser.add_argument(
        "--base-offset",
        type=int,
        default=None,
        help="Explicit SSD byte offset for this layer's first tensor. "
        "Overrides --layer-stride.",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=None,
        help="If set (and --base-offset isn't), base offset = layer_idx * layer_stride. "
        "Use this to reserve a fixed-size slot per layer when laying out a whole "
        "model; round it up to at least one layer's aligned span (see "
        "layer_span_bytes in this script's own output).",
    )
    parser.add_argument(
        "--from-trace",
        default=None,
        help="Path to a model_weight_ranges.txt produced by "
        "PyTorchSim/tests/Llama/test_llama2_7B.py's _dump_module_weight_ranges(). "
        "When given, real captured tensor names/sizes are used instead of the "
        "hardcoded Llama2-7B architecture table.",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Prefix to filter/strip from --from-trace's dumped names, including "
        "the trailing dot (e.g. 'model.layers.5.'). Defaults to "
        "'model.layers.<layer-idx>.' (Llama's convention) when not given -- pass "
        "this explicitly for other architectures, e.g. GPT-NeoX-20B dumps use "
        "'gpt_neox.layers.<N>.' (see tests/GPT/test_GPT_NeoX_20B.py).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON mapping to this path (default: stdout only).",
    )
    parser.add_argument(
        "--offsets-tsv",
        default=None,
        help="Also write a plain '<name> <offset> <size_bytes>' TSV (one line per "
        "tensor, unaligned size_bytes -- the real transfer length) for "
        "SimpleSSD-Standalone/sim/layer_load_main.cc's simplessd-layerload tool "
        "to consume directly, without needing a JSON parser in C++.",
    )
    parser.add_argument(
        "--pytorchsim-offsets-tsv",
        default=None,
        help="Also write a '<name> <dram_base> <dram_end> <ssd_offset> <ssd_length>' "
        "TSV for sim/legosim_pytorchsim_main.cc's live bridge (see this script's "
        "module docstring). Requires --from-trace, since dram_base/dram_end must "
        "be real captured addresses.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.alignment % 512 != 0:
        parser.error("--alignment must be a multiple of 512 (SSD LBA size)")

    if args.pytorchsim_offsets_tsv and not args.from_trace:
        parser.error("--pytorchsim-offsets-tsv requires --from-trace (needs real DRAM addresses)")

    if args.from_trace:
        tensors = load_tensors_from_trace(
            args.from_trace, args.layer_idx, name_prefix=args.name_prefix
        )
    else:
        dtype_bytes = DTYPE_BYTES[args.dtype]
        tensors = [
            {
                "name": name,
                "size_bytes": numel(shape) * dtype_bytes,
                "dram_base": None,
                "dram_end": None,
            }
            for name, shape in LLAMA2_7B_LAYER_TENSORS
        ]

    if args.base_offset is not None:
        base_offset = args.base_offset
    elif args.layer_stride is not None:
        base_offset = args.layer_idx * args.layer_stride
    else:
        base_offset = 0

    entries, layer_span_bytes = build_offset_map(
        tensors, base_offset, args.alignment
    )

    result = {
        "layer_idx": args.layer_idx,
        "base_offset": base_offset,
        "alignment": args.alignment,
        "raw_total_bytes": sum(e["size_bytes"] for e in entries),
        "layer_span_bytes": layer_span_bytes,
        "tensors": entries,
    }

    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(
            "Wrote %d tensor offsets for layer %d to %s"
            % (len(entries), args.layer_idx, args.output),
            file=sys.stderr,
        )
    if args.offsets_tsv:
        with open(args.offsets_tsv, "w") as f:
            for e in entries:
                f.write("%s\t%d\t%d\n" % (e["name"], e["ssd_offset"], e["size_bytes"]))
        print(
            "Wrote %d tensor offsets for layer %d to %s"
            % (len(entries), args.layer_idx, args.offsets_tsv),
            file=sys.stderr,
        )
    if args.pytorchsim_offsets_tsv:
        with open(args.pytorchsim_offsets_tsv, "w") as f:
            for e in entries:
                f.write(
                    "%s\t%d\t%d\t%d\t%d\n"
                    % (e["name"], e["dram_base"], e["dram_end"], e["ssd_offset"], e["size_bytes"])
                )
        print(
            "Wrote %d tensor offsets for layer %d to %s"
            % (len(entries), args.layer_idx, args.pytorchsim_offsets_tsv),
            file=sys.stderr,
        )
    print(text)


if __name__ == "__main__":
    main()
