"""
Token generation using the real GPT-NeoX-20B pretrained weights loaded from HuggingFace
transformers. CPU by default; pass --stream_layers --npu to stream layer weights onto the NPU
simulator instead (see generate_streamed_npu).

Usage:
    python generate.py [--model_id <hf_model_id>] [--prompt <text>]
                       [--max_new_tokens <n>] [--temperature <t>]
                       [--top_k <k>] [--top_p <p>]
                       [--stream_layers] [--npu]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from transformers.masking_utils import create_causal_mask
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding


DEFAULT_MODEL_ID = "EleutherAI/gpt-neox-20b"
DEFAULT_PROMPT = "Once upon a time in a land far away,"

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(model_id: str, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model onto CPU (this may take several minutes for 20B) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s  (dtype={dtype})")

    if device != "cpu":
        print(f"Moving model to {device} ...")
        model = model.to(dtype=dtype, device=torch.device(device))

    return tokenizer, model


@torch.no_grad()
def generate(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.0,
    dtype: str = "float32",
    device: str = "cpu",
):
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    print(f"Generating with model {model_id} (dtype={torch_dtype}, device={device}) ...")
    tokenizer, model = load_model(model_id, dtype=torch_dtype, device=device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.device(device))
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")
    print("Generating...")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\nOutput:\n{generated_text}")


class StreamedCheckpointLoader:
    """Reads weights straight out of the HF safetensors shards, one submodule at a time, so the
    full ~40GB GPT-NeoX-20B checkpoint never has to be materialized in memory at once."""

    def __init__(self, model_id_or_path):
        self.snapshot_dir = (
            model_id_or_path
            if os.path.isdir(model_id_or_path)
            else snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
        )
        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            shard_name = "model.safetensors"
            with safe_open(os.path.join(self.snapshot_dir, shard_name), framework="pt") as f:
                self.weight_map = {name: shard_name for name in f.keys()}
        self._handles = {}

    def _handle(self, filename):
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(os.path.join(self.snapshot_dir, filename), framework="pt")
            self._handles[filename] = handle
        return handle

    def load_module_(self, module, prefix, device, dtype):
        """Materializes `module`'s parameters from the checkpoint onto `device`."""
        for name, _ in list(module.named_parameters(recurse=True)):
            full_name = f"{prefix}.{name}"
            tensor = self._handle(self.weight_map[full_name]).get_tensor(full_name).to(dtype=dtype)
            set_module_tensor_to_device(module, name, device, value=tensor)

    def unload_module_(self, module):
        """Frees `module`'s parameters back to the meta device (no storage retained)."""
        for name, _ in list(module.named_parameters(recurse=True)):
            set_module_tensor_to_device(module, name, "meta")


def _build_meta_model(config, torch_dtype):
    """Builds the GPT-NeoX module tree on the meta device (no weight memory used yet). The rotary
    embedding is a non-persistent buffer derived from config rather than the checkpoint (built once
    at the model level as of transformers>=4.54, mirroring Llama's own earlier migration off
    per-attention-module rotary state -- GPTNeoXAttention no longer has `_init_rope`/`_init_bias` at
    all), so it comes out as a meta tensor under the meta context; rebuild it fresh outside the meta
    context to give it real values, the same way Llama's test does for `model.model.rotary_emb`."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    model.gpt_neox.rotary_emb = GPTNeoXRotaryEmbedding(config=config)
    return model


def _dump_module_weight_ranges(module, name_prefix):
    """Writes `module`'s parameter address ranges to model_weight_ranges.txt so TOGSim's live
    LegoSim SSD path can tell weight DMAs apart from activations/KV-cache by address instead of by
    name (see TOGSim/include/WeightAddressRanges.h). Same as
    tests/Llama/test_llama2_7B.py::_dump_module_weight_ranges -- duplicated here rather than
    imported, matching this file's existing StreamedCheckpointLoader duplication.

    No-op if TOGSIM_SSD_TRACE_DIR/TOGSIM_SSD_TRACE_NAME aren't set, i.e. when not running under the
    LegoSim SSD integration at all.
    """
    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        return
    out_dir = os.path.join(trace_dir, trace_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_weight_ranges.txt")
    with open(out_path, "w") as f:
        for name, p in module.named_parameters(recurse=True):
            if p is None:
                continue
            base = p.data_ptr()
            size_bytes = p.untyped_storage().size()
            end = base + size_bytes
            f.write(
                f"{name_prefix}.{name}\tbase={base}\tend={end}\tsize_bytes={size_bytes}"
                f"\tshape={tuple(p.shape)}\tdtype={p.dtype}\n"
            )
        f.flush()
        os.fsync(f.fileno())
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "../../merge_weight_ranges.py")],
        check=True,
    )
    # Rebuild the live bridge's DRAM-addr -> SSD-offset table for whichever
    # layer is now current (see sim/legosim_pytorchsim_main.cc's file
    # comment: it matches requests by DRAM address, not by name, so it needs
    # dram_base/dram_end straight from the trace we just wrote). GPT-NeoX
    # dumps use "gpt_neox.layers.<N>." rather than Llama's "model.layers.<N>.",
    # hence --name-prefix instead of relying on the script's default.
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "../../build_layer_ssd_offset_map.py"),
            "--from-trace", out_path,
            "--name-prefix", f"{name_prefix}.",
            "--pytorchsim-offsets-tsv", os.path.join(out_dir, "ssd_offsets.tsv"),
        ],
        check=True,
    )


def _prelude(base_model, input_ids, attention_mask, past_key_values, cache_position):
    """embed_in + causal-mask + rotary embeddings, split out of _forward_streamed_npu so it can be
    compiled as its own graph instead of dispatching each op eagerly."""
    inputs_embeds = base_model.embed_in(input_ids)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=base_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    position_embeddings = base_model.rotary_emb(inputs_embeds, position_ids)
    return inputs_embeds, causal_mask, position_ids, position_embeddings


def _epilogue(base_model, embed_out, hidden_states):
    """final_layer_norm + embed_out, split out of _forward_streamed_npu for the same reason as
    _prelude."""
    return embed_out(base_model.final_layer_norm(hidden_states)).float()


@torch.no_grad()
def _forward_streamed_npu(model, loader, prelude_fn, epilogue_fn, input_ids, attention_mask, past_key_values, device, dtype,
                           include_epilogue=True):
    """Same layer-streamed weight loading as _forward_streamed, but the embed/mask/rope prelude and
    the norm/embed_out epilogue each run through a caller-supplied (compiled) callable, so only the
    weight load/unload glue between layers still runs eagerly. `past_key_values` is a single shared
    `StaticCache` reused across every layer, same as tests/Llama/test_tinyllama.py.

    include_epilogue=False skips calling epilogue_fn entirely and returns the raw decoder-layer
    output instead: for a decode/prefill-only timing run scoped to a truncated number of decoder
    layers, the epilogue (final_layer_norm + embed_out) is a separate, unrelated norm and a large
    matmul against the full vocab that has nothing to do with what's being measured -- same
    reasoning as tests/Llama/test_llama2_7B.py's _forward_streamed_npu. Since torch.compile is
    lazy, never calling epilogue_fn also means that kernel is never traced/compiled/dispatched to
    the NPU simulator at all. generate_streamed_npu (real multi-token generation, which needs real
    logits to sample from every step) always uses the default True."""
    base_model = model.gpt_neox
    past_seen_tokens = int(past_key_values.get_seq_length())
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
    )

    inputs_embeds, causal_mask, position_ids, position_embeddings = prelude_fn(
        base_model, input_ids, attention_mask, past_key_values, cache_position
    )
    hidden_states = inputs_embeds

    for layer_idx, layer in enumerate(base_model.layers):
        loader.load_module_(layer, f"gpt_neox.layers.{layer_idx}", device, dtype)
        _dump_module_weight_ranges(layer, f"gpt_neox.layers.{layer_idx}")
        # GPTNeoXLayer.forward still returns a tuple (unlike Llama's now-plain-tensor return).
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            layer_past=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        loader.unload_module_(layer)

    if not include_epilogue:
        return hidden_states

    return epilogue_fn(base_model, model.embed_out, hidden_states)


@torch.no_grad()
def _forward_streamed(model, loader, input_ids, attention_mask, past_key_values, device, dtype,
                       include_epilogue=True):
    """Equivalent to GPTNeoXForCausalLM.forward(), except each decoder layer's weights are loaded
    from disk right before it runs and discarded right after, so only one layer's worth of weights
    (~450MB for GPT-NeoX-20B) is resident at a time instead of the full ~40GB model.

    `past_key_values` is a single shared `StaticCache` (or `DynamicCache`) reused across every
    layer -- each layer's `GPTNeoXAttention.update()` call routes to its own slot internally via
    its own real `layer_idx`, the same way Llama's `_forward_streamed` (tests/Llama/test_tinyllama.py)
    works. GPT-NeoX's `_attn_projections_and_rope`/`_attn` submethods that `SimGPTNeoXCache` used to
    patch onto no longer exist in transformers>=4.54 (folded into `GPTNeoXAttention.forward`
    directly, mirroring Llama's own Cache-API migration), so this now goes through the model's
    native cache handling instead.

    include_epilogue=False skips final_layer_norm + embed_out and returns the raw decoder-layer
    output instead -- see _forward_streamed_npu's docstring for why.
    """
    base_model = model.gpt_neox
    inputs_embeds = base_model.embed_in(input_ids)

    past_seen_tokens = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=base_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids)

    for layer_idx, layer in enumerate(base_model.layers):
        loader.load_module_(layer, f"gpt_neox.layers.{layer_idx}", device, dtype)
        # GPTNeoXLayer.forward still returns a tuple (unlike Llama's now-plain-tensor return),
        # so outputs[0] is still correct here.
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            layer_past=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        loader.unload_module_(layer)

    if not include_epilogue:
        return hidden_states

    hidden_states = base_model.final_layer_norm(hidden_states)
    logits = model.embed_out(hidden_states).float()
    return logits


def _sample_next_token(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_streamed(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 3,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    dtype: str = "bfloat16",
):
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    device = torch.device("cpu")
    print(f"Streaming generation with model {model_id} (dtype={torch_dtype}, layer-by-layer on CPU) ...")

    print(f"Loading tokenizer/config from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    print("Building model skeleton on the meta device (no weights loaded yet)")
    model = _build_meta_model(config, torch_dtype)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)

    print("Loading persistent weights (embed_in, final_layer_norm, embed_out)")
    loader.load_module_(model.gpt_neox.embed_in, "gpt_neox.embed_in", device, torch_dtype)
    loader.load_module_(model.gpt_neox.final_layer_norm, "gpt_neox.final_layer_norm", device, torch_dtype)
    loader.load_module_(model.embed_out, "embed_out", device, torch_dtype)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")

    gen_ids = input_ids
    gen_mask = torch.ones_like(gen_ids)
    max_cache_len = input_ids.shape[1] + max_new_tokens
    print("Using StaticCache for the KV cache")
    past_key_values = StaticCache(
        config=config, max_batch_size=input_ids.shape[0], max_cache_len=max_cache_len, device=device, dtype=torch_dtype
    )
    print("Generating (streaming one decoder layer at a time)...")
    for step in range(max_new_tokens):
        step_input_ids = gen_ids if step == 0 else gen_ids[:, -1:]
        logits = _forward_streamed(
            model, loader, step_input_ids, gen_mask, past_key_values, device, torch_dtype
        )
        next_token = _sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            print("EOS reached, stopping early.")
            break

        generated_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nOutput:{generated_text}")


@torch.no_grad()
def generate_streamed_npu(
    device,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 3,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    dtype: str = "bfloat16",
    num_layers: int = None,
):
    """Same layer-streamed loading as generate_streamed, but each decoder layer's weights are
    streamed straight onto the NPU (simulated DRAM) instead of host CPU memory, and each layer's
    forward is compiled once so weight-swapping happens between compiled calls rather than inside a
    single whole-model graph -- the same structure as Llama's run_tinyllama_gen_streamed_npu
    (tests/Llama/test_tinyllama.py), using a real `StaticCache` and `create_causal_mask` now that
    GPT-NeoX's own attention code natively supports the Cache API (transformers>=4.54) instead of
    the growing tuple + torch.cat it used to use.

    `num_layers`, if given, truncates `config.num_hidden_layers` before building the model -- lets
    a fast 1-2 layer smoke test exercise the full NPU compile/simulate pipeline without paying for
    every layer (useful since GPT-NeoX-20B has 44 layers and this is a cold-compile-per-kernel path).
    """
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    print(f"\n[Running GPT-NeoX-20B streamed-layer NPU Test]")

    print(f"Loading tokenizer/config from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    if num_layers is not None:
        print(f"Truncating to {num_layers} layer(s) for a fast smoke test")
        config.num_hidden_layers = num_layers

    print("Building model skeleton on the meta device (no weights loaded yet)")
    model = _build_meta_model(config, torch_dtype)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)

    print("Loading persistent weights (embed_in, final_layer_norm, embed_out) onto the NPU")
    loader.load_module_(model.gpt_neox.embed_in, "gpt_neox.embed_in", device, torch_dtype)
    loader.load_module_(model.gpt_neox.final_layer_norm, "gpt_neox.final_layer_norm", device, torch_dtype)
    loader.load_module_(model.embed_out, "embed_out", device, torch_dtype)

    # The rotary table is a config-derived (non-persistent) buffer, not part of the checkpoint, so
    # the loader never touches it -- move it onto the NPU by hand (built fresh on CPU by
    # _build_meta_model; there's no per-layer bias buffer to move anymore, unlike the old
    # transformers version this file was originally written against).
    model.gpt_neox.rotary_emb = model.gpt_neox.rotary_emb.to(device=device)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")
    gen_mask = torch.ones_like(input_ids)

    max_cache_len = input_ids.shape[1] + max_new_tokens
    print("Using StaticCache for the KV cache")
    past_key_values = StaticCache(
        config=config, max_batch_size=input_ids.shape[0], max_cache_len=max_cache_len, device=device, dtype=torch_dtype
    )

    # Every layer's distinct layer_idx forces its own specialized compile of the shared
    # GPTNeoXLayer.forward code object, as does prefill vs. decode shape -- same reasoning as
    # Llama's run_tinyllama_gen_streamed_npu. Raise the limit so every layer actually gets compiled.
    torch._dynamo.config.recompile_limit = max(256, config.num_hidden_layers * 4)

    print("Compiling each decoder layer once (weights are swapped between calls, not the graph)")
    for layer in model.gpt_neox.layers:
        layer.forward = torch.compile(layer.forward, dynamic=False)

    compiled_prelude = torch.compile(_prelude, dynamic=False)
    compiled_epilogue = torch.compile(_epilogue, dynamic=False)

    # Patch TOGSimulator.launch_kernel to trace every NPU dispatch
    from Simulator.simulator import TOGSimulator
    _dispatch_log = []
    _orig_launch = TOGSimulator.launch_kernel
    def _traced_launch(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        _dispatch_log.append(tog_path)
        print(f"  [NPU dispatch #{len(_dispatch_log)}] {tog_path}")
        return _orig_launch(self, device_index, stream_index, tog_path, attribute_path, timestamp)
    TOGSimulator.launch_kernel = _traced_launch

    gen_ids = input_ids
    print("Generating on NPU (streaming one decoder layer at a time)...")
    for step in range(max_new_tokens):
        step_input_ids = gen_ids if step == 0 else gen_ids[:, -1:]
        logits = _forward_streamed_npu(
            model, loader, compiled_prelude, compiled_epilogue,
            step_input_ids, gen_mask, past_key_values, device, torch_dtype,
        )
        next_token = _sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
        print(f"Step {step}: outputs={tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
        if next_token.item() == tokenizer.eos_token_id:
            print("[NPU streamed] EOS reached, stopping early.")
            break

    TOGSimulator.launch_kernel = _orig_launch
    print(f"\n[NPU dispatch summary] {len(_dispatch_log)} kernel(s) launched through NPU simulator")


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-NeoX-20B CPU token generation")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or npu:0)")
    parser.add_argument("--stream_layers", action="store_true",
                         help="Load one decoder layer's weights at a time instead of the whole model "
                              "(CPU by default, or NPU if combined with --npu)")
    parser.add_argument("--npu", action="store_true",
                         help="With --stream_layers, stream layers onto the NPU instead of the CPU")
    parser.add_argument("--num_layers", type=int, default=None,
                         help="With --npu, truncate config.num_hidden_layers to this many layers "
                              "for a fast smoke test instead of compiling the full model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))

    if args.npu:
        torch.compiler.is_compiling = lambda: True  # FIXME. How to fix this?
        generate_streamed_npu(
            device=torch.device("npu:0"),
            prompt=args.prompt,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
            num_layers=args.num_layers,
        )
    elif args.stream_layers:
        generate_streamed(
            prompt=args.prompt,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
        )
    else:
        generate(
            prompt=args.prompt,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
            device=args.device,
        )
