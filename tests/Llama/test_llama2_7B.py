import os
import json
import subprocess, sys
import argparse
import copy
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel


class StreamedCheckpointLoader:
    """Reads weights straight out of the HF safetensors shards, one submodule at a time,
    so the full checkpoint never has to be materialized in memory at once."""

    def __init__(self, model_id):
        self.snapshot_dir = (
            model_id
            if os.path.isdir(model_id)
            else snapshot_download(model_id, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
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


class SimStaticCache(StaticCache):
    """Fixed-shape, pre-allocated KV cache, like HF's `StaticCache`, but without the things in it
    that don't lower cleanly through the NPU simulator: `torch._dynamo.mark_static_address`
    (a dynamo-only bookkeeping call), `index_copy_` with a device-side index tensor, and deriving
    the write offset from `cache_position` via `int(cache_position[0])` inside `update()`.

    That last one matters under `torch.compile`: `int(...)` on a tensor forces a data-dependent
    host sync. Left in, Dynamo either graph-breaks on it (and the resumed subgraph then hits a
    real MLIR memref-aliasing bug under `torch.no_grad()`'s more aggressive buffer reuse), or,
    with `capture_scalar_outputs=True` to suppress the break, fails to guard on the resulting
    symbolic int inside `slice_forward` instead. Both were confirmed by hand against the real
    simulator. Since the caller (`_forward_streamed`/`_forward_streamed_npu`) already knows the
    write offset as a plain Python int before it ever builds a `cache_position` tensor
    (`past_key_values.get_seq_length()` returns one), `set_write_offset()` lets it hand that int
    to the cache directly, so `update()` never touches `cache_position` or calls `.item()` at all.

    Subclasses `StaticCache` (rather than duck-typing `Cache`) purely so that
    `LlamaModel._update_causal_mask`'s `isinstance(past_key_values, StaticCache)` check still takes
    the fixed-length mask branch, sized off `get_max_length()` instead of the growing attention mask.

    `update()` ignores the `layer_idx` argument HF's attention code passes in and uses
    `self._active_layer` instead, set via `set_active_layer()` right before each layer's forward.
    On the NPU path, every decoder layer's `self_attn.layer_idx` is forced to 0 (see
    run_llama_gen_streamed_npu) so all 32 layers share Dynamo guards and compile to a single
    reused kernel instead of 32 separate specializations -- which means the `layer_idx` that
    actually reaches `update()` is meaningless there, and the real target has to come from
    outside the compiled call instead.
    """

    def __init__(self, config, max_batch_size, max_cache_len, device, dtype=None):
        # Deliberately does not call StaticCache.__init__/Cache.__init__: skips its
        # `torch._dynamo.mark_static_address` calls, which are dynamo-only bookkeeping we don't want.
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )
        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache = [
            torch.zeros(cache_shape, dtype=self.dtype, device=device) for _ in range(config.num_hidden_layers)
        ]
        self.value_cache = [
            torch.zeros(cache_shape, dtype=self.dtype, device=device) for _ in range(config.num_hidden_layers)
        ]
        self._seen_tokens = 0
        self._write_offset = 0
        self._active_layer = 0

    def set_write_offset(self, offset: int):
        """Sets the host-side start position for the next update() call on every layer. Called
        once per forward pass (all layers write at the same offset within a single step)."""
        self._write_offset = offset

    def set_active_layer(self, real_layer_idx: int):
        """Sets which persistent cache slot the next update() call should target, keyed by the
        real layer index (not whatever `layer_idx` update() itself receives -- see class
        docstring). Called by the outer streaming loop right before each layer's forward."""
        self._active_layer = real_layer_idx

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_out = self.key_cache[self._active_layer]
        v_out = self.value_cache[self._active_layer]

        start = self._write_offset
        end = start + key_states.shape[-2]
        k_out[:, :, start:end, :] = key_states
        v_out[:, :, start:end, :] = value_states

        if self._active_layer == 0:
            self._seen_tokens = end
        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def get_max_length(self):
        return self.max_cache_len

    def reset(self):
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()
        self._seen_tokens = 0
        self._write_offset = 0
        self._active_layer = 0


@torch.no_grad()
def _forward_streamed(model, loader, input_ids, attention_mask, past_key_values, device, dtype):
    """Equivalent to LlamaForCausalLM.forward(), except each decoder layer's weights are
    loaded from disk right before it runs and discarded right after, so only one layer's
    worth of weights is resident at a time."""
    base_model = model.model
    inputs_embeds = base_model.embed_tokens(input_ids)

    # int(...): SimStaticCache.get_seq_length() already returns a plain int, but HF's real
    # StaticCache returns a tensor -- this call is eager (outside any torch.compile boundary),
    # so converting it here is a harmless host sync either way.
    past_seen_tokens = int(past_key_values.get_seq_length())
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

    if hasattr(past_key_values, "set_write_offset"):
        past_key_values.set_write_offset(past_seen_tokens)
    for layer_idx, decoder_layer in enumerate(base_model.layers):
        loader.load_module_(decoder_layer, f"model.layers.{layer_idx}", device, dtype)
        if hasattr(past_key_values, "set_active_layer"):
            past_key_values.set_active_layer(layer_idx)
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        loader.unload_module_(decoder_layer)

    hidden_states = base_model.norm(hidden_states)
    logits = model.lm_head(hidden_states).float()
    return logits


def _prelude(base_model, input_ids, attention_mask, past_key_values, cache_position):
    """embed_tokens + causal-mask + rotary embeddings, split out of _forward_streamed so the NPU
    path can compile it as its own graph instead of dispatching each op eagerly."""
    inputs_embeds = base_model.embed_tokens(input_ids)
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


def _epilogue(base_model, lm_head, hidden_states):
    """final norm + lm_head, split out of _forward_streamed for the same reason as _prelude."""
    return lm_head(base_model.norm(hidden_states)).float()


def _dump_module_weight_ranges(module, name_prefix):
    """Writes `module`'s parameter address ranges to model_weight_ranges.txt (same
    format/location run_llama_gen already uses for the whole-model case) so
    TOGSim's live LegoSim SSD path can tell weight DMAs apart from activations/KV-cache
    by address instead of by name (see TOGSim/include/WeightAddressRanges.h).

    Overwrites the file rather than appending: call this right after load_module_()
    for the layer about to run. unload_module_() frees these addresses back to the
    allocator, and the next layer's load_module_() call can reuse them, so only the
    most recently loaded layer's ranges are valid at any given moment -- stale entries
    from an already-unloaded layer would misattribute a later access to the wrong
    tensor (or the wrong tensor kind entirely, since freed memory can be reused for a
    KV-cache write instead of a weight).

    No-op if TOGSIM_SSD_TRACE_DIR/TOGSIM_SSD_TRACE_NAME aren't set, i.e. when not
    running under the LegoSim SSD integration at all.
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


@torch.no_grad()
def _forward_streamed_npu(model, loader, prelude_fn, epilogue_fn, input_ids, attention_mask, past_key_values, device, dtype):
    """Same as _forward_streamed, but the embed/mask/rope prelude and the norm/lm_head epilogue
    are each run through a caller-supplied (compiled) callable instead of plain eager ops, so only
    the necessary glue between layers (the weight load/unload calls) still runs eagerly."""
    base_model = model.model
    past_seen_tokens = int(past_key_values.get_seq_length())
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
    )

    inputs_embeds, causal_mask, position_ids, position_embeddings = prelude_fn(
        base_model, input_ids, attention_mask, past_key_values, cache_position
    )
    hidden_states = inputs_embeds

    # set_write_offset/set_active_layer only exist on SimStaticCache -- HF's real StaticCache
    # derives the write position and the real layer_idx from the calling attention module
    # directly, so neither call applies there.
    if hasattr(past_key_values, "set_write_offset"):
        past_key_values.set_write_offset(past_seen_tokens)
    for layer_idx, decoder_layer in enumerate(base_model.layers):
        loader.load_module_(decoder_layer, f"model.layers.{layer_idx}", device, dtype)
        _dump_module_weight_ranges(decoder_layer, f"model.layers.{layer_idx}")
        if hasattr(past_key_values, "set_active_layer"):
            past_key_values.set_active_layer(layer_idx)
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        loader.unload_module_(decoder_layer)

    return epilogue_fn(base_model, model.lm_head, hidden_states)


@torch.no_grad()
def run_llama_gen_streamed_cpu(
    model_id="meta-llama/Llama-2-7b-hf",
    prompt="Hello!",
    dtype="float32",
    max_new_tokens=5,
    num_layers=None,
):
    torch.manual_seed(0)
    print("\n[Running Llama-2-7B streamed-layer CPU Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    device = torch.device("cpu")

    print(f"Loading tokenizer/config from HF: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    config = AutoConfig.from_pretrained(model_id)
    if num_layers is not None:
        print(f"Truncating to {num_layers} layer(s) for a fast smoke test")
        config.num_hidden_layers = num_layers

    print("Building model skeleton on the meta device (no weights loaded yet)")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    # inv_freq is derived from config, not stored per-checkpoint, so build it fresh
    # with real (non-meta) values instead of trying to stream it from disk.
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)

    print("Loading persistent weights (embed_tokens, norm, lm_head)")
    loader.load_module_(model.model.embed_tokens, "model.embed_tokens", device, torch_dtype)
    loader.load_module_(model.model.norm, "model.norm", device, torch_dtype)
    loader.load_module_(model.lm_head, "lm_head", device, torch_dtype)

    inputs = tokenizer(prompt, return_tensors="pt")
    gen_ids = inputs["input_ids"]
    gen_mask = inputs["attention_mask"]

    max_cache_len = gen_ids.shape[1] + max_new_tokens
    print("Using StaticCache for the KV cache")
    past_key_values = StaticCache(
        config=config, max_batch_size=gen_ids.shape[0], max_cache_len=max_cache_len, device=device, dtype=torch_dtype
    )
    print("Generating on CPU (streaming one decoder layer at a time)...")
    for step in range(max_new_tokens):
        step_input_ids = gen_ids if step == 0 else gen_ids[:, -1:]
        logits = _forward_streamed(model, loader, step_input_ids, gen_mask, past_key_values, device, torch_dtype)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
        print(f"Step {step}: outputs={tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
        if next_token.item() == tokenizer.eos_token_id:
            print("[CPU streamed] EOS reached, stopping early.")
            break


@torch.no_grad()
def run_llama_gen_streamed_npu(
    device,
    model_id="meta-llama/Llama-2-7b-hf",
    prompt="Hello!",
    dtype="float32",
    max_new_tokens=5,
    num_layers=None,
):
    """Same layer-streamed loading as run_llama_gen_streamed_cpu, but each decoder layer's
    weights are streamed straight onto the NPU (simulated DRAM) instead of host CPU memory, and
    each layer's forward is compiled once so weight-swapping happens between compiled calls
    rather than inside a single whole-model graph (which wouldn't tolerate weights changing
    device/identity mid-graph).

    `num_layers`, if given, truncates `config.num_hidden_layers` before building the model -- lets
    a fast 1-2 layer smoke test (or a single-layer timing sample to extrapolate from, since every
    layer has identical shapes) exercise the full NPU compile/simulate pipeline without paying for
    all 32 layers."""
    torch.manual_seed(0)
    print("\n[Running Llama-2-7B streamed-layer NPU Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading tokenizer/config from HF: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    config = AutoConfig.from_pretrained(model_id)
    if num_layers is not None:
        print(f"Truncating to {num_layers} layer(s) for a fast smoke test")
        config.num_hidden_layers = num_layers

    print("Building model skeleton on the meta device (no weights loaded yet)")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    # inv_freq is derived from config, not stored per-checkpoint. Build it fresh with real values
    # on CPU (torch.arange under the meta context above would give it a meta inv_freq too), then
    # move the (tiny) result onto the NPU, where it needs to live permanently for RoPE.
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config).to(device=device)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)

    print("Loading persistent weights (embed_tokens, norm, lm_head) onto the NPU")
    loader.load_module_(model.model.embed_tokens, "model.embed_tokens", device, torch_dtype)
    loader.load_module_(model.model.norm, "model.norm", device, torch_dtype)
    loader.load_module_(model.lm_head, "lm_head", device, torch_dtype)

    # Dynamo treats `self.self_attn.layer_idx` (a plain int attribute read inside
    # LlamaAttention.forward) as a static guard value, so every layer's distinct layer_idx forces
    # its own specialized compile of the shared LlamaDecoderLayer.forward code object -- as does
    # prefill (q_len>1) vs. decode (q_len==1) shape, on top of that. That's on purpose (each layer
    # needs its own compiled kernel anyway for the weight-streaming glue in between), but it means
    # the number of specializations for this one code object can exceed Dynamo's default
    # cache_size_limit (8), which would otherwise make it silently give up and fall back to eager
    # for the remaining layers -- confirmed: layers past the limit dropped to 0 compiled kernels
    # and ~43 individual eager ops each instead of 1 compiled kernel. Raise the limit so every
    # layer actually gets compiled.
    
    torch._dynamo.config.recompile_limit = max(256, config.num_hidden_layers * 4)

    print("Compiling each decoder layer once (weights are swapped between calls, not the graph)")
    for decoder_layer in model.model.layers:
        decoder_layer.forward = torch.compile(decoder_layer.forward, dynamic=False)

    # embed_tokens/mask/rope and norm/lm_head don't need weight-streaming glue in between, so they
    # can each be compiled as one graph too instead of dispatching ~15 tiny ops eagerly per step.
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

    inputs = tokenizer(prompt, return_tensors="pt")
    gen_ids = inputs["input_ids"].to(device)
    gen_mask = inputs["attention_mask"].to(device)

    max_cache_len = gen_ids.shape[1] + max_new_tokens
    print("Using StaticCache for the KV cache")
    past_key_values = StaticCache(
        config=config, max_batch_size=gen_ids.shape[0], max_cache_len=max_cache_len, device=device, dtype=torch_dtype
    )

    print("Generating on NPU (streaming one decoder layer at a time)...")
    # Wrap the whole decode loop in one TOGSimulator context so every kernel
    # across every step dispatches via TOGSimulator.launch_kernel() into a
    # single persistent TOGSim process (Simulator/simulator.py), instead of
    # falling through to TOGSimulator.run_standalone(), which spawns a fresh
    # TOGSim process per kernel. extension_codecache.py's run_kernel_simulation
    # only takes the persistent-process path when torch.npu.get_tog_simulator()
    # is non-None, which requires an active TOGSimulator context -- without
    # one (as before this change), every kernel used run_standalone() instead.
    with TOGSimulator():
        for step in range(max_new_tokens):
            step_input_ids = gen_ids if step == 0 else gen_ids[:, -1:]
            logits = _forward_streamed_npu(
                model, loader, compiled_prelude, compiled_epilogue,
                step_input_ids, gen_mask, past_key_values, device, torch_dtype,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
            print(f"Step {step}: outputs={tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
            if next_token.item() == tokenizer.eos_token_id:
                print("[NPU streamed] EOS reached, stopping early.")
                break

    TOGSimulator.launch_kernel = _orig_launch
    print(f"\n[NPU dispatch summary] {len(_dispatch_log)} kernel(s) launched through NPU simulator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompt", type=str, default="Machine learning is a powerful tool that can be used to")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--npu", action="store_true",
                         help="With --stream_layers, stream layers onto the NPU instead of the CPU")
    parser.add_argument("--num_layers", type=int, default=None,
                         help="Truncate config.num_hidden_layers to this many layers for a fast "
                              "smoke test instead of compiling/running the full model")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))

    if args.npu:
        torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?
        run_llama_gen_streamed_npu(
            device=torch.device("npu:0"),
            model_id=args.hf_model,
            prompt=args.prompt,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            num_layers=args.num_layers,
        )
    else:
        run_llama_gen_streamed_cpu(
            model_id=args.hf_model,
            prompt=args.prompt,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            num_layers=args.num_layers,
        )
