import os
import sys
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

sys.path.append(os.path.dirname(__file__))
from test_llama2_7B import (
    StreamedCheckpointLoader,
    _forward_streamed,
    _forward_streamed_npu,
    _prelude,
    _epilogue,
)

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _build_model_and_loader(model_id, dtype, device, num_layers):
    """Same meta-skeleton + streamed-checkpoint setup as test_llama2_7B.py -- weights are still
    real checkpoint values (only the token ids and, for decode, the KV cache are randomized), so
    the layer shapes and DMA sizes this produces match a real run exactly."""
    torch_dtype = DTYPE_MAP.get(dtype, torch.float32)

    config = AutoConfig.from_pretrained(model_id)
    if num_layers is not None:
        print(f"Truncating to {num_layers} layer(s)")
        config.num_hidden_layers = num_layers

    print("Building model skeleton on the meta device (no weights loaded yet)")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config).to(device=device)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)
    print("Loading persistent weights (embed_tokens, norm, lm_head)")
    loader.load_module_(model.model.embed_tokens, "model.embed_tokens", device, torch_dtype)
    loader.load_module_(model.model.norm, "model.norm", device, torch_dtype)
    loader.load_module_(model.lm_head, "lm_head", device, torch_dtype)
    return model, config, torch_dtype, loader


def _random_input_ids(batch, q_len, vocab_size, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch, q_len), generator=gen, dtype=torch.long)
    return input_ids.to(device), gen


def _fill_random_kv(past_key_values, context_len, gen, torch_dtype, device):
    """Stands in for a prefill that was never actually run: fills every layer's cache with random
    (nonzero) keys/values for positions [0, context_len). StaticLayer.get_seq_length() detects the
    context length by scanning for nonzero entries, so this makes the cache look exactly like the
    result of a real context_len-token prefill to everything downstream (attention shapes, DMA
    sizes, cache_position derivation) without having to spend the time actually running one."""
    for layer in past_key_values.layers:
        batch, num_heads, _, head_dim = layer.keys.shape
        k = torch.randn((batch, num_heads, context_len, head_dim), generator=gen)
        v = torch.randn((batch, num_heads, context_len, head_dim), generator=gen)
        layer.keys[:, :, :context_len, :].copy_(k.to(dtype=torch_dtype, device=device))
        layer.values[:, :, :context_len, :].copy_(v.to(dtype=torch_dtype, device=device))


def _compile_for_npu(model, config):
    torch._dynamo.config.recompile_limit = max(256, config.num_hidden_layers * 4)
    for decoder_layer in model.model.layers:
        decoder_layer.forward = torch.compile(decoder_layer.forward, dynamic=False)
    compiled_prelude = torch.compile(_prelude, dynamic=False)
    compiled_epilogue = torch.compile(_epilogue, dynamic=False)
    return compiled_prelude, compiled_epilogue


def _run_forward(model, loader, input_ids, attention_mask, past_key_values, device, torch_dtype, config, npu,
                  use_togsimulator=True):
    if not npu:
        return _forward_streamed(model, loader, input_ids, attention_mask, past_key_values, device, torch_dtype)
    compiled_prelude, compiled_epilogue = _compile_for_npu(model, config)
    if not use_togsimulator:
        # Every kernel dispatches through TOGSimulator.run_standalone() (a
        # fresh TOGSim process per kernel) instead -- useful to fall back to
        # when the persistent-process path below isn't wanted (e.g. per-kernel
        # isolation/timeout, as autotune relies on run_standalone() for).
        return _forward_streamed_npu(
            model, loader, compiled_prelude, compiled_epilogue,
            input_ids, attention_mask, past_key_values, device, torch_dtype,
        )
    # Wrap the actual forward call (not the torch.compile() calls above, which
    # are lazy and don't dispatch anything by themselves) in a TOGSimulator
    # context so this phase's kernels dispatch through
    # TOGSimulator.launch_kernel() into a single persistent TOGSim process
    # (Simulator/simulator.py), instead of TOGSimulator.run_standalone()
    # spawning a fresh TOGSim process per kernel -- same fix as
    # test_llama2_7B.py's decode loop. extension_codecache.py's
    # run_kernel_simulation only takes the persistent-process path when
    # torch.npu.get_tog_simulator() is non-None, which requires an active
    # TOGSimulator context. Covers run_prefill/run_decode/run_compare alike,
    # since they all call through here for their NPU dispatch.
    from Simulator.simulator import TOGSimulator
    with TOGSimulator():
        return _forward_streamed_npu(
            model, loader, compiled_prelude, compiled_epilogue,
            input_ids, attention_mask, past_key_values, device, torch_dtype,
        )


def _build_phase_inputs(phase, config, batch, length, seed, torch_dtype, device):
    """Builds the (input_ids, attention_mask, past_key_values) triple for one phase. `length` is
    the prompt length for prefill, or the pre-existing (randomly filled) context length for
    decode. Same seed + same shapes always produces the same random values (the generator is
    seeded fresh from `seed` each call), which is what lets run_compare() give CPU and NPU
    identical inputs to diff against each other."""
    if phase == "prefill":
        input_ids, _ = _random_input_ids(batch, length, config.vocab_size, seed, device)
        attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)
        past_key_values = StaticCache(
            config=config, max_batch_size=batch, max_cache_len=length, device=device, dtype=torch_dtype
        )
    else:
        input_ids, gen = _random_input_ids(batch, 1, config.vocab_size, seed, device)
        attention_mask = torch.ones((batch, length + 1), dtype=torch.long, device=device)
        past_key_values = StaticCache(
            config=config, max_batch_size=batch, max_cache_len=length + 1, device=device, dtype=torch_dtype
        )
        print(f"Pre-filling KV cache with random values for {length} context token(s)")
        _fill_random_kv(past_key_values, length, gen, torch_dtype, device)
    return input_ids, attention_mask, past_key_values


@torch.no_grad()
def run_prefill(model_id, seq_len, batch, dtype, num_layers, device, npu, seed, use_togsimulator=True):
    label = "NPU" if npu else "CPU"
    print(f"\n[Running Llama-2-7B PREFILL-only phase, {label}, seq_len={seq_len}, batch={batch}]")
    model, config, torch_dtype, loader = _build_model_and_loader(model_id, dtype, device, num_layers)
    input_ids, attention_mask, past_key_values = _build_phase_inputs(
        "prefill", config, batch, seq_len, seed, torch_dtype, device
    )
    logits = _run_forward(model, loader, input_ids, attention_mask, past_key_values, device, torch_dtype, config, npu,
                           use_togsimulator)
    print(f"[Prefill] done. logits shape={tuple(logits.shape)}")


@torch.no_grad()
def run_decode(model_id, context_len, batch, dtype, num_layers, device, npu, seed, use_togsimulator=True):
    label = "NPU" if npu else "CPU"
    print(f"\n[Running Llama-2-7B DECODE-only phase, {label}, context_len={context_len}, batch={batch}]")
    model, config, torch_dtype, loader = _build_model_and_loader(model_id, dtype, device, num_layers)
    input_ids, attention_mask, past_key_values = _build_phase_inputs(
        "decode", config, batch, context_len, seed, torch_dtype, device
    )
    logits = _run_forward(model, loader, input_ids, attention_mask, past_key_values, device, torch_dtype, config, npu,
                           use_togsimulator)
    print(f"[Decode] done. logits shape={tuple(logits.shape)}")


def _report_comparison(name, out, ref, rtol, atol):
    out_cpu, ref_cpu = out.float().cpu(), ref.float().cpu()
    passed = torch.allclose(out_cpu, ref_cpu, rtol=rtol, atol=atol)
    max_abs_diff = (out_cpu - ref_cpu).abs().max().item()
    status = "PASSED" if passed else "FAILED"
    print(f"[{name}] {status} (rtol={rtol}, atol={atol}, max_abs_diff={max_abs_diff:.6g})")
    return passed


def _kv_head_shape(config):
    num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    return num_kv_heads, head_dim


def _generate_phase_content(phase, config, batch, length, seed):
    """Generates the phase's random content (token ids and, for decode, the KV pre-fill) exactly
    once, on CPU. run_compare() then moves this same data onto each device instead of calling
    into the RNG a second time -- two independently-seeded generators would very likely agree,
    but there's no reason to depend on that when the tensors can just be reused directly."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if phase == "prefill":
        input_ids = torch.randint(0, config.vocab_size, (batch, length), generator=gen, dtype=torch.long)
        return input_ids, None
    input_ids = torch.randint(0, config.vocab_size, (batch, 1), generator=gen, dtype=torch.long)
    num_kv_heads, head_dim = _kv_head_shape(config)
    kv_fill = [
        (
            torch.randn((batch, num_kv_heads, length, head_dim), generator=gen),
            torch.randn((batch, num_kv_heads, length, head_dim), generator=gen),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return input_ids, kv_fill


def _place_phase_content(phase, config, batch, length, input_ids, kv_fill, torch_dtype, device):
    input_ids = input_ids.to(device)
    if phase == "prefill":
        attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)
        past_key_values = StaticCache(
            config=config, max_batch_size=batch, max_cache_len=length, device=device, dtype=torch_dtype
        )
    else:
        attention_mask = torch.ones((batch, length + 1), dtype=torch.long, device=device)
        past_key_values = StaticCache(
            config=config, max_batch_size=batch, max_cache_len=length + 1, device=device, dtype=torch_dtype
        )
        for layer, (k, v) in zip(past_key_values.layers, kv_fill):
            layer.keys[:, :, :length, :].copy_(k.to(dtype=torch_dtype, device=device))
            layer.values[:, :, :length, :].copy_(v.to(dtype=torch_dtype, device=device))
    return input_ids, attention_mask, past_key_values


@torch.no_grad()
def run_compare(model_id, phase, length, batch, dtype, num_layers, seed, rtol=1e-3, atol=1e-3, use_togsimulator=True):
    """Runs the same phase (identical random input ids and, for decode, identical random KV
    cache, generated exactly once and placed on each device) on CPU and NPU, then diffs the
    resulting logits. Only meaningful with pytorchsim_functional_mode enabled: with it off the
    NPU path returns all-zero output (spike never runs), so every comparison would trivially fail."""
    print(f"\n[Comparing CPU vs NPU, phase={phase}, length={length}, batch={batch}]")

    print("--- CPU reference ---")
    cpu_device = torch.device("cpu")
    model, config, torch_dtype, loader = _build_model_and_loader(model_id, dtype, cpu_device, num_layers)
    base_input_ids, base_kv_fill = _generate_phase_content(phase, config, batch, length, seed)
    input_ids, attention_mask, past_key_values = _place_phase_content(
        phase, config, batch, length, base_input_ids, base_kv_fill, torch_dtype, cpu_device
    )
    cpu_logits = _run_forward(
        model, loader, input_ids, attention_mask, past_key_values, cpu_device, torch_dtype, config, npu=False
    )

    print("--- NPU ---")
    torch.compiler.is_compiling = lambda: True  # FIXME. How to fix this?
    npu_device = torch.device("npu:0")
    model, config, torch_dtype, loader = _build_model_and_loader(model_id, dtype, npu_device, num_layers)
    input_ids, attention_mask, past_key_values = _place_phase_content(
        phase, config, batch, length, base_input_ids, base_kv_fill, torch_dtype, npu_device
    )
    npu_logits = _run_forward(
        model, loader, input_ids, attention_mask, past_key_values, npu_device, torch_dtype, config, npu=True,
        use_togsimulator=use_togsimulator,
    )

    _report_comparison(f"{phase} logits (CPU vs NPU)", npu_logits, cpu_logits, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a Llama-2-7B prefill or decode phase in isolation. Token ids and "
                     "(for decode) the pre-existing KV cache are random with a fixed seed -- "
                     "functional correctness isn't checked, only timing."
    )
    parser.add_argument("--phase", type=str, required=True, choices=["prefill", "decode"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=500, help="prefill: number of prompt tokens to simulate")
    parser.add_argument("--context_len", type=int, default=500,
                         help="decode: number of prior tokens the single decode step attends over "
                              "(the KV cache is randomly pre-filled to this length instead of being "
                              "produced by a real prefill)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_layers", type=int, default=1,
                         help="Truncate config.num_hidden_layers to this many layers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--compare", action="store_true",
                         help="Run both CPU and NPU for the chosen --phase and diff the logits, "
                              "instead of running a single device path. Requires "
                              "pytorchsim_functional_mode=1 in the TOGSim config, otherwise the "
                              "NPU output is all zeros and the comparison is meaningless.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--togsimulator", action=argparse.BooleanOptionalAction, default=True,
                         help="Wrap the NPU forward in a TOGSimulator() context so its kernels "
                              "dispatch through a single persistent TOGSim process instead of "
                              "TOGSimulator.run_standalone() spawning a fresh TOGSim process per "
                              "kernel. Default on; pass --no-togsimulator for the per-kernel path "
                              "(e.g. for per-kernel isolation/timeout). No effect without --npu.")
    parser.add_argument("--ssd-backend", choices=["simplessd", "formula"], default=None,
                         help="Which LegoSim SSD phase1 process to use when TOGSIM_LEGOSIM_SSD=1 "
                              "(see extension_config.py's CONFIG_LEGOSIM_SSD_BACKEND): 'simplessd' "
                              "(the default in extension_config.py) runs every weight read through "
                              "a real cycle-accurate SimpleSSD engine; 'formula' falls back to the "
                              "original ssd_simlet.cpp bandwidth+base-latency placeholder. Sets the "
                              "TOGSIM_LEGOSIM_SSD_BACKEND env var; leave unset to use whatever's "
                              "already in the environment (or the default).")
    parser.add_argument("--ssd-yaml", type=str, default=None,
                         help="Path to the interchiplet phase1 process entry (cmd/args/log/"
                              "clock_rate) used for the 'simplessd' --ssd-backend -- see "
                              "configs/legosim/simplessd.yml (the default) for the expected shape. "
                              "Only takes effect with --ssd-backend simplessd (or "
                              "TOGSIM_LEGOSIM_SSD_BACKEND=simplessd). Sets the "
                              "TOGSIM_LEGOSIM_SSD_YAML env var; leave unset to use the default.")
    parser.add_argument("--ssd-channels", type=int, default=None,
                         help="How many NAND flash chiplets to run, one per flash channel (see "
                              "TOGSim/include/SsdLegoSimLink.h). Only applies to --ssd-backend "
                              "simplessd. Leave unset to take the count from [pal] Channel in the "
                              "SimpleSSD device config named by the --ssd-yaml entry, which is the "
                              "arrangement to prefer -- overriding it here models a device whose "
                              "config says otherwise. Sets TOGSIM_LEGOSIM_SSD_NUM_CHANNELS.")
    parser.add_argument("--zero-compute", action="store_true",
                         help="Charge zero cycles for every NPU compute instruction, leaving only "
                              "data movement (see TOGSim/include/ZeroComputeMode.h). Use it to get "
                              "the lower bound the memory system alone imposes -- how much of the "
                              "runtime is flash, not the systolic array. Sets TOGSIM_ZERO_COMPUTE.")
    args = parser.parse_args()

    if args.ssd_backend:
        os.environ["TOGSIM_LEGOSIM_SSD_BACKEND"] = args.ssd_backend
    if args.ssd_yaml:
        os.environ["TOGSIM_LEGOSIM_SSD_YAML"] = args.ssd_yaml
    if args.ssd_channels:
        os.environ["TOGSIM_LEGOSIM_SSD_NUM_CHANNELS"] = str(args.ssd_channels)
    if args.zero_compute:
        os.environ["TOGSIM_ZERO_COMPUTE"] = "1"

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))

    if args.compare:
        length = args.seq_len if args.phase == "prefill" else args.context_len
        run_compare(
            args.hf_model, args.phase, length, args.batch, args.dtype, args.num_layers,
            args.seed, args.rtol, args.atol, use_togsimulator=args.togsimulator,
        )
    else:
        device = torch.device("npu:0") if args.npu else torch.device("cpu")
        if args.npu:
            torch.compiler.is_compiling = lambda: True  # FIXME. How to fix this?
        if args.phase == "prefill":
            run_prefill(args.hf_model, args.seq_len, args.batch, args.dtype, args.num_layers, device, args.npu, args.seed,
                        use_togsimulator=args.togsimulator)
        else:
            run_decode(args.hf_model, args.context_len, args.batch, args.dtype, args.num_layers, device, args.npu, args.seed,
                       use_togsimulator=args.togsimulator)
