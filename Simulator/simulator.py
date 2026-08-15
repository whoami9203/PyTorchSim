import os
import shlex
import ctypes
import subprocess
import signal
import re
import sys
import yaml
import time
import threading
from pathlib import Path

import torch
import numpy as np

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend import extension_config

# Configure logger for Simulator module
logger = extension_config.setup_logger()
from tqdm import tqdm


class ProgressBar:
    def __init__(self, desc, silent_mode=False, update_interval=0.5):
        self.desc = desc
        self.silent_mode = silent_mode
        self.update_interval = update_interval
        self.pbar = None
        self.finished = False
        self.progress_thread = None

    def __enter__(self):
        if not self.silent_mode:
            self.pbar = tqdm(
                desc=self.desc,
                bar_format='{desc}: {elapsed}',
                leave=False,  # Don't leave the bar when done (it will disappear)
                ncols=80,
                disable=False,
                total=100,  # Use a total for smooth animation
            )
            # Update progress bar in a separate thread
            def update_progress():
                while not self.finished:
                    self.pbar.update(1)
                    time.sleep(self.update_interval)

            self.progress_thread = threading.Thread(target=update_progress, daemon=True)
            self.progress_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished = True
        if not self.silent_mode and self.pbar is not None:
            self.pbar.close()
        return False


TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.uint8,
    torch.bfloat16: np.float16,
    torch.float16: np.float16,
}

class FunctionalSimulator():
    def __init__(self, path, key):
        self.path = path
        self.key = key

    def load_tensor(self, arg, arg_name, arg_attribute, path):
        # path = os.path.join(dump_path, arg_name, f'{n_call}.raw')
        with open(path, 'rb') as f:
            np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[arg.dtype])
            src_tensor = torch.as_strided(torch.from_numpy(np_array), arg.size(), arg.stride())
            arg.copy_(src_tensor.to(dtype=arg.dtype))

    def get_biggest_filename(self, path):
        return len(os.listdir(path))

    def write_arg(self, arg, path, name):
        dump_path = os.path.join(path, name)
        os.makedirs(dump_path, exist_ok=True)
        index = self.get_biggest_filename(dump_path)

        if (isinstance(arg, torch.Tensor)):
            data_path = os.path.join(dump_path, f'{index}.raw')
            tensor = arg.cpu().detach()
            buffer_size = tensor.untyped_storage().size()
            buffer = (ctypes.c_char * buffer_size).from_address(tensor.data_ptr())
            t_arr = np.frombuffer(buffer, dtype=TORCH_TO_NUMPY[tensor.dtype], count=buffer_size // tensor.element_size())
            t_arr.tofile(data_path)
        else:
            assert(0)
        return index

    def dump_args(self, args, arg_attributes, load_path, dump_path):
        array_size = []
        file_path = []
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            size = arg_attribute[2] if arg_attribute[1] != torch.bool else (arg_attribute[2] + 7) // 8
            array_size.append(size)
            if MLIRKernelArgs.is_mlir_arg_in(arg_attribute[0]):
                index = self.write_arg(arg, load_path, arg_name)
                file_path.append(os.path.join(load_path, arg_name, f'{index}.raw'))
            elif MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                path = os.path.join(dump_path, arg_name)
                os.makedirs(path, exist_ok=True)
                file_path.append(os.path.join(path, f'{self.get_biggest_filename(path)}.raw'))

        return array_size, file_path

    def run_spike(self, args, arg_attributes, runtime_path, binary, vectorlane_size=4, spad_info=None, cleanup=False, silent_mode=False):
        load_path = runtime_path
        dump_path = runtime_path

        target_binary = os.path.join(self.path, binary)
        objdump = f"riscv64-unknown-elf-objdump -d {target_binary} > {os.path.join(self.path, 'binary.dump')}"
        kernel_start = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}'"
        kernel_end = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}' | xargs -I {{}} awk '/{{}}/,0' {os.path.join(self.path, 'binary.dump')} | grep ret | awk 'NR==1 {{print $1}}' | awk '{{gsub(/:$/, \"\"); print}}'"

        subprocess.run(objdump, shell=True)
        kernel_start_addr = subprocess.run(kernel_start, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')
        kernel_end_addr = subprocess.run(kernel_end, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')

        _, file_path = self.dump_args(args, arg_attributes, load_path, dump_path)
        file_path_str = ' '.join(file_path)

        # Set hardware information
        spad_option = f"-m0x{0x80000000:x}:0x{100<<30:x},0x{spad_info['spad_paddr']:x}:0x{spad_info['spad_size']*vectorlane_size:x} " + \
            f"--scratchpad-base-paddr={spad_info['spad_paddr']} " + \
            f"--scratchpad-base-vaddr={spad_info['spad_vaddr']} " + \
            f"--scratchpad-size={spad_info['spad_size']} "
        vectorlane_option = f"--vectorlane-size={vectorlane_size}"
        kernel_address = f"--kernel-addr={kernel_start_addr}:{kernel_end_addr}"
        base_path= f"--base-path={runtime_path}"
        os.makedirs(os.path.join(runtime_path, "indirect_access"), exist_ok=True)
        os.makedirs(os.path.join(runtime_path, "dma_access"), exist_ok=True)
        run = f'spike --isa rv64gcv_zfh --varch=vlen:256,elen:64 {vectorlane_option} {spad_option} {kernel_address} {base_path} /workspace/riscv-pk/build/pk {target_binary} {file_path_str}'
        if not silent_mode:
            logger.debug(f"[Spike] cmd> {run}")
            logger.info("[Spike] Running Spike simulator")
        run_cmd = shlex.split(run)
        try:
            stdout_setting = subprocess.DEVNULL if silent_mode else None
            stderr_setting = subprocess.DEVNULL if silent_mode else None
            with ProgressBar("[Spike] Running simulation", silent_mode=silent_mode):
                subprocess.check_call(run_cmd, stdout=stdout_setting, stderr=stderr_setting)
        except subprocess.CalledProcessError as e:
            if not silent_mode:
                logger.error(f"[Spike] Command failed with exit code {e.returncode}")
            error_msg = ""
            if e.returncode == 200:
                error_msg = "INVALID_SPAD_ACCESS"
            elif e.returncode == 201:
                error_msg = "STACK_OVERFLOW"
            else:
                error_msg = "UNKNOWN_ERROR"
            raise RuntimeError(f"{error_msg}")

        for (arg_name, arg_attribute), arg, path in zip(arg_attributes, args, file_path):
            if MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, path)

        if cleanup:
            for path in file_path:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def get_runtime_dump_path(base_path, prefix="runtime", zfill=4):
        indices = [
            int(match.group(1))
            for d in os.listdir(base_path)
            if (match := re.fullmatch(rf"{prefix}_(\d{{{zfill}}})", d))
        ]

        max_index = max(indices, default=-1)
        next_index = max_index + 1
        folder_name = f"{prefix}_{str(next_index).zfill(zfill)}"
        full_path = os.path.join(base_path, folder_name)

        os.makedirs(full_path)
        return full_path

class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, vectorlane_size, silent_mode=False):
        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_script_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "gem5_script/script_systolic.py")
        gem5_cmd = [extension_config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path, gem5_script_path, "-c", target_binary, "--vlane", str(vectorlane_size)]

        if not silent_mode:
            logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
            logger.info("[Gem5] Gem5 simulation started")

        try:
            #with ProgressBar("[Gem5] Running simulation", silent_mode=is_dryrun):
            output = subprocess.check_output(gem5_cmd, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            sto_log_path = os.path.join(dir_path, "sto.log")
            try:
                with open(sto_log_path, "r") as f:
                    sto_log = f.read()
            except OSError:
                sto_log = "<sto.log not found>"
            logger.debug(f"[Gem5] Gem5 simulation failed. sto.log:\n{sto_log}")
            raise RuntimeError(f"Gem5 Simulation Failed.\nsto.log:\n{sto_log}")

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list

class TOGSimulator():
    TOGSIM_RESULT_PATH_KEY = "TOGSIM_RESULT_PATH"
    FINISH_STR = "Simulation finished"
    ALLOC_POOL = dict() # For eagermode buffer plan
    _TOGSIM_CONFIG_ENV_UNSET = object()
    # interchiplet's shared low-level sync API (pipe_comm.h/sync_protocol.h --
    # used identically by TOGSim's DramLegoSimLink/SsdLegoSimLink and every
    # simlet across LegoSim) unconditionally echoes its own wire protocol to
    # TOGSim's stdout: "[INTERCMD] SEND/WRITE/...", "[RESPONSE][INTERCMD]
    # RESULT/SYNC...", "Open pipe file ...", "Read/Write N B to/from ...".
    # interchiplet's bridge_thread captures that whole stdout stream verbatim
    # into proc_r*_p1_t0/togsim.log, mixing it with TOGSim's own log lines --
    # see _split_togsim_log().
    _INTERCHIPLET_LINE_RE = re.compile(
        r"^(\[INTERCMD\]|\[RESPONSE\]|Open pipe file |(Read|Write) \d+ B (from|to) )"
    )

    def __init__(self, config_path=None, togsim_path=None) -> None:
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        self.base_dir = togsim_path
        self.config_path = config_path
        self.config_yaml = self.load_yaml(self.config_path)
        self._next_kernel_id = 0  # Auto-incrementing kernel ID

        self.trace_log = "# command_type, kernel_id, device_index, stream_index, tog_path, attribute_path, timestamp\n"

        # Batching state -- kernels/syncs accumulate here instead of
        # streaming to one long-lived process; see _flush_batch() for why
        # (interchiplet's multi-round NoC convergence needs TOGSim to
        # actually exit and restart each round, which a process Python is
        # still actively feeding can't do without deadlocking). Each flush
        # is a fresh, self-contained TOGSim invocation; _checkpoint_path/
        # _ssd_trace_continue_path thread the previous flush's cycle
        # counters/SSD trace file into the next one so they stay continuous
        # despite each flush being a new process.
        self._pending_batch_lines = []
        self._checkpoint_path = None
        self._ssd_trace_continue_path = None
        self._last_result_path = None
        self._batch_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)
        self._batch_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        """Context manager entry.

        Sets ``TOGSIM_CONFIG`` to this instance's config path so that compilation
        (``extension_config`` / codegen) uses the same YAML as TOGSim. Previous
        value is restored in ``__exit__``.
        """
        if "TOGSIM_CONFIG" in os.environ:
            self._old_togsim_config_env = os.environ["TOGSIM_CONFIG"]
        else:
            self._old_togsim_config_env = self._TOGSIM_CONFIG_ENV_UNSET
        os.environ["TOGSIM_CONFIG"] = os.path.abspath(self.config_path)

        self.old_tog_simulator = torch.npu.get_tog_simulator()
        torch.npu.set_tog_simulator(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        self.until()
        torch.npu.set_tog_simulator(self.old_tog_simulator)

        if self._old_togsim_config_env is self._TOGSIM_CONFIG_ENV_UNSET:
            os.environ.pop("TOGSIM_CONFIG", None)
        else:
            os.environ["TOGSIM_CONFIG"] = self._old_togsim_config_env

    @staticmethod
    def _find_ssd_trace_path(log_bytes):
        """
        Parses "[SSD] trace enabled: <path>" out of a batch's own log (see
        SsdTraceManager's constructor, TOGSim/src/SsdTrace.cc) to learn which
        dma_trace_N.csv file it picked, so the *next* batch can continue
        appending to that same file (via TOGSIM_SSD_TRACE_CONTINUE_PATH)
        instead of starting a new numbered one. Returns None if SSD tracing
        wasn't enabled for this batch (line never printed).
        """
        m = re.search(rb"\[SSD\] trace enabled: (\S+)", log_bytes)
        return m.group(1).decode() if m else None

    def _flush_batch(self):
        """
        Run everything accumulated in self._pending_batch_lines since the
        last flush through a fresh, self-contained TOGSim (or interchiplet,
        if SSD/DRAM legosim is enabled) invocation -- structurally the same
        shape run_standalone() already uses for one kernel (write a trace
        file, run to completion, read back one result), just with however
        many kernels/syncs accumulated in this batch instead of always 1.

        A fresh process per batch (rather than one long-lived FIFO-fed
        process for the whole session) is what lets TOGSIM_LEGOSIM_DRAM_NOC's
        multi-round convergence work: interchiplet's -t N respawns every
        phase1 process (including TOGSim) fresh each round, which only works
        for a process that's actually expected to exit and restart -- not one
        Python is still streaming kernels into (that deadlocks: a respawned
        round would open a writer-less FIFO and block forever).

        self._checkpoint_path/self._ssd_trace_continue_path thread the
        previous batch's outputs into --checkpoint_in/TOGSIM_SSD_TRACE_CONTINUE_PATH
        so TOGSim's cycle counters (Simulator::save_checkpoint/load_checkpoint,
        TOGSim/src/Simulator.cc) and SSD trace file (SsdTraceManager::open_trace's
        continuation mode, TOGSim/src/SsdTrace.cc) stay continuous across
        batches despite each one being a fresh process. This is only safe
        because a batch boundary is always a DEVICE_SYNC or until() call --
        Simulator::cycle() runs until running() is false (every core, every
        partition scheduler, the interconnect, and DRAM all fully drained),
        so there's never any in-flight state left over to lose.
        """
        if not self._pending_batch_lines:
            return

        idx = TOGSimulator._next_result_index(self._batch_dir)
        trace_file_path = self._batch_dir / f"{idx}.trace"
        with open(trace_file_path, "w") as f:
            f.write("\n".join(self._pending_batch_lines) + "\n")
        self._pending_batch_lines = []

        checkpoint_out = self._batch_dir / f"{idx}.checkpoint"
        checkpoint_args = []
        if self._checkpoint_path is not None:
            checkpoint_args += ["--checkpoint_in", str(self._checkpoint_path)]
        checkpoint_args += ["--checkpoint_out", str(checkpoint_out)]

        use_legosim_ssd = extension_config.CONFIG_TOGSIM_LEGOSIM_SSD
        use_legosim_dram = extension_config.CONFIG_TOGSIM_LEGOSIM_DRAM
        use_dram_noc = use_legosim_dram and extension_config.CONFIG_TOGSIM_LEGOSIM_DRAM_NOC
        core_freq_mhz = self.config_yaml["core_freq_mhz"] if use_legosim_dram else None

        env_overrides = {}
        if self._ssd_trace_continue_path is not None:
            env_overrides["TOGSIM_SSD_TRACE_CONTINUE_PATH"] = str(self._ssd_trace_continue_path)

        if use_legosim_ssd or use_legosim_dram:
            togsim_bin = os.path.join(self.base_dir, "build/bin/Simulator")
            run_dir = self._batch_dir / f"{idx}.legosim_run"
            yaml_path = TOGSimulator._build_legosim_yaml(
                togsim_bin, os.path.join(self.base_dir, self.config_path), trace_file_path, run_dir,
                log_level=extension_config.CONFIG_TOGSIM_DEBUG_LEVEL,
                use_ssd=use_legosim_ssd, use_dram=use_legosim_dram, use_dram_noc=use_dram_noc,
                core_freq_mhz=core_freq_mhz, extra_togsim_args=checkpoint_args,
            )
            interchiplet_bin = os.path.join(extension_config.CONFIG_LEGOSIM_ROOT, "interchiplet/bin/interchiplet")
            rounds = extension_config.CONFIG_LEGOSIM_DRAM_NOC_ROUNDS if use_dram_noc else 1
            cmd = f"{interchiplet_bin} {yaml_path} -w 3 -f 2 -t {rounds}"

            path_desc = "+".join(
                p for p, on in (("SSD", use_legosim_ssd), ("DRAM", use_legosim_dram), ("NoC", use_dram_noc)) if on
            )
            logger.debug(f"[TOGSim] cmd> {cmd}")
            logger.info(f"[TOGSim] TOGSim batch {idx} started (LegoSim {path_desc} path)")

            env = TOGSimulator._legosim_env(
                use_ssd=use_legosim_ssd, use_dram=use_legosim_dram, core_freq_mhz=core_freq_mhz,
            )
            env.update(env_overrides)

            interchiplet_stdout, _ = TOGSimulator._run_interchiplet(
                shlex.split(cmd), cwd=run_dir, env=env, timeout_sec=None,
            )
            phase1_basenames = ["Simulator"]
            if use_legosim_ssd:
                phase1_basenames.append("ssd_simlet")
            if use_legosim_dram:
                phase1_basenames.append("dram_simlet")
            phase2_basenames = ["popnet"] if use_dram_noc else ["true"]
            TOGSimulator._split_interchiplet_log(run_dir, interchiplet_stdout, phase1_basenames, phase2_basenames)
            TOGSimulator._split_togsim_logs(run_dir)
            # See run_standalone()'s matching comment: with use_dram_noc, TOGSim's
            # reported cycles differ round to round (converging by the last one),
            # so always read the last round -- never round 1.
            pytorchsim_log = run_dir / f"proc_r{rounds}_p1_t0" / "pytorchsim.log"
            result_bytes = pytorchsim_log.read_bytes() if pytorchsim_log.exists() else b""
        else:
            cmd = f"{TOGSimulator.get_togsim_command(self.config_path, self.base_dir)} --models_list {trace_file_path}"
            cmd += " " + " ".join(checkpoint_args)
            if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
                cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

            logger.debug(f"[TOGSim] cmd> {cmd}")
            logger.info(f"[TOGSim] TOGSim batch {idx} started")

            env = os.environ.copy()
            env.update(env_overrides)
            completed = subprocess.run(shlex.split(cmd), capture_output=True, check=True, env=env)
            result_bytes = completed.stdout

        result_path = self._batch_dir / f"{idx}.log"
        with open(result_path, "wb") as f:
            f.write(result_bytes)
        logger.info(f'[TOGSim] Simulation log is stored to "{result_path}"')

        # Continue the chain: the next batch picks up from here.
        self._checkpoint_path = checkpoint_out
        ssd_trace_path = TOGSimulator._find_ssd_trace_path(result_bytes)
        if ssd_trace_path is not None:
            self._ssd_trace_continue_path = ssd_trace_path
        self._last_result_path = result_path

    def _send_command(self, command_type, device_index, stream_index, tog_path="", attribute_path="", timestamp=0):
        """
        Queue a command into the pending batch (run through a fresh TOGSim
        process by _flush_batch(), triggered by device_synchronize() or
        until()) instead of writing it straight to a live process's FIFO --
        see _flush_batch() for why.

        Args:
            command_type: Type of command ("LAUNCH_KERNEL" or "DEVICE_SYNC")
            device_index: Device index
            stream_index: Stream index
            tog_path: Path to TOG file (ONNX model) - empty for DEVICE_SYNC
            attribute_path: Path to attribute file - empty for DEVICE_SYNC
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this command
        """
        # Get and increment kernel ID
        kernel_id = self._next_kernel_id
        self._next_kernel_id += 1

        # Format command: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
        command = f"{command_type},{kernel_id},{device_index},{stream_index},{tog_path},{attribute_path},{timestamp}"

        self._pending_batch_lines.append(command)
        self.trace_log += command + '\n'
        logger.debug(f"[TOGSim] Queued command: {command}")
        return kernel_id

    def until(self):
        # Make sure that all kernels in the stream are finished. This calls
        # back into device_synchronize() (via torch.npu's active-simulator
        # hook), which flushes the pending batch -- the explicit
        # _flush_batch() below is a safety net for anything queued after
        # that (or if nothing ever explicitly synced at all), mirroring
        # process_trace_file()'s own EOF-triggered cycle() call.
        torch.npu.synchronize()
        self._flush_batch()

        if self.trace_log:
            log_base_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)
            log_base_dir.mkdir(parents=True, exist_ok=True)
            idx = TOGSimulator._next_result_index(log_base_dir)
            trace_path = log_base_dir / f"{idx}.trace"
            with open(trace_path, "w") as f:
                f.write(self.trace_log)
            logger.info(f'[TOGSim] Trace log is stored to "{trace_path}"')

    def launch_kernel(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        """
        Queue a kernel launch -- actually runs on the next device_synchronize()
        or until() (see _flush_batch()), not immediately.

        Args:
            device_index: Device index
            stream_index: Stream index
            tog_path: Path to TOG file (ONNX model)
            attribute_path: Path to attribute file
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this launch
        """
        return self._send_command("LAUNCH_KERNEL", device_index, stream_index, tog_path, attribute_path, timestamp)

    def device_synchronize(self, device_index):
        """
        Synchronize all streams on a device -- queues a DEVICE_SYNC and
        flushes the accumulated batch through a fresh TOGSim process (see
        _flush_batch()).

        Args:
            device_index: Device index to synchronize
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The command ID assigned to this synchronization
        """
        # For device_synchronize, stream_index is not meaningful, use 0
        kernel_id = self._send_command("DEVICE_SYNC", device_index, 0, "", "", 0)
        self._flush_batch()
        return kernel_id

    @classmethod
    def sram_alloc(cls, buf_name, addr_range):
        cls.ALLOC_POOL[buf_name] = addr_range

    @classmethod
    def sram_dealloc(cls, buf_name, addr_range):
        if buf_name in cls.ALLOC_POOL:
            del cls.ALLOC_POOL[buf_name]

    @staticmethod
    def write_kernel_attribute_file(attribute_dir, inputs, alloc_pool=None, arg_attributes=None):
        """
        Write kernel attribute YAML (address_info + sram_alloc) under attribute_dir.

        Does not require a TOGSimulator instance. alloc_pool defaults to class ALLOC_POOL.

        Args:
            attribute_dir: Directory to hold numbered attribute files (created if needed)
            inputs: Kernel input tensors (data_ptr used for address_info)
            alloc_pool: Optional dict like ALLOC_POOL; defaults to TOGSimulator.ALLOC_POOL
            arg_attributes: Optional list of [arg_name, [direction, dtype, numel, ...]]
                from MLIRKernelArgs.mlir_argdefs().  When provided, an ``arg_meta``
                section is written with direction flags and byte sizes so that
                LegOSim simlets can classify MVIN vs MVOUT arguments without
                re-parsing the tile graph ONNX.

        Returns:
            Path to the written YAML file.
        """
        if alloc_pool is None:
            alloc_pool = TOGSimulator.ALLOC_POOL
        address_info = {}
        sram_buffer = {}
        yaml_content = {}

        os.makedirs(attribute_dir, exist_ok=True)
        index = str(len(os.listdir(attribute_dir)))
        attribute_file = os.path.join(attribute_dir, index)

        for idx, tensor in enumerate(inputs):
            address_info[f"arg{idx}"] = tensor.data_ptr()
        yaml_content["address_info"] = address_info

        # Optional per-argument metadata for LegOSim simlets.
        # arg_attributes[i] == [outer_name, [direction, dtype, numel, sizes, strides]]
        if arg_attributes is not None:
            arg_meta = {}
            for idx, (tensor, attr_entry) in enumerate(zip(inputs, arg_attributes)):
                attr = attr_entry[1]  # [direction, dtype, numel, ...]
                direction = int(attr[0])
                nbytes = int(tensor.numel()) * tensor.element_size()
                arg_meta[f"arg{idx}"] = {"direction": direction, "bytes": nbytes}
            yaml_content["arg_meta"] = arg_meta

        for buf_name, range in alloc_pool.items():
            sram_buffer[buf_name] = range
        yaml_content["sram_alloc"] = sram_buffer

        with open(attribute_file, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno())
        return attribute_file

    def load_yaml(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    def get_core_freq(self):
        if "core_freq_mhz" in self.config_yaml:
            return self.config_yaml["core_freq_mhz"] * 1000 * 1000 # MHz
        else:
            raise KeyError("Key 'core_freq' not found in JSON.")

    @staticmethod
    def _next_result_index(base_dir: Path) -> int:
        """Return the next sequential index for log/trace file naming."""
        existing = set()
        for f in base_dir.iterdir():
            if f.suffix in (".log", ".trace") and f.stem.isdigit():
                existing.add(int(f.stem))
        idx = 1
        while idx in existing:
            idx += 1
        return idx

    @staticmethod
    def get_togsim_command(config_path, togsim_path=None):
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")
        bin = os.path.join(togsim_path, "build/bin/Simulator")
        config = os.path.join(togsim_path, config_path)
        cmd = f"{bin} --config {config}"
        return cmd

    @staticmethod
    def _build_legosim_yaml(togsim_bin, config, trace_file_path, run_dir, log_level="",
                             use_ssd=True, use_dram=False, use_dram_noc=False, core_freq_mhz=None,
                             extra_togsim_args=None):
        """
        Write an interchiplet benchmark YAML pairing TOGSim (phase1[0], chiplet
        (0,0)) with whichever LegoSim simlet(s) are requested: the SSD simlet
        (chiplet (1,0); see TOGSim/legosim/ssd_simlet.cpp, weight-read latency
        only) and/or the DRAM simlet (chiplet (2,0); see
        TOGSim/legosim/dram_simlet.cpp, catch-all for every other DMA access,
        replacing TOGSim's real Dram/Interconnect models).

        phase2 is normally a no-op /bin/true filler -- interchiplet indexes
        phase2[0] unconditionally even though we don't need NoC modeling here
        (each simlet's answer already carries the real latency). When
        use_dram_noc is set (only meaningful together with use_dram), phase2
        instead runs a real popnet against TOGSim/legosim/topology/dram_noc_3.gv
        (covering every chiplet coordinate this integration uses), exercising
        interchiplet's real two-phase fixed-point loop -- dram_simlet and
        DramLegoSimLink track a running timeNow (see their own comments)
        specifically so this NoC delay has somewhere to land instead of being
        silently discarded.

        TOGSim's own coordinates/peer coordinates must match SsdLegoSimLink's/
        DramLegoSimLink's env-var defaults (TOGSIM_LEGOSIM_X/Y=0,0,
        TOGSIM_SSD_LEGOSIM_X/Y=1,0, TOGSIM_DRAM_PEER_LEGOSIM_X/Y=2,0), which
        _legosim_env() sets for this same subprocess.

        TOGSim's clock_rate is set to core_freq_mhz/1000 (when use_dram --
        dram_simlet/ssd_simlet stay at 1.0, already ns-native) so
        interchiplet can correctly reconcile TOGSim's core-cycle domain
        against dram_simlet's ns domain when resolving a round trip's end
        cycle -- see DramLegoSimLink::query_latency_ns()'s comment for how
        that resolved cycle is used. SsdLegoSimLink is unaffected either
        way: it never reads back interchiplet's resolved cycle.
        """
        togsim_args = ["--config", str(config), "--models_list", str(trace_file_path)]
        if log_level:
            togsim_args += ["--log_level", log_level]
        if extra_togsim_args:
            togsim_args += list(extra_togsim_args)

        togsim_clock_rate = core_freq_mhz / 1000.0 if (use_dram and core_freq_mhz) else 1.0
        phase1 = [
            {
                "cmd": str(togsim_bin),
                "args": togsim_args,
                "log": "togsim.log",
                "is_to_stdout": False,
                "clock_rate": togsim_clock_rate,
            },
        ]

        if use_ssd:
            ssd_bin = os.path.join(os.path.dirname(togsim_bin), "ssd_simlet")
            bandwidth = extension_config.CONFIG_LEGOSIM_SSD_BANDWIDTH_GBPS
            base_latency = extension_config.CONFIG_LEGOSIM_SSD_BASE_LATENCY_NS
            phase1.append({
                "cmd": str(ssd_bin),
                "args": ["1", "0", "0", "0", str(bandwidth), str(base_latency)],
                "log": "ssd_simlet.log",
                "is_to_stdout": False,
                "clock_rate": 1.0,
            })

        if use_dram:
            dram_bin = os.path.join(os.path.dirname(togsim_bin), "dram_simlet")
            bandwidth = extension_config.CONFIG_LEGOSIM_DRAM_BANDWIDTH_GBPS
            base_latency = extension_config.CONFIG_LEGOSIM_DRAM_BASE_LATENCY_NS
            phase1.append({
                "cmd": str(dram_bin),
                "args": ["2", "0", "0", "0", str(bandwidth), str(base_latency)],
                "log": "dram_simlet.log",
                "is_to_stdout": False,
                "clock_rate": 1.0,
            })

        if use_dram_noc:
            popnet_bin = os.path.join(extension_config.CONFIG_LEGOSIM_ROOT, "popnet_chiplet/build/popnet")
            # togsim_bin is <togsim_root>/build/bin/Simulator; strip those
            # three components back to <togsim_root> to find legosim/topology/.
            togsim_root = os.path.dirname(os.path.dirname(os.path.dirname(togsim_bin)))
            topology_path = os.path.join(togsim_root, "legosim", "topology", "dram_noc_3.gv")
            phase2 = [
                {
                    "cmd": str(popnet_bin),
                    "args": [
                        "-A", "3", "-c", "1", "-V", "2", "-B", "8", "-O", "4", "-F", "2",
                        "-L", "100", "-T", "1000000", "-r", "1", "-I", "../bench.txt",
                        "-R", "4", "-G", str(topology_path), "-D", "../delayInfo.txt", "-P",
                    ],
                    "log": "popnet_0.log",
                    "is_to_stdout": False,
                    "clock_rate": 1.0,
                },
            ]
        else:
            phase2 = [
                {
                    "cmd": "/bin/true",
                    "args": [],
                    "log": "noop.log",
                    "is_to_stdout": False,
                    "clock_rate": 1.0,
                },
            ]

        yaml_doc = {
            "phase1": phase1,
            "phase2": phase2,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = run_dir / "legosim.yml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_doc, f)
        return yaml_path

    @staticmethod
    def _legosim_env(use_ssd=True, use_dram=False, core_freq_mhz=None):
        env = os.environ.copy()
        legosim_root = extension_config.CONFIG_LEGOSIM_ROOT
        env["SIMULATOR_ROOT"] = legosim_root
        env["TOGSIM_LEGOSIM_X"] = "0"
        env["TOGSIM_LEGOSIM_Y"] = "0"
        if use_ssd:
            env["TOGSIM_SSD_LEGOSIM"] = "1"
            env["TOGSIM_SSD_LEGOSIM_X"] = "1"
            env["TOGSIM_SSD_LEGOSIM_Y"] = "0"
        if use_dram:
            env["TOGSIM_DRAM_LEGOSIM"] = "1"
            env["TOGSIM_DRAM_PEER_LEGOSIM_X"] = "2"
            env["TOGSIM_DRAM_PEER_LEGOSIM_Y"] = "0"
            # Lets DramLegoSimLink convert the core-cycle delta interchiplet
            # resolves for each round trip back into ns -- must be the same
            # value used to set TOGSim's clock_rate in _build_legosim_yaml.
            if core_freq_mhz:
                env["TOGSIM_DRAM_LEGOSIM_CORE_FREQ_MHZ"] = str(core_freq_mhz)
        return env

    @staticmethod
    def _run_interchiplet(cmd_list, cwd, env, timeout_sec):
        """
        Runs `interchiplet` and waits for it, killing its whole process group
        on timeout: interchiplet forks TOGSim/ssd_simlet/dram_simlet as
        grandchildren, and a plain `Popen.kill()` on timeout would only kill
        interchiplet itself, leaving them orphaned and blocked on each other's
        pipes.

        interchiplet's own stdout/stderr (its top-level spdlog output --
        round/phase markers, and with use_dram_noc, the per-round
        "Difference related to previous round is X%." convergence line) is
        captured via the pipe rather than inherited, so on success it would
        otherwise be silently discarded (only surfaced via
        CalledProcessError.output/.stderr on failure). The caller is
        responsible for routing it into per-process logs -- see
        _split_interchiplet_log().
        """
        proc = subprocess.Popen(
            cmd_list, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd_list, output=stdout, stderr=stderr)
        return stdout, stderr

    @staticmethod
    def _split_interchiplet_log(run_dir, stdout, phase1_basenames, phase2_basenames):
        """
        Route interchiplet's own top-level stdout (round/phase markers,
        Load/Dump counts, Benchmark elapses, Difference related to previous
        round, process start/terminate lines, ...) into a per-process
        interchiplet.log under each proc_r<round>_p<phase>_t<thread>/
        directory -- the same directories bridge_thread already writes
        togsim.log/dram_simlet.log/etc. into -- instead of one shared file
        at the top of run_dir that mixes every process's start/stop events
        together and ends up reading like it "belongs" to whichever process
        is mentioned most.

        "Start simulation process PID. Command: <path>" / "Simulation
        process PID terminate with status = N." lines are routed to exactly
        the one process they're about. The thread index is looked up by
        matching <path>'s basename against phase1_basenames/phase2_basenames
        (the same ordered command lists _build_legosim_yaml wrote), NOT by
        the order these lines happen to appear in the log: bridge_thread
        runs each process's capture on its own pthread, so "Start
        simulation process" lines from concurrently-launched processes can
        print in either order even though pthread_create() itself was
        called in YAML order -- and processes don't necessarily *terminate*
        in start order either (e.g. a simlet finishing well before TOGSim
        does). Every other line inside a round/phase block (the round
        marker itself, Load/Dump counts, Benchmark elapses, Difference
        related to..., All process has exit, Round N elapses) applies to
        every process in that block and is duplicated into all of them, so
        each resulting log is a complete, self-contained account of that
        round from that process's point of view.
        """
        round_re = re.compile(r"Round (\d+) Phase (\d)")
        start_re = re.compile(r"Start simulation process (\d+)\. Command: (\S+)")
        terminate_re = re.compile(r"Simulation process (\d+) terminate with status")

        round_num, phase_num = None, None
        pid_to_thread = {}
        dest_lines = {}
        active_dests = []

        def basenames_for(phase):
            return phase1_basenames if phase == 1 else phase2_basenames

        for line in stdout.decode("utf-8", errors="replace").split("\n"):
            m = round_re.search(line)
            if m:
                round_num, phase_num = int(m.group(1)), int(m.group(2))
                pid_to_thread = {}
                active_dests = [
                    (round_num, phase_num, t) for t in range(len(basenames_for(phase_num)))
                ]
                for dest in active_dests:
                    dest_lines.setdefault(dest, []).append(line)
                continue

            if round_num is None:
                continue  # banner lines before the first Round marker

            sm = start_re.search(line)
            if sm:
                pid, cmd_path = sm.group(1), sm.group(2)
                cmd_name = os.path.basename(cmd_path)
                names = basenames_for(phase_num)
                thread_idx = names.index(cmd_name) if cmd_name in names else None
                if thread_idx is not None:
                    pid_to_thread[pid] = thread_idx
                    dest_lines.setdefault((round_num, phase_num, thread_idx), []).append(line)
                continue

            tm = terminate_re.search(line)
            if tm:
                t = pid_to_thread.get(tm.group(1))
                if t is not None:
                    dest_lines.setdefault((round_num, phase_num, t), []).append(line)
                continue

            for dest in active_dests:
                dest_lines[dest].append(line)

        for (r, p, t), lines in dest_lines.items():
            out_dir = Path(run_dir) / f"proc_r{r}_p{p}_t{t}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "interchiplet.log").write_bytes("\n".join(lines).encode("utf-8"))

    @staticmethod
    def _split_togsim_logs(run_dir):
        """
        interchiplet's bridge_thread writes TOGSim's raw, unfiltered stdout
        into every round's proc_r<round>_p1_t0/togsim.log (TOGSim is always
        phase1[0]/thread0) -- a mix of TOGSim's own log output and the
        interchiplet wire-protocol chatter described in _INTERCHIPLET_LINE_RE's
        comment. Split each round's file in place into two: togsim.log keeps
        only the interchiplet protocol lines (matching its name -- the
        interchiplet-facing log of that process), and TOGSim's own log lines
        move to a sibling pytorchsim.log in the same directory.
        """
        for togsim_log in sorted(Path(run_dir).glob("proc_r*_p1_t0/togsim.log")):
            lines = togsim_log.read_bytes().decode("utf-8", errors="replace").split("\n")
            interchiplet_lines, pytorchsim_lines = [], []
            for line in lines:
                (interchiplet_lines if TOGSimulator._INTERCHIPLET_LINE_RE.match(line)
                 else pytorchsim_lines).append(line)
            togsim_log.write_bytes("\n".join(interchiplet_lines).encode("utf-8"))
            (togsim_log.parent / "pytorchsim.log").write_bytes(
                "\n".join(pytorchsim_lines).encode("utf-8")
            )

    @staticmethod
    def run_standalone(
        model_path,
        attribute_path="",
        autotune_mode=False,
        config_path=None,
        togsim_path=None,
        timeout_sec=None,
    ):
        """
        Run a single kernel simulation in standalone mode.
        This method starts a new TOGSim process, runs the kernel, and waits for completion.
        For streaming multiple kernels, use launch_kernel() instead.

        Args:
            model_path: Path to TOG file (ONNX model)
            attribute_path: Path to attribute file
            autotune_mode: If True, run in autotune mode (silent)
            config_path: Path to TOGSim config file (required)
            togsim_path: Path to TOGSim directory (optional, defaults to CONFIG_TORCHSIM_DIR/TOGSim)
            timeout_sec: If set, terminate the Simulator subprocess after this many seconds
                (autotune uses this to skip very slow tile candidates).

        Returns:
            Path to the simulation result log file
        """
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        # Create result path with appropriate filename
        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
        else:
            base_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)

        base_dir.mkdir(parents=True, exist_ok=True)
        idx = TOGSimulator._next_result_index(base_dir)
        result_path = base_dir / f"{idx}.log"
        trace_file_path = base_dir / f"{idx}.trace"

        # Create trace file in result directory
        kernel_id, device_index, stream_index, timestamp = 0, 0, 0, 0
        command = f"LAUNCH_KERNEL,{kernel_id},{device_index},{stream_index},{model_path},{attribute_path},{timestamp}\n"
        with open(trace_file_path, 'w') as trace_file:
            trace_file.write(command)
            trace_file.flush()
            os.fsync(trace_file.fileno())

        use_legosim_ssd = extension_config.CONFIG_TOGSIM_LEGOSIM_SSD
        use_legosim_dram = extension_config.CONFIG_TOGSIM_LEGOSIM_DRAM
        # Only meaningful together with use_legosim_dram -- ignore the toggle
        # otherwise rather than requiring callers to keep the two in sync.
        use_dram_noc = use_legosim_dram and extension_config.CONFIG_TOGSIM_LEGOSIM_DRAM_NOC
        # Needed so DramLegoSimLink can convert interchiplet's resolved
        # core-cycle delta back to ns -- see _build_legosim_yaml's
        # clock_rate comment and DramLegoSimLink::query_latency_ns().
        core_freq_mhz = None
        if use_legosim_dram:
            with open(os.path.join(togsim_path, config_path), "r") as f:
                core_freq_mhz = yaml.safe_load(f)["core_freq_mhz"]

        try:
            if use_legosim_ssd or use_legosim_dram:
                togsim_bin = os.path.join(togsim_path, "build/bin/Simulator")
                run_dir = base_dir / f"{idx}.legosim_run"
                yaml_path = TOGSimulator._build_legosim_yaml(
                    togsim_bin, os.path.join(togsim_path, config_path), trace_file_path, run_dir,
                    log_level=extension_config.CONFIG_TOGSIM_DEBUG_LEVEL,
                    use_ssd=use_legosim_ssd, use_dram=use_legosim_dram, use_dram_noc=use_dram_noc,
                    core_freq_mhz=core_freq_mhz,
                )
                interchiplet_bin = os.path.join(
                    extension_config.CONFIG_LEGOSIM_ROOT, "interchiplet/bin/interchiplet"
                )
                # -t 1: run exactly one round -- TOGSim already ran its one kernel
                # to completion, and with the no-op phase2 filler there's nothing
                # to re-converge on a second round. With use_dram_noc, phase2 is a
                # real popnet instead, so we let it iterate CONFIG_LEGOSIM_DRAM_NOC_ROUNDS
                # times: dram_simlet/DramLegoSimLink track a running timeNow (see
                # their own comments) specifically so popnet's delayInfo.txt feeds
                # back into a real round 2+ instead of being computed once and
                # discarded. Note interchiplet's own convergence check (round-over-
                # round cycle difference) never fires here regardless -- it's driven
                # by explicit CYCLE sync commands neither dram_simlet nor
                # DramLegoSimLink issue (matching DDR.cpp/HBM.cpp, which don't
                # either) -- so all requested rounds always run; -t is a hard cap,
                # not a target. -w 3 (not 2) so chiplet address 2 (dram_simlet) is
                # always in range, needed for use_dram_noc's real popnet topology
                # and harmless otherwise (DIM_Y is always 0 for every chiplet this
                # integration uses, so width doesn't affect address resolution).
                rounds = extension_config.CONFIG_LEGOSIM_DRAM_NOC_ROUNDS if use_dram_noc else 1
                cmd = f"{interchiplet_bin} {yaml_path} -w 3 -f 2 -t {rounds}"

                if not autotune_mode:
                    logger.debug(f"[TOGSim] cmd> {cmd}")
                    path_desc = "+".join(
                        p for p, on in (
                            ("SSD", use_legosim_ssd), ("DRAM", use_legosim_dram), ("NoC", use_dram_noc),
                        ) if on
                    )
                    logger.info(f"[TOGSim] TOGSim simulation started (LegoSim {path_desc} path)")
                with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                    interchiplet_stdout, _ = TOGSimulator._run_interchiplet(
                        shlex.split(cmd), cwd=run_dir,
                        env=TOGSimulator._legosim_env(
                            use_ssd=use_legosim_ssd, use_dram=use_legosim_dram, core_freq_mhz=core_freq_mhz,
                        ),
                        timeout_sec=timeout_sec,
                    )
                # Route interchiplet's own top-level log into a per-process
                # interchiplet.log (see _split_interchiplet_log()) -- these
                # basenames and their order must match _build_legosim_yaml's
                # phase1/phase2 command list exactly.
                phase1_basenames = ["Simulator"]
                if use_legosim_ssd:
                    phase1_basenames.append("ssd_simlet")
                if use_legosim_dram:
                    phase1_basenames.append("dram_simlet")
                phase2_basenames = ["popnet"] if use_dram_noc else ["true"]
                TOGSimulator._split_interchiplet_log(
                    run_dir, interchiplet_stdout, phase1_basenames, phase2_basenames
                )
                # Split every round's raw togsim.log into pure interchiplet
                # protocol (kept as togsim.log) vs. TOGSim's own log (moved to
                # a sibling pytorchsim.log) -- see _split_togsim_logs().
                TOGSimulator._split_togsim_logs(run_dir)
                # TOGSim is phase1[0] of the YAML above -> phase 1, thread 0. With
                # use_dram_noc, DMA.cc's DramLegoSimLink path resolves latency from
                # the real interchiplet-elapsed cycle (see
                # DramLegoSimLink::query_latency_ns()), which popnet only feeds
                # real NoC delay into starting round 2 (round 1 has no prior
                # delayInfo.txt yet) -- so TOGSim's reported cycles are NOT
                # round-invariant here (confirmed: round 1 under-reports vs. the
                # converged round). Always read the last round, which is what the
                # checkpoint (written by every round, so left holding the last
                # round's counters once interchiplet exits) already reflects.
                pytorchsim_log = run_dir / f"proc_r{rounds}_p1_t0" / "pytorchsim.log"
                result = pytorchsim_log.read_bytes() if pytorchsim_log.exists() else b""
            else:
                cmd = f"{TOGSimulator.get_togsim_command(config_path, togsim_path)} --models_list {trace_file_path}"
                if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
                    cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

                if not autotune_mode:
                    logger.debug(f"[TOGSim] cmd> {cmd}")
                    logger.info("[TOGSim] TOGSim simulation started")
                with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                    completed = subprocess.run(
                        shlex.split(cmd),
                        capture_output=True,
                        check=True,
                        timeout=timeout_sec,
                    )
                    result = completed.stdout
        except subprocess.TimeoutExpired as e:
            logger.warning(
                "[TOGSim] Simulator subprocess exceeded timeout (%.1f s); terminating.",
                float(timeout_sec) if timeout_sec is not None else -1.0,
            )
            raise RuntimeError("TOGSim subprocess timeout") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"[TOGSim] Command failed with exit code {e.returncode}")
            logger.error(f"[TOGSim] Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
            assert 0

        # Prevent race condition
        with open(result_path, "w") as f:
            f.write(result.decode())
            f.flush()
            os.fsync(f.fileno())

        if not autotune_mode:
            import logging as _logging
            model_path_log = f' of "{model_path}" ' if logger.isEnabledFor(_logging.DEBUG) else " "
            logger.info(f'[TOGSim] Simulation log{model_path_log}is stored to "{result_path}"')
        return result_path

    @staticmethod
    def get_result_from_file(result_path):
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = 0.0
        simulation_time = float("inf")
        total_cycle = float("inf")

        # Read and find total stat position
        with open(result_path, "r") as f:
            lines = f.readlines()

        simulation_finished_idx = -1
        simulation_finished = False
        for idx, line in enumerate(lines):
            if TOGSimulator.FINISH_STR in line:
                simulation_finished = True
                simulation_finished_idx = idx
                break

        if simulation_finished_idx == -1:
            logger.warning(f"[TOGSim] Warning: Unable to parse the output file ({result_path}). The file may be improperly formatted.")
            return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time

        total_stat_lines = lines[simulation_finished_idx:]

        for line in total_stat_lines:
            # Parse core metrics (MatMul active cycle, Vector active cycle, etc.)
            if 'Core' in line:
                if 'MatMul active cycle' in line:
                    matmul_cycle = re.search(r'MatMul active cycle (\d+)', line).group(1)
                    vector_cycle = re.search(r'Vector active cycle (\d+)', line).group(1)
                    core_metrics['MatMul_active_cycle'] = int(matmul_cycle)
                    core_metrics['Vector_active_cycle'] = int(vector_cycle)
                elif 'Systolic Array Utilization' in line:
                    systolic_util = re.search(r'Systolic Array Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    vector_util = re.search(r'Vector Unit Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    total_cycle = re.search(r'Total cycle: (\d+)', line).group(1)
                    core_metrics['Systolic_Array_Utilization'] = float(systolic_util)
                    core_metrics['Vector_Unit_Utilization'] = float(vector_util)
                    core_metrics['Total_cycle'] = int(total_cycle)

            # Parse DRAM channel bandwidth utilization
            if 'DRAM CH' in line:
                channel = re.search(r'DRAM CH\[(\d+)\]', line).group(1)
                bw_util = re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1)
                dram_channel_bw[f'CH[{channel}]'] = float(bw_util)

            # Parse average DRAM bandwidth
            if 'DRAM: AVG BW Util' in line:
                avg_dram_bw = float(re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1))

            if 'Total execution cycles' in line:
                total_cycle = int(re.search(r'Total execution cycles: (\d+)', line).group(1))

            # Parse total simulation time
            if 'Wall-clock time for simulation' in line:
                simulation_time = float(re.search(r'Wall-clock time for simulation: (\d+\.?\d*) seconds', line).group(1))
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle

if __name__ == "__main__":
    # Example paths (adjust these to your actual test files)
    test_tog_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/tile_graph.onnx"
    test_attribute_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/runtime_0001/attribute/0"

    # Test: Launch multiple kernels
    sim = TOGSimulator(config_path="/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
    with sim:
        try:
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")

        try:
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")
    print(sim.trace_log)