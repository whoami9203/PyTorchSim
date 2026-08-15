#include "SsdTrace.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <ctime>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace {

std::string trim_copy(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (!token.empty())
      out.push_back(token);
  }
  return out;
}

}  // namespace

SsdTraceManager& SsdTraceManager::instance() {
  static SsdTraceManager instance;
  return instance;
}

SsdTraceManager::SsdTraceManager() {
  const char* trace_name = std::getenv("TOGSIM_SSD_TRACE_NAME");
  if (!trace_name || std::string(trace_name).empty()) {
    return;
  }
  _trace_name = trace_name;

  const char* base_dir = std::getenv("TOGSIM_SSD_TRACE_DIR");
  _trace_base_dir = base_dir && std::string(base_dir).size()
                        ? std::string(base_dir)
                        : std::string("/workspace/PyTorchSim/ssd_traces");

  _trace_path = _trace_base_dir + "/" + _trace_name + "/ssd_trace.csv";
  _weights_path = _trace_base_dir + "/" + _trace_name + "/weights.txt";
  _process_index = assign_process_index();
  std::ostringstream latency_path_builder;
  latency_path_builder << _trace_base_dir << "/" << _trace_name
                       << "/ssd_latency/dma_trace_" << _process_index << ".csv";
  _latency_path = latency_path_builder.str();
  _enabled = true;
  open_trace();
  load_weight_list();
  load_latency_file();

  spdlog::info("[SSD] trace enabled: {}", _trace_path);
  if (_latency_enabled) {
    spdlog::info("[SSD] latency replay enabled: {}", _latency_path);
  } else {
    spdlog::info("[SSD] latency replay disabled (trace-only)");
  }
}

void SsdTraceManager::open_trace() {
  namespace fs = std::filesystem;

  const char* continue_path_env = std::getenv("TOGSIM_SSD_TRACE_CONTINUE_PATH");
  if (continue_path_env && std::string(continue_path_env).size() && fs::exists(continue_path_env)) {
    // Continuing a previous batch's trace file (see the multi-kernel
    // batching path in Simulator/simulator.py's TOGSimulator, which restarts
    // this process once per DEVICE_SYNC-bounded batch) -- append instead of
    // starting a new numbered file, so ssd_traces/.../dma_trace_N.csv stays
    // one continuous file across batches instead of fragmenting into one
    // file per batch.
    _trace_path = continue_path_env;
    _trace_file.open(_trace_path, std::ios::out | std::ios::app);
    if (!_trace_file.is_open()) {
      spdlog::error("[SSD] Failed to append to continued trace file: {}", _trace_path);
      _enabled = false;
      return;
    }
    // No header write -- the file already has one from when open_trace()
    // first created it for the earliest batch in this chain.
    return;
  }

  const fs::path counter_path = fs::path(_trace_base_dir) / _trace_name / "dma_trace_counter.txt";
  uint64_t trace_counter = 0;
  if (fs::exists(counter_path)) {
    std::ifstream in(counter_path);
    if (in.is_open()) {
      in >> trace_counter;
    }
  }
  ++trace_counter;
  std::ostringstream name_builder;
  name_builder << "dma_trace_" << trace_counter << ".csv";
  _trace_path = _trace_base_dir + "/" + _trace_name + "/" + name_builder.str();
  fs::path out_path(_trace_path);
  fs::create_directories(out_path.parent_path());
  {
    std::ofstream out(counter_path, std::ios::out | std::ios::trunc);
    if (out.is_open()) {
      out << trace_counter << "\n";
      out.flush();
    }
  }
  _trace_file.open(_trace_path, std::ios::out | std::ios::trunc);
  if (!_trace_file.is_open()) {
    spdlog::error("[SSD] Failed to open trace file: {}", _trace_path);
    _enabled = false;
    return;
  }
  _trace_file << "seq,op,addr,len,timestamp_ns,core_id,inst_id,addr_name\n";
  _trace_file.flush();
}

void SsdTraceManager::load_weight_list() {
  namespace fs = std::filesystem;
  if (fs::exists(_weights_path)) {
    std::ifstream in(_weights_path);
    std::string line;
    while (std::getline(in, line)) {
      line = trim_copy(line);
      if (line.empty() || line[0] == '#')
        continue;
      _weight_entries.push_back(line);
    }
  }

  const char* substr_env = std::getenv("TOGSIM_SSD_WEIGHT_SUBSTR");
  if (substr_env && std::string(substr_env).size()) {
    _weight_substrings = split_csv(substr_env);
  }

  if (_weight_entries.empty() && _weight_substrings.empty()) {
    spdlog::warn("[SSD] No weight filters found (weights.txt or TOGSIM_SSD_WEIGHT_SUBSTR). SSD tracing will be empty.");
  }
}

uint64_t SsdTraceManager::assign_process_index() {
  namespace fs = std::filesystem;
  // Lives under the results dir (TORCHSIM_LOG_PATH, i.e. togsim_results/<trace>/),
  // not ssd_traces/ -- this counts TOGSim process launches for _latency_path
  // numbering, unrelated to the SSD trace *data* under ssd_traces/ (whose own
  // dma_trace_counter.txt in open_trace() rightly stays there). Keeping it next
  // to the results that already get cleaned/reset together avoids it drifting
  // out of sync across ad hoc reruns that clear togsim_results/ but not
  // ssd_traces/ (or vice versa).
  const char* result_dir_env = std::getenv("TORCHSIM_LOG_PATH");
  const fs::path result_dir = result_dir_env && std::string(result_dir_env).size()
                                   ? fs::path(result_dir_env)
                                   : fs::path("/workspace/PyTorchSim/togsim_results") / _trace_name;
  const fs::path counter_path = result_dir / "togsim_process_counter.txt";
  uint64_t counter = 0;
  if (fs::exists(counter_path)) {
    std::ifstream in(counter_path);
    if (in.is_open()) {
      in >> counter;
    }
  }
  ++counter;
  fs::create_directories(counter_path.parent_path());
  std::ofstream out(counter_path, std::ios::out | std::ios::trunc);
  if (out.is_open()) {
    out << counter << "\n";
    out.flush();
  }
  return counter;
}

std::string SsdTraceManager::make_latency_key(uint32_t core_id, uint64_t inst_id, const std::string& addr_name) const {
  return std::to_string(core_id) + "|" + std::to_string(inst_id) + "|" + addr_name;
}

void SsdTraceManager::load_latency_file() {
  namespace fs = std::filesystem;
  if (!fs::exists(_latency_path)) {
    _latency_enabled = false;
    return;
  }
  std::ifstream in(_latency_path);
  if (!in.is_open()) {
    _latency_enabled = false;
    return;
  }

  std::string line;
  bool first_line = true;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    if (first_line) {
      first_line = false;
      if (line.find("seq,") == 0)
        continue;
    }
    const auto tokens = split_csv(line);
    if (tokens.size() < 9)
      continue;
    const uint32_t core_id = static_cast<uint32_t>(std::stoul(tokens[5]));
    const uint64_t inst_id = static_cast<uint64_t>(std::stoull(tokens[6]));
    const std::string& addr_name = tokens[7];
    const uint64_t latency_ns = static_cast<uint64_t>(std::stoull(tokens[8]));
    const std::string key = make_latency_key(core_id, inst_id, addr_name);
    _latency_map[key].push_back(latency_ns);
  }
  _latency_enabled = !_latency_map.empty();
}

bool SsdTraceManager::match_weight(const std::string& name) const {
  if (name.empty())
    return false;
  if (_weight_entries.empty() && _weight_substrings.empty()) {
    if (_runtime_inputs.empty())
      return false;
    return _runtime_inputs.find(name) == _runtime_inputs.end();
  }
  for (const auto& entry : _weight_entries) {
    if (entry == "*")
      return true;
    if (!entry.empty() && entry.back() == '*') {
      const std::string prefix = entry.substr(0, entry.size() - 1);
      if (name.rfind(prefix, 0) == 0)
        return true;
    } else if (entry == name) {
      return true;
    }
  }
  for (const auto& sub : _weight_substrings) {
    if (name.find(sub) != std::string::npos)
      return true;
  }
  return false;
}

bool SsdTraceManager::is_weight_name(const std::string& name) const {
  return match_weight(name);
}

void SsdTraceManager::set_runtime_inputs(const std::vector<std::string>& inputs) {
  _runtime_inputs.clear();
  for (const auto& name : inputs) {
    if (!name.empty())
      _runtime_inputs.insert(name);
  }
}


void SsdTraceManager::maybe_trace_and_mark(uint32_t core_id,
                                           cycle_type core_cycle,
                                           uint32_t core_freq_mhz,
                                           Instruction& inst,
                                           mem_fetch* access) {
  if (!_enabled)
    return;
  if (!inst.is_dma_read() && !inst.is_dma_write())
    return;

  const double period_ns = core_freq_mhz > 0 ? 1000.0 / static_cast<double>(core_freq_mhz) : 0.0;
  const uint64_t timestamp_ns = static_cast<uint64_t>(std::llround(static_cast<double>(core_cycle) * period_ns));
  const uint64_t inst_id = inst.get_global_inst_id();

  if (_inst_traced.find(inst_id) == _inst_traced.end()) {
    const uint64_t total_bits = static_cast<uint64_t>(inst.get_tile_numel()) *
                                static_cast<uint64_t>(inst.get_elem_bits());
    const uint64_t total_bytes = (total_bits + 7) >> 3;
    const char op = inst.is_dma_write() ? 'W' : 'R';
    const uint64_t seq = _trace_seq++;

    _trace_file << seq
                << "," << op << "," << static_cast<uint64_t>(inst.get_base_dram_address())
                << "," << total_bytes
                << "," << timestamp_ns
                << "," << core_id
                << "," << inst_id
                << "," << inst.get_addr_name()
                << "\n";
    _trace_file.flush();
    _inst_traced.insert(inst_id);
  }
}

bool SsdTraceManager::pop_latency_for_instruction(uint32_t core_id,
                                                  const Instruction& inst,
                                                  uint64_t* latency_ns) {
  if (!_latency_enabled || latency_ns == nullptr)
    return false;
  const std::string key = make_latency_key(core_id, inst.get_global_inst_id(), inst.get_addr_name());
  auto it = _latency_map.find(key);
  if (it == _latency_map.end() || it->second.empty())
    return false;
  *latency_ns = it->second.front();
  it->second.pop_front();
  if (it->second.empty())
    _latency_map.erase(it);
  return true;
}
