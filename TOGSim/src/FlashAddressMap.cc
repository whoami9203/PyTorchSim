#include "FlashAddressMap.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

FlashAddressMap& FlashAddressMap::instance() {
  static FlashAddressMap map;
  return map;
}

FlashAddressMap::FlashAddressMap() {
  const char* trace_name = std::getenv("TOGSIM_SSD_TRACE_NAME");
  if (!trace_name || std::string(trace_name).empty()) {
    return;
  }
  const char* trace_dir_env = std::getenv("TOGSIM_SSD_TRACE_DIR");
  std::string trace_dir = (trace_dir_env && std::string(trace_dir_env).size())
                              ? std::string(trace_dir_env)
                              : std::string("/workspace/PyTorchSim/ssd_traces");
  std::string path = trace_dir + "/" + trace_name + "/ssd_offsets.tsv";

  std::ifstream in(path);
  if (!in.is_open()) {
    spdlog::warn(
        "[FlashAddressMap] Could not open {} -- weight DMAs will be placed in flash by their "
        "DRAM address instead of build_layer_ssd_offset_map.py's layout.",
        path);
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string name;
    Entry e{};
    if (!std::getline(iss, name, '\t')) continue;
    if (!(iss >> e.dram_base >> e.dram_end >> e.ssd_offset >> e.ssd_length)) continue;
    if (e.dram_end < e.dram_base) continue;
    _entries.push_back(e);
  }
  std::sort(_entries.begin(), _entries.end(),
            [](const Entry& a, const Entry& b) { return a.dram_base < b.dram_base; });
  _enabled = !_entries.empty();
  spdlog::info("[FlashAddressMap] Loaded {} tensor placement(s) from {}", _entries.size(), path);
}

bool FlashAddressMap::translate(uint64_t addr, uint64_t* flash_offset) const {
  if (!_entries.empty()) {
    // First entry whose base is > addr; the only candidate containing addr is
    // the one right before it (entries are sorted and don't overlap -- they
    // come from one layer's distinct tensors).
    auto it = std::upper_bound(_entries.begin(), _entries.end(), addr,
                               [](uint64_t value, const Entry& e) { return value < e.dram_base; });
    if (it != _entries.begin()) {
      --it;
      if (addr < it->dram_end) {
        *flash_offset = it->ssd_offset + (addr - it->dram_base);
        return true;
      }
    }
  }
  *flash_offset = addr;
  return false;
}
