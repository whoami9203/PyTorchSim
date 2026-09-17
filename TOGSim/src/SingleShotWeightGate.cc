#include "SingleShotWeightGate.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace {
// Tensor names look like "model.layers.3.input_layernorm.weight" (Llama) or
// "gpt_neox.layers.3.input_layernorm.weight" (GPT-NeoX) -- both contain
// "layers.<N>." somewhere in the middle. Returns -1 if no such segment is
// found (e.g. a persistent, non-per-layer tensor that somehow ended up in
// this file).
int parse_layer_index(const std::string& name) {
  static const std::regex re("layers\\.(\\d+)\\.");
  std::smatch m;
  if (std::regex_search(name, m, re)) {
    return std::stoi(m[1].str());
  }
  return -1;
}
}  // namespace

SingleShotWeightGate& SingleShotWeightGate::instance() {
  static SingleShotWeightGate inst;
  return inst;
}

SingleShotWeightGate::SingleShotWeightGate() {
  const char* dir = std::getenv("TOGSIM_SSD_TRACE_DIR");
  const char* name = std::getenv("TOGSIM_SSD_TRACE_NAME");
  if (!dir || !name)
    return;

  const std::string path = std::string(dir) + "/" + name + "/model_weight_ranges_1d.txt";
  std::ifstream f(path);
  if (!f.is_open())
    return;

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::string tname, base_s, end_s, size_s;
    if (!std::getline(iss, tname, '\t'))
      continue;
    if (!std::getline(iss, base_s, '\t'))
      continue;
    if (!std::getline(iss, end_s, '\t'))
      continue;
    if (!std::getline(iss, size_s, '\t'))
      continue;

    Tracked t;
    t.base = std::stoull(base_s);
    t.end = std::stoull(end_s);
    t.size_bytes = std::stoull(size_s);
    const int layer_idx = parse_layer_index(tname);
    t.max_loads = (layer_idx >= 0) ? (layer_idx + 1) : -1;
    _tensors.push_back(t);
  }

  std::sort(_tensors.begin(), _tensors.end(),
            [](const Tracked& a, const Tracked& b) { return a.base < b.base; });
  _enabled = !_tensors.empty();
}

bool SingleShotWeightGate::classify(uint64_t addr, bool* should_charge, uint64_t* tensor_bytes) {
  if (!_enabled)
    return false;

  // Last tracked tensor with base <= addr.
  auto it = std::upper_bound(
      _tensors.begin(), _tensors.end(), addr,
      [](uint64_t value, const Tracked& t) { return value < t.base; });
  if (it == _tensors.begin())
    return false;
  --it;
  const size_t idx = static_cast<size_t>(it - _tensors.begin());
  Tracked& t = _tensors[idx];
  if (addr < t.base || addr >= t.end)
    return false;

  if (addr == t.base) {
    // Start of a fresh load. Bill it unless we've already seen as many
    // loads as this tensor's layer index says are legitimate (guards
    // against an unrelated tensor coincidentally reusing this freed
    // address after the real loads are done).
    const bool under_cap = (t.max_loads < 0) || (t.loads_seen < t.max_loads);
    if (under_cap)
      t.loads_seen++;
    *should_charge = under_cap;
  } else {
    // A later fragment of the same load that's already been billed (or
    // already exhausted its cap) -- never charge again until the next
    // base-address touch starts a new load.
    *should_charge = false;
  }
  *tensor_bytes = t.size_bytes;
  return true;
}
