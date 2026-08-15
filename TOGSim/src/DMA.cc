#include "DMA.h"
#include "DramLegoSimLink.h"
#include "SsdLegoSimLink.h"
#include "SsdTrace.h"
#include "TileGraph.h"
#include "TraceLogTags.h"
#include "WeightAddressRanges.h"

#include <cmath>

DMA::DMA(uint32_t id, uint32_t dram_req_size, bool l2_datacache_enabled, uint32_t core_freq_mhz) {
  _id = id;
  _dram_req_size = dram_req_size;
  _l2_datacache_enabled = l2_datacache_enabled;
  _core_freq_mhz = core_freq_mhz;
  _current_inst = nullptr;
  _finished = true;
}

void DMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  _ssd_pending = false;
  _ssd_finish_cycle = 0;
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  if (tile_size.size() <= 0 || tile_size.size() > get_max_dim()) {
    spdlog::error("[DMA {}] issued tile is not supported format.. tile.size: {}, tile_size: [{}]", _id, tile_size.size(), fmt::join(tile_size, ", "));
    exit(EXIT_FAILURE);
  }
  _finished = false;
}

void DMA::update_ssd(cycle_type core_cycle) {
  if (_ssd_pending && core_cycle >= _ssd_finish_cycle) {
    _ssd_pending = false;
    _finished = true;
    _generated_once = false;
    if (_current_inst != nullptr) {
      _ssd_finished_inst = std::move(_current_inst);
      _current_inst = nullptr;
    }
  }
}

std::shared_ptr<Instruction> DMA::take_ssd_finished() {
  if (_ssd_finished_inst == nullptr)
    return nullptr;
  return std::move(_ssd_finished_inst);
}

std::shared_ptr<std::vector<mem_fetch*>> DMA::get_memory_access(cycle_type core_cycle, int nr_req) {
  auto access_vec = std::make_shared<std::vector<mem_fetch *>>();

  if (_ssd_pending)
    return access_vec;

  if (!_generated_once) {
    uint64_t latency_ns = 0;
    bool have_latency = false;
    if (_current_inst->is_dma_read()) {
      if (SsdLegoSimLink::instance().enabled()) {
        // Live path: only DMAs landing inside a currently-loaded weight
        // tensor's address range are routed to the SSD simlet -- activations,
        // KV cache, etc. keep using the normal DRAM timing model below.
        //
        // Classifying by name (SsdTraceManager::is_weight_name()'s "not a
        // registered runtime input" fallback) was tried and reverted:
        // TileGraphParser.cc registers *every* top-level kernel argument
        // (weights included) as a "runtime input" via address_info, so that
        // heuristic excluded real weight DMAs too. Classifying by address
        // instead works because PyTorchSimDevice's tensors are backed by
        // real host memory, so a weight's data_ptr() on the Python side is
        // the exact same address reported here. See WeightAddressRanges.h.
        const uint64_t base_addr = static_cast<uint64_t>(_current_inst->get_base_dram_address());
        if (WeightAddressRanges::instance().is_weight_address(base_addr)) {
          const uint64_t total_bits = static_cast<uint64_t>(_current_inst->get_tile_numel()) *
                                      static_cast<uint64_t>(_current_inst->get_elem_bits());
          const uint64_t total_bytes = (total_bits + 7) >> 3;
          latency_ns = SsdLegoSimLink::instance().query_latency_ns(
              base_addr, total_bytes, _current_inst->get_global_inst_id(),
              _current_inst->get_addr_name(), static_cast<uint64_t>(core_cycle));
          have_latency = true;
        }
      } else {
        have_latency =
            SsdTraceManager::instance().pop_latency_for_instruction(_id, *_current_inst, &latency_ns);
      }
    }
    if (!have_latency && DramLegoSimLink::instance().enabled()) {
      // Catch-all: every DMA (read or write) not already claimed by the SSD
      // path above is routed to the DRAM-legosim simlet instead of the real
      // Dram/Interconnect timing model -- this is what lets DRAM-legosim
      // mode fully replace Dram/Interconnect rather than just standing in
      // for weight reads the way the SSD path does.
      const uint64_t base_addr = static_cast<uint64_t>(_current_inst->get_base_dram_address());
      const uint64_t total_bits = static_cast<uint64_t>(_current_inst->get_tile_numel()) *
                                  static_cast<uint64_t>(_current_inst->get_elem_bits());
      const uint64_t total_bytes = (total_bits + 7) >> 3;
      latency_ns = DramLegoSimLink::instance().query_latency_ns(
          base_addr, total_bytes, _current_inst->get_global_inst_id(),
          _current_inst->get_addr_name(), static_cast<uint64_t>(core_cycle));
      have_latency = true;
    }
    if (have_latency) {
      const double period_ns = _core_freq_mhz > 0 ? 1000.0 / static_cast<double>(_core_freq_mhz) : 0.0;
      uint64_t latency_cycles = 0;
      if (period_ns > 0.0) {
        latency_cycles = static_cast<uint64_t>(std::ceil(static_cast<double>(latency_ns) / period_ns));
      }
      if (latency_cycles == 0)
        latency_cycles = 1;
      _ssd_pending = true;
      _ssd_finish_cycle = core_cycle + latency_cycles;
      _finished = false;
      return access_vec;
    }
    std::shared_ptr<std::set<addr_type>> addr_set =
      _current_inst->get_dram_address(_dram_req_size);

    Tile* owner = (Tile*)_current_inst->get_owner();
    std::shared_ptr<TileSubGraph> owner_subgraph = owner->get_owner();
    unsigned long long base_daddr = _current_inst->get_base_dram_address();

    bool is_cacheable =
      owner_subgraph->is_cacheable(base_daddr, base_daddr + _dram_req_size);

    if (_l2_datacache_enabled) {
      spdlog::trace(
          "[{}][Core {}][{}][INST_ID={}] dram=0x{:016x} cacheable={}",
          core_cycle,
          _id,
          TraceLogTag::pad15(TraceLogTag::kL2CacheableStatusForAddress),
          _current_inst->get_global_inst_id(),
          base_daddr,
          is_cacheable);
    }
    spdlog::trace(
        "[{}][Core {}][{}][INST_ID={}] core_id={} subgraph_id={} numa_id={} addr_name={} is_write={}",
        core_cycle,
        _id,
        TraceLogTag::pad15(TraceLogTag::kDmaNumaPlacement),
        _current_inst->get_global_inst_id(),
        owner_subgraph->get_core_id(),
        _current_inst->subgraph_id,
        _current_inst->get_numa_id(),
        _current_inst->get_addr_name(),
        _current_inst->is_dma_write());
    for (const auto& addr : *addr_set) {
      mem_access_type acc_type =
        _current_inst->is_dma_write() ? mem_access_type::GLOBAL_ACC_W
                                          : mem_access_type::GLOBAL_ACC_R;
      mf_type type =
        _current_inst->is_dma_write() ? mf_type::WRITE_REQUEST
                                          : mf_type::READ_REQUEST;

      mem_fetch* access = new mem_fetch(
          addr, acc_type, type, _dram_req_size,
          _current_inst->get_numa_id(),
          static_cast<void*>(_current_inst.get()));

      access->set_cacheable(is_cacheable);
      SsdTraceManager::instance().maybe_trace_and_mark(
          _id, core_cycle, _core_freq_mhz, *_current_inst, access);
      _current_inst->inc_waiting_request();
      _pending_accesses.push(access);
    }
    _generated_once = true;
  }

  if (nr_req == -1)
    nr_req = _pending_accesses.size();

  // Return pending accesses up to nr_req
  for (int i = 0; i < nr_req; i++) {
      if (_pending_accesses.empty())
        break;
      access_vec->push_back(_pending_accesses.front());
      _pending_accesses.pop();
  }

  if (_pending_accesses.empty()) {
    _finished = true;
    _generated_once = false;
  }

  return access_vec;
}

uint32_t DMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}