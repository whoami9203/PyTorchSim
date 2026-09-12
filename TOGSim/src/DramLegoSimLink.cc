#include "DramLegoSimLink.h"

#include <cmath>
#include <cstdlib>

#include <spdlog/spdlog.h>

namespace {
long env_long(const char* name, long fallback) {
  const char* v = std::getenv(name);
  return v ? std::atol(v) : fallback;
}
double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : fallback;
}
}  // namespace

DramLegoSimLink& DramLegoSimLink::instance() {
  static DramLegoSimLink link;
  return link;
}

DramLegoSimLink::DramLegoSimLink() {
  const char* on = std::getenv("TOGSIM_DRAM_LEGOSIM");
  _enabled = on && std::string(on) == "1";
  if (!_enabled) return;

  _self_x = env_long("TOGSIM_LEGOSIM_X", 0);
  _self_y = env_long("TOGSIM_LEGOSIM_Y", 0);
  _peer_x = env_long("TOGSIM_DRAM_PEER_LEGOSIM_X", 2);
  _peer_y = env_long("TOGSIM_DRAM_PEER_LEGOSIM_Y", 0);
  _core_freq_mhz = env_double("TOGSIM_DRAM_LEGOSIM_CORE_FREQ_MHZ", 0.0);

  spdlog::info("[DramLegoSimLink] enabled: self=({},{}) dram_peer=({},{}) core_freq_mhz={}",
               _self_x, _self_y, _peer_x, _peer_y, _core_freq_mhz);
}

DramLegoSimLink::RoundTrip DramLegoSimLink::round_trip(const SsdLatencyRequest& req,
                                                        InterChiplet::TimeType cycle) {
  // Request leg: self -> peer, issued at the caller's real core_cycle (not
  // a placeholder) so interchiplet's read/write pairing resolves against
  // TOGSim's actual timeline -- required for a real phase-2 NoC simlet's
  // delay to land on this transfer instead of being computed against a
  // fixed baseline every time.
  std::string req_file = InterChiplet::sendSync(_self_x, _self_y, _peer_x, _peer_y);
  _pipe_comm.write_data(req_file.c_str(), const_cast<SsdLatencyRequest*>(&req), sizeof(req));
  InterChiplet::writeSync(cycle, _self_x, _self_y, _peer_x, _peer_y, sizeof(req), 0);

  // Response leg: peer -> self. dram_simlet already folded its own modeled
  // DRAM latency into the cycle it wrote the response at, so the resolved
  // cycle interchiplet hands back here already reflects DRAM latency plus
  // whatever transport delay it resolved for both legs of the round trip.
  std::string resp_file = InterChiplet::receiveSync(_peer_x, _peer_y, _self_x, _self_y);
  SsdLatencyResponse resp{};
  _pipe_comm.read_data(resp_file.c_str(), &resp, sizeof(resp));
  InterChiplet::TimeType resolved_cycle =
      InterChiplet::readSync(cycle, _peer_x, _peer_y, _self_x, _self_y, sizeof(resp), 0);
  return {resp, resolved_cycle};
}

uint64_t DramLegoSimLink::query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                                           const std::string& addr_name, uint64_t core_cycle) {
  SsdLatencyRequest req{};
  req.addr = addr;
  req.nbytes = nbytes;
  req.inst_id = inst_id;
  req.set_addr_name(addr_name);
  req.kind = kSsdReqRead;
  req.terminate = 0;
  RoundTrip result = round_trip(req, static_cast<InterChiplet::TimeType>(core_cycle));

  if (_core_freq_mhz <= 0.0) {
    // TOGSIM_DRAM_LEGOSIM_CORE_FREQ_MHZ wasn't set (shouldn't happen via
    // the normal simulator.py path, but guard anyway) -- fall back to
    // dram_simlet's own modeled latency alone, ignoring transport delay,
    // rather than dividing by zero.
    return result.response.latency_ns;
  }
  // resolved_cycle is already back in TOGSim's own core-cycle domain
  // (interchiplet converts via this process's own clock_rate before
  // acking) -- convert the elapsed cycles to ns for this method's
  // contract; DMA.cc converts back to cycles itself.
  InterChiplet::TimeType elapsed_cycles =
      result.resolved_cycle > core_cycle ? result.resolved_cycle - core_cycle : 1;
  return static_cast<uint64_t>(std::ceil(elapsed_cycles * (1000.0 / _core_freq_mhz)));
}

void DramLegoSimLink::shutdown() {
  if (!_enabled || _shutdown_sent) return;
  _shutdown_sent = true;
  SsdLatencyRequest req{};
  req.kind = kSsdReqTerminate;
  req.terminate = 1;
  round_trip(req, 0);
  spdlog::info("[DramLegoSimLink] sent terminate sentinel, DRAM simlet acked.");
}
