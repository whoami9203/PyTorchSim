#pragma once

#include <cstdint>
#include <string>

#include "pipe_comm.h"
#include "ssd_protocol.h"

// Live counterpart to SsdTraceManager's replay path (see SsdTrace.h):
// instead of popping a pre-recorded latency out of a trace file, this asks
// an external LegoSim SSD/DRAM simlet for a latency, over the interchiplet
// sync protocol, and blocks until it answers. Only makes sense when TOGSim
// itself is running as one of interchiplet's phase1 processes (see
// artifact/auto_transformer/CMakeLists.txt's `run` target for the general
// pattern) -- outside of that, there's nothing on the other end of stdin.
//
// Enabled by setting TOGSIM_SSD_LEGOSIM=1. Chiplet coordinates default to
// the (0,0)=compute / (1,0)=DRAM convention used elsewhere in this
// integration, overridable via TOGSIM_LEGOSIM_X/Y and
// TOGSIM_SSD_LEGOSIM_X/Y.
//
// Every query is issued at TOGSim's real, caller-supplied core_cycle, and
// (like DramLegoSimLink) reads back interchiplet's resolved end cycle for
// the round trip rather than assuming zero transport cost -- see
// DramLegoSimLink's class comment for the full mechanism (cycle-domain
// reconciliation via each process's clock_rate). Unlike DramLegoSimLink's
// NoC participation, which is opt-in (TOGSIM_LEGOSIM_DRAM_NOC), a real
// phase-2 NoC simlet always runs whenever SsdLegoSimLink is enabled --
// see _build_legosim_yaml's use_ssd handling in Simulator/simulator.py.
class SsdLegoSimLink {
 public:
  static SsdLegoSimLink& instance();

  bool enabled() const { return _enabled; }

  // Blocks until the SSD simlet answers, and returns the real elapsed
  // interchiplet latency (in ns) for the round trip issued at
  // `core_cycle` -- not just ssd_simlet's own modeled latency, see class
  // comment. `addr`/`nbytes` describe the DMA access; `inst_id`/`addr_name`
  // are only for identification/logging on the simlet side (addr_name is
  // truncated to kSsdAddrNameCapacity-1 bytes).
  uint64_t query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                            const std::string& addr_name, uint64_t core_cycle);

  // Tells the SSD simlet to exit its request loop and waits for its ack.
  // Call once, right before TOGSim's process would otherwise exit. Safe to
  // call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  SsdLegoSimLink();

  struct RoundTrip {
    SsdLatencyResponse response;
    // End cycle interchiplet resolved for this round trip, already
    // converted back into TOGSim's own core-cycle domain.
    InterChiplet::TimeType resolved_cycle;
  };
  RoundTrip round_trip(const SsdLatencyRequest& req, InterChiplet::TimeType cycle);

  long _self_x = 0;
  long _self_y = 0;
  long _peer_x = 1;
  long _peer_y = 0;
  bool _enabled = false;
  bool _shutdown_sent = false;
  InterChiplet::PipeComm _pipe_comm;
  // TOGSim's core clock, needed to convert the resolved core-cycle delta
  // back into ns for query_latency_ns()'s return contract. 0 (unset) falls
  // back to resp.latency_ns -- see .cc.
  double _core_freq_mhz = 0.0;
};
