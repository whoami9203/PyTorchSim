#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pipe_comm.h"
#include "ssd_protocol.h"

// The NPU's integrated flash controller.
//
// Live counterpart to SsdTraceManager's replay path (see SsdTrace.h): instead
// of popping a pre-recorded latency out of a trace file, this drives a set of
// external LegoSim NAND flash chiplets over the interchiplet sync protocol and
// blocks until they answer. Only makes sense when TOGSim itself is running as
// one of interchiplet's phase1 processes -- outside of that, there's nothing
// on the other end of stdin.
//
// The architecture being modeled is an NPU with the flash controller on-die,
// talking raw NAND over D2D links to flash chiplets. So everything an SSD
// controller owns stays on this side:
//   - tensor -> flash-address translation, via FlashAddressMap (the flat part
//     of an FTL's job; see FlashAddressMap.h);
//   - channel interleaving: consecutive `stripe_bytes`-sized chunks of the
//     flash address space are assigned round-robin to channels, so one DMA is
//     split across all the channels it touches and issued to them at once --
//     which is exactly where the flash device's read bandwidth comes from
//     (one channel alone is only a few hundred MB/s).
// And the chiplets are raw NAND packages: one process per channel, each
// running SimpleSSD's PAL layer for its own channel only (see
// SimpleSSD-Standalone/sim/legosim_flash_chiplet_main.cc).
//
// Enabled by setting TOGSIM_SSD_LEGOSIM=1. Chiplet coordinates: TOGSim is at
// (TOGSIM_LEGOSIM_X, TOGSIM_LEGOSIM_Y), default (0,0); flash channel i is at
// (TOGSIM_SSD_LEGOSIM_X + i, TOGSIM_SSD_LEGOSIM_Y), default (1+i, 0), for i in
// [0, TOGSIM_SSD_LEGOSIM_NUM_CHANNELS). Simulator/simulator.py's
// _legosim_env()/_build_legosim_yaml() set all of these consistently with the
// phase1 process list and the generated popnet topology.
//
// Every query is issued at TOGSim's real, caller-supplied core_cycle, and
// reads back interchiplet's resolved end cycle for the round trip rather than
// assuming zero transport cost -- see DramLegoSimLink's class comment for the
// full mechanism (cycle-domain reconciliation via each process's clock_rate).
// Unlike DramLegoSimLink's NoC participation, which is opt-in, a real phase-2
// NoC simlet always runs whenever SsdLegoSimLink is enabled -- see
// _build_legosim_yaml's use_ssd handling in Simulator/simulator.py.
class SsdLegoSimLink {
 public:
  static SsdLegoSimLink& instance();

  bool enabled() const { return _enabled; }
  size_t num_channels() const { return _channels.size(); }

  // Blocks until every flash channel this access touches has answered, and
  // returns the real elapsed interchiplet latency (in ns) for the round trip
  // issued at `core_cycle` -- the slowest channel's, since the DMA isn't
  // complete until its last byte arrives. Not just the chiplets' own modeled
  // NAND latency: see class comment. `addr`/`nbytes` describe the DMA;
  // `inst_id`/`addr_name` are only for identification/logging on the chiplet
  // side (addr_name is truncated to kSsdAddrNameCapacity-1 bytes).
  uint64_t query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                            const std::string& addr_name, uint64_t core_cycle);

  // Tells every flash chiplet to exit its request loop and waits for their
  // acks. Call once, right before TOGSim's process would otherwise exit. Safe
  // to call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  SsdLegoSimLink();

  struct Channel {
    long x = 0;
    long y = 0;
    // Bytes this channel serves for the request currently being issued --
    // scratch space reused across queries so the hot path allocates nothing.
    uint64_t pending_offset = 0;
    uint64_t pending_bytes = 0;
  };

  // One request/response exchange with a single channel, as two separate
  // halves so a query can issue to every channel it touches before waiting on
  // any of them (that concurrency is the whole point of having channels).
  void issue(Channel& channel, const SsdLatencyRequest& req, InterChiplet::TimeType cycle);
  InterChiplet::TimeType collect(Channel& channel, SsdLatencyResponse* resp,
                                 InterChiplet::TimeType cycle);

  // Ask channel 0 for the flash geometry and adopt its stripe size, so the
  // interleaving granularity is never configured twice (once in the SimpleSSD
  // .cfg, once here) and left to drift. Runs once, on the first query.
  void handshake();

  // Fill in each channel's pending_offset/pending_bytes for a read of
  // [flash_offset, flash_offset + nbytes), and return how many channels the
  // access actually touches (those are the ones with pending_bytes > 0).
  size_t plan(uint64_t flash_offset, uint64_t nbytes);

  long _self_x = 0;
  long _self_y = 0;
  std::vector<Channel> _channels;
  bool _enabled = false;
  bool _shutdown_sent = false;
  bool _handshake_done = false;
  // Interleaving granularity: one PAL superpage on one channel. Learned from
  // the chiplets (see handshake()); TOGSIM_SSD_LEGOSIM_STRIPE_BYTES overrides
  // that, and is also the value used if the handshake doesn't answer.
  uint64_t _stripe_bytes = 4096;
  InterChiplet::PipeComm _pipe_comm;
  // TOGSim's core clock, needed to convert the resolved core-cycle delta back
  // into ns for query_latency_ns()'s return contract. 0 (unset) falls back to
  // the chiplets' own modeled latency -- see .cc.
  double _core_freq_mhz = 0.0;
};
