#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Wire format shared between TOGSim's SsdLegoSimLink (DMA.cc caller side) and
// whichever LegoSim chiplet answers it:
//   - TOGSim/legosim/ssd_simlet.cpp          (the "formula" backend)
//   - SimpleSSD-Standalone/sim/legosim_flash_chiplet_main.cc
//                                            (the "simplessd" backend: one
//                                             process per NAND flash channel,
//                                             timing its own channel's PAL)
// All of them are always built by the same toolchain on the same host, so raw
// struct layout is fine to send verbatim through the interchiplet pipe -- no
// portable serialization needed. Anything that changes this header must
// rebuild BOTH TOGSim and the SimpleSSD flash chiplet.

// Fixed-size room for Instruction::get_addr_name(), e.g.
// "model.layers.5.self_attn.q_proj.weight". Longer names are truncated
// (see set_addr_name()) rather than growing the message to a variable size.
constexpr size_t kSsdAddrNameCapacity = 96;

// What a request is asking for.
enum SsdRequestKind : uint8_t {
  // "How long does reading `nbytes` at channel-local flash byte offset
  // `flash_offset` take on your channel?" -- the normal, hot-path request.
  kSsdReqRead = 0,
  // Sentinel: every other field is ignored, the chiplet acknowledges once
  // and then exits its request loop (see SsdLegoSimLink::shutdown()).
  kSsdReqTerminate = 1,
  // Startup handshake: "describe your flash geometry". The reply fills in
  // SsdLatencyResponse's stripe_bytes/capacity_bytes/num_channels/channel_id
  // and leaves latency_ns at 0. This is how the NPU-side flash controller
  // learns the interleaving granularity instead of having it configured
  // twice (once in the SimpleSSD .cfg, once on the NPU) and silently
  // drifting apart -- see SsdLegoSimLink::_handshake().
  kSsdReqGeometry = 2,
};

// TOGSim (NPU with the integrated flash controller) -> one flash chiplet.
struct SsdLatencyRequest {
  // Host DRAM address of the DMA that triggered this. Only carried for
  // logging/identification and for the "formula" backend, which has no
  // notion of a flash address space; the "simplessd" backend addresses NAND
  // with flash_offset below.
  uint64_t addr;
  // Bytes this *particular* chiplet has to serve (already the caller's
  // per-channel share, not the whole DMA -- see SsdLegoSimLink::
  // query_latency_ns()'s striping).
  uint64_t nbytes;
  uint64_t inst_id;
  // Byte offset into THIS channel's own flash address space (the NPU-side
  // controller has already done both the tensor->flash-offset translation
  // and the channel de-interleaving).
  uint64_t flash_offset;
  char addr_name[kSsdAddrNameCapacity];
  // Legacy alias of `kind == kSsdReqTerminate`, kept so the field order (and
  // therefore the meaning of an old-style request) is stable.
  uint8_t terminate;
  uint8_t kind;

  void set_addr_name(const std::string& name) {
    std::strncpy(addr_name, name.c_str(), kSsdAddrNameCapacity - 1);
    addr_name[kSsdAddrNameCapacity - 1] = '\0';
  }
};

// Flash chiplet -> TOGSim.
struct SsdLatencyResponse {
  // Modeled latency in nanoseconds for a kSsdReqRead. Zero for the other
  // two kinds.
  uint64_t latency_ns;
  // --- kSsdReqGeometry only ---
  // Interleaving granularity: one PAL superpage on this channel, in bytes
  // (PAL::Parameter::superPageSize with the chiplet's Channel forced to 1).
  // The NPU-side controller stripes consecutive stripe_bytes-sized chunks
  // round-robin across channels.
  uint64_t stripe_bytes;
  // Total addressable bytes on this one channel. The controller wraps flash
  // offsets into this so a model larger than the configured NAND still
  // produces plausible (if aliased) NAND traffic instead of a panic.
  uint64_t capacity_bytes;
  // How many channels this chiplet was told the whole flash device has, and
  // which one it is. Cross-checked against the NPU's own view.
  uint32_t num_channels;
  uint32_t channel_id;
};
