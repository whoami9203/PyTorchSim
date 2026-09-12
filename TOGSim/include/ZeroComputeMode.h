#pragma once

#include <cstdlib>
#include <string>

// "What if the NPU's compute were free?"
//
// Set TOGSIM_ZERO_COMPUTE=1 and every COMP instruction is charged zero cycles,
// leaving only data movement -- DMA, and in the LegoSim flash configuration
// the D2D round trips and NAND reads behind it (see SsdLegoSimLink.h). The
// resulting runtime is the lower bound the memory system alone imposes, which
// is what you want when the question is "is this workload flash-bound, and by
// how much?" rather than "how fast is this systolic array?".
//
// Implemented at Instruction::get_compute_cycle()/get_overlapping_cycle()
// rather than by skipping instruction issue: COMP instructions still issue,
// still carry their dependencies, and still retire in order, so the tile
// graph, the DMA prefetch distance and the SRAM buffer plan all behave exactly
// as they would in a normal run. Core::cycle() already has a path for a
// zero-cycle COMP (it retires the instruction immediately), so this reuses it.
//
// Read once and cached: it is consulted per instruction on the hot path.
class ZeroComputeMode {
 public:
  static bool enabled() {
    static const bool on = [] {
      const char* value = std::getenv("TOGSIM_ZERO_COMPUTE");
      return value != nullptr && std::string(value) == "1";
    }();
    return on;
  }
};
