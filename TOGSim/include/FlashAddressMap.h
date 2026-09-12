#pragma once

#include <cstdint>
#include <vector>

// Where does this weight tensor live in flash?
//
// In the architecture this models, the NPU has the flash controller
// integrated, so address translation is the NPU's job: the flash chiplets on
// the other side of the D2D link are raw NAND (SimpleSSD's PAL layer only --
// see SimpleSSD-Standalone/sim/legosim_flash_chiplet_main.cc) and understand
// nothing but "read N bytes at offset X of your own channel". This class is
// the NPU-side half of that: the flat-address-space part of what an SSD's FTL
// would normally do.
//
// It loads ssd_offsets.tsv -- "<name>\t<dram_base>\t<dram_end>\t<ssd_offset>\t
// <ssd_length>" per tensor, regenerated for whichever layer is current by
// _dump_module_weight_ranges()'s build_layer_ssd_offset_map.py call (see
// tests/Llama/test_llama2_7B.py) -- and maps a DMA's real host DRAM address to
// the flash byte offset that tensor was laid out at.
//
// Matching is by address, not by name, for the same reason DMA.cc classifies
// weight DMAs by address: TOGSim's addr_name is a positional kernel-argument
// label ("arg0", "arg1", ...) reused across unrelated kernels, not a stable
// tensor identity. PyTorchSimDevice's tensors are backed by real host memory,
// so a tensor's data_ptr() on the Python side is the exact address seen here.
//
// Loaded lazily, once, from TOGSIM_SSD_TRACE_DIR/TOGSIM_SSD_TRACE_NAME/
// ssd_offsets.tsv (the same env vars WeightAddressRanges uses). A fresh TOGSim
// process is spawned per batch, so loading once at startup is enough. If the
// file is missing, translate() falls back to using the DRAM address itself as
// the flash offset: striping across channels stays deterministic and every
// channel still sees a representative share of the traffic, the layout just
// isn't the one build_layer_ssd_offset_map.py planned.
class FlashAddressMap {
 public:
  static FlashAddressMap& instance();

  bool enabled() const { return _enabled; }

  // Flash byte offset corresponding to host DRAM address `addr`. Returns true
  // when `addr` fell inside a known tensor, false when the fallback was used.
  bool translate(uint64_t addr, uint64_t* flash_offset) const;

 private:
  FlashAddressMap();

  struct Entry {
    uint64_t dram_base;
    uint64_t dram_end;    // exclusive
    uint64_t ssd_offset;
    uint64_t ssd_length;
  };
  std::vector<Entry> _entries;  // sorted by dram_base
  bool _enabled = false;
};
