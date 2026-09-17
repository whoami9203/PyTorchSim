#pragma once

#include <cstdint>
#include <vector>

// Loads model_weight_ranges_1d.txt (written by merge_weight_ranges.py, which
// filters the per-tensor model_weight_ranges.txt down to just the 1-D
// weights -- LayerNorm/RMSNorm weight and bias vectors -- and keeps each as
// its own row rather than coalescing it with a neighbor, unlike
// model_weight_ranges_merged.txt/WeightAddressRanges).
//
// These tensors are a few KB each, and the compiled kernel that consumes
// them (generic Inductor scheduling, no dedicated template the way GEMM
// inputs get one) DMAs each in many small fragments. Routing every fragment
// through the live SSD path would charge its (tiny, real) per-request
// latency once per fragment, wildly overstating the cost of loading a few
// KB. Instead: the DMA that first touches a tracked tensor (address ==
// tensor base) is billed once for the tensor's *full* size; every later DMA
// landing inside the same [base, end) before the tensor is reloaded is
// treated as an ordinary DRAM access with no extra SSD latency, matching
// that once a tensor is resident in DRAM it isn't re-fetched from SSD per
// cache line.
//
// A tracked tensor's address is freed and reused once per decoder layer
// that streams over it, so a fresh "address == base" touch normally means
// "next layer's load of this same tensor". But the address can also be
// reused by something this file never tracks at all (e.g. the model's final
// norm weight, loaded once outside the per-layer streaming loop) after the
// last real layer -- billing would then keep firing for a tensor that
// doesn't exist anymore. Guarded by capping the number of charges per
// tracked tensor at (layer index parsed from its name) + 1.
class SingleShotWeightGate {
 public:
  static SingleShotWeightGate& instance();

  bool enabled() const { return _enabled; }

  // If `addr` falls inside a tracked 1-D tensor, returns true and sets
  // *should_charge (whether this specific DMA should be billed SSD latency)
  // and *tensor_bytes (the full tensor size to bill when it should).
  // Returns false if `addr` isn't inside any tracked tensor -- the caller
  // should fall back to its normal per-fragment weight billing in that case.
  bool classify(uint64_t addr, bool* should_charge, uint64_t* tensor_bytes);

 private:
  SingleShotWeightGate();

  struct Tracked {
    uint64_t base;
    uint64_t end;
    uint64_t size_bytes;
    int max_loads;    // layer_index + 1 parsed from the tensor's name, or -1 if unbounded
    int loads_seen = 0;
  };

  std::vector<Tracked> _tensors;  // sorted by base
  bool _enabled = false;
};
