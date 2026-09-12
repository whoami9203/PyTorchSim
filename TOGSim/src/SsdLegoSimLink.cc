#include "SsdLegoSimLink.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <spdlog/spdlog.h>

#include "FlashAddressMap.h"

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

SsdLegoSimLink& SsdLegoSimLink::instance() {
  static SsdLegoSimLink link;
  return link;
}

SsdLegoSimLink::SsdLegoSimLink() {
  const char* on = std::getenv("TOGSIM_SSD_LEGOSIM");
  _enabled = on && std::string(on) == "1";
  if (!_enabled) return;

  _self_x = env_long("TOGSIM_LEGOSIM_X", 0);
  _self_y = env_long("TOGSIM_LEGOSIM_Y", 0);
  _core_freq_mhz = env_double("TOGSIM_SSD_LEGOSIM_CORE_FREQ_MHZ", 0.0);

  const long base_x = env_long("TOGSIM_SSD_LEGOSIM_X", 1);
  const long base_y = env_long("TOGSIM_SSD_LEGOSIM_Y", 0);
  long count = env_long("TOGSIM_SSD_LEGOSIM_NUM_CHANNELS", 1);
  if (count < 1) count = 1;

  // Channels occupy consecutive X coordinates starting at the base. Keep this
  // in step with _build_legosim_yaml()'s phase1 process list and the popnet
  // topology it generates.
  _channels.resize(static_cast<size_t>(count));
  for (long i = 0; i < count; i++) {
    _channels[static_cast<size_t>(i)].x = base_x + i;
    _channels[static_cast<size_t>(i)].y = base_y;
  }

  const long stripe_override = env_long("TOGSIM_SSD_LEGOSIM_STRIPE_BYTES", 0);
  if (stripe_override > 0) {
    _stripe_bytes = static_cast<uint64_t>(stripe_override);
    // An explicit stripe size means the caller has already decided the
    // interleaving; don't let the handshake overwrite it.
    _handshake_done = true;
  }

  spdlog::info(
      "[SsdLegoSimLink] enabled: self=({},{}) flash channels={} at x=[{}..{}] y={} "
      "core_freq_mhz={}",
      _self_x, _self_y, _channels.size(), base_x, base_x + count - 1, base_y, _core_freq_mhz);
}

void SsdLegoSimLink::issue(Channel& channel, const SsdLatencyRequest& req,
                           InterChiplet::TimeType cycle) {
  // Request leg: self -> channel, issued at the caller's real core_cycle (not
  // a placeholder) so interchiplet's read/write pairing resolves against
  // TOGSim's actual timeline -- required for the real phase-2 NoC simlet's
  // delay to land on this transfer instead of being computed against a fixed
  // baseline every time.
  std::string req_file = InterChiplet::sendSync(_self_x, _self_y, channel.x, channel.y);
  _pipe_comm.write_data(req_file.c_str(), const_cast<SsdLatencyRequest*>(&req), sizeof(req));
  InterChiplet::writeSync(cycle, _self_x, _self_y, channel.x, channel.y, sizeof(req), 0);
}

InterChiplet::TimeType SsdLegoSimLink::collect(Channel& channel, SsdLatencyResponse* resp,
                                               InterChiplet::TimeType cycle) {
  // Response leg: channel -> self. The chiplet already folded its own modeled
  // NAND latency into the cycle it wrote the response at, so the resolved
  // cycle interchiplet hands back here already reflects that latency plus
  // whatever transport delay it resolved for both legs of the round trip.
  std::string resp_file = InterChiplet::receiveSync(channel.x, channel.y, _self_x, _self_y);
  _pipe_comm.read_data(resp_file.c_str(), resp, sizeof(*resp));
  return InterChiplet::readSync(cycle, channel.x, channel.y, _self_x, _self_y, sizeof(*resp), 0);
}

void SsdLegoSimLink::handshake() {
  if (_handshake_done) return;
  _handshake_done = true;

  SsdLatencyRequest req{};
  req.kind = kSsdReqGeometry;
  req.set_addr_name("<geometry>");

  SsdLatencyResponse resp{};
  issue(_channels[0], req, 0);
  collect(_channels[0], &resp, 0);

  if (resp.stripe_bytes == 0) {
    spdlog::warn(
        "[SsdLegoSimLink] flash chiplet reported no geometry; keeping stripe={} B. (Older "
        "chiplet build, or the 'formula' backend?)",
        _stripe_bytes);
    return;
  }

  _stripe_bytes = resp.stripe_bytes;
  if (resp.num_channels != _channels.size()) {
    spdlog::warn(
        "[SsdLegoSimLink] flash chiplet says the device has {} channel(s) but LegoSim spawned "
        "{} -- the modeled device is not the configured one.",
        resp.num_channels, _channels.size());
  }
  spdlog::info(
      "[SsdLegoSimLink] flash geometry: stripe={} B/channel, capacity={} B/channel, {} channel(s) "
      "-> {} B total, {} B per full stripe round",
      resp.stripe_bytes, resp.capacity_bytes, _channels.size(),
      resp.capacity_bytes * _channels.size(), _stripe_bytes * _channels.size());
}

size_t SsdLegoSimLink::plan(uint64_t flash_offset, uint64_t nbytes) {
  const uint64_t n = _channels.size();
  const uint64_t first = flash_offset / _stripe_bytes;
  const uint64_t last = (flash_offset + nbytes - 1) / _stripe_bytes;

  size_t touched = 0;
  for (uint64_t c = 0; c < n; c++) {
    Channel& channel = _channels[static_cast<size_t>(c)];
    channel.pending_bytes = 0;

    // Stripe s lives on channel s % n, so the stripes of [first, last] that
    // belong to channel c are the arithmetic progression c, c+n, c+2n, ...
    // clipped to that window. Computed directly rather than by walking every
    // stripe: one DMA can span a lot of them.
    const uint64_t lo = first + ((c + n) - (first % n)) % n;
    if (lo > last) continue;
    const uint64_t hi = last - ((last % n) + n - c) % n;

    // Consecutive stripes on one channel are consecutive in that channel's own
    // address space, so the share is always one contiguous run -- no holes.
    channel.pending_offset = (lo / n) * _stripe_bytes;
    channel.pending_bytes = ((hi / n) - (lo / n) + 1) * _stripe_bytes;
    touched++;
  }
  return touched;
}

uint64_t SsdLegoSimLink::query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                                          const std::string& addr_name, uint64_t core_cycle) {
  if (nbytes == 0) return 0;
  handshake();

  // The integrated controller's address translation: where does the tensor
  // this DMA is reading actually sit in flash?
  uint64_t flash_offset = 0;
  FlashAddressMap::instance().translate(addr, &flash_offset);
  plan(flash_offset, nbytes);

  const auto cycle = static_cast<InterChiplet::TimeType>(core_cycle);

  // Fan out to every channel the access touches before waiting on any of them:
  // the channels work in parallel, which is the point of having several. Each
  // chiplet is idle in its own receiveSync() when this arrives, so issuing to
  // one never blocks on another having answered.
  SsdLatencyRequest req{};
  req.addr = addr;
  req.inst_id = inst_id;
  req.kind = kSsdReqRead;
  req.terminate = 0;
  req.set_addr_name(addr_name);
  for (Channel& channel : _channels) {
    if (channel.pending_bytes == 0) continue;
    req.flash_offset = channel.pending_offset;
    req.nbytes = channel.pending_bytes;
    issue(channel, req, cycle);
  }

  // Fan in. The DMA is not complete until its last byte lands, so the access
  // costs whatever the slowest channel cost.
  InterChiplet::TimeType slowest = cycle;
  uint64_t fallback_latency_ns = 0;
  for (Channel& channel : _channels) {
    if (channel.pending_bytes == 0) continue;
    SsdLatencyResponse resp{};
    InterChiplet::TimeType resolved = collect(channel, &resp, cycle);
    slowest = std::max(slowest, resolved);
    fallback_latency_ns = std::max(fallback_latency_ns, resp.latency_ns);
  }

  if (_core_freq_mhz <= 0.0) {
    // TOGSIM_SSD_LEGOSIM_CORE_FREQ_MHZ wasn't set (shouldn't happen via the
    // normal simulator.py path, but guard anyway) -- fall back to the
    // chiplets' own modeled latency alone, ignoring transport delay, rather
    // than dividing by zero.
    return fallback_latency_ns;
  }
  // `slowest` is already back in TOGSim's own core-cycle domain (interchiplet
  // converts via this process's own clock_rate before acking) -- convert the
  // elapsed cycles to ns for this method's contract; DMA.cc converts back to
  // cycles itself.
  InterChiplet::TimeType elapsed_cycles = slowest > cycle ? slowest - cycle : 1;
  return static_cast<uint64_t>(std::ceil(elapsed_cycles * (1000.0 / _core_freq_mhz)));
}

void SsdLegoSimLink::shutdown() {
  if (!_enabled || _shutdown_sent) return;
  _shutdown_sent = true;

  SsdLatencyRequest req{};
  req.kind = kSsdReqTerminate;
  req.terminate = 1;
  for (Channel& channel : _channels) {
    issue(channel, req, 0);
  }
  for (Channel& channel : _channels) {
    SsdLatencyResponse resp{};
    collect(channel, &resp, 0);
  }
  spdlog::info("[SsdLegoSimLink] sent terminate sentinel, {} flash chiplet(s) acked.",
               _channels.size());
}
