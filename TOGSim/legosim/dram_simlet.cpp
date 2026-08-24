// LegoSim simlet standing in for a real DRAM+interconnect timing model.
// Answers latency queries from TOGSim's DMA engine (see DramLegoSimLink in
// TOGSim/include/DramLegoSimLink.h) over the interchiplet sync protocol.
// Unlike ssd_simlet.cpp (which only covers weight-tensor DMA reads), this
// simlet is meant to be the catch-all for every DMA access -- reads and
// writes alike -- replacing TOGSim's real Dram/Interconnect models
// entirely for the duration of a run.
//
// Model: same page-granular bandwidth+base-latency formula as
// ssd_simlet.cpp (request rounded up to whole 4KB pages it touches,
// addr-aligned, then latency_ns = base_latency_ns + ceil(pages * page_size
// / bandwidth_GBps); 1 GB/s == 1 byte/ns). This is a placeholder -- swap
// compute_latency_ns() for a real DRAM/NoC simulator's estimate to get
// actual modeled numbers.
//
// Like artifact/HBM_DDR/DDR.cpp and HBM.cpp, this simlet tracks a running
// `timeNow` across requests instead of always reporting cycle 0 -- this is
// what lets a real phase-2 NoC simlet (see TOGSIM_LEGOSIM_DRAM_NOC in
// Simulator/simulator.py) actually influence the read/write pairing
// (interchiplet's getEndCycle()) instead of being computed against a fixed
// baseline every time.
//
// This also already gives every core sharing this one simlet process (all
// requests funnel through the one DramLegoSimLink singleton -- see its own
// comment) correct FCFS queueing on the shared channel, with no extra
// bookkeeping needed here: timeNow is passed as the cycle argument to every
// readSync() call below, and interchiplet's own resolution
// (net_delay.h's DelayList::getEndCycle(), "normal communication" branch)
// clamps a request's resolved arrival to max(its own declared cycle +
// transport delay, that timeNow) -- i.e. never earlier than when the
// channel finished the previous request. A `channel_free_at` variable
// shadowing timeNow was tried here and reverted: it was provably always
// equal to timeNow at the point it would matter (proven by an isolated
// interchiplet-level A/B test sending two overlapping requests: identical
// resolved cycles with and without it), so it was dead weight that only
// made the already-correct queueing look like it depended on this file
// instead of on interchiplet's own protocol.
//
// argv: <self_x> <self_y> <peer_x> <peer_y> [bandwidth_gbps] [base_latency_ns]
// Defaults match the (0,0)=NPU / (2,0)=DRAM convention used elsewhere in
// this integration (ssd_simlet occupies (1,0), so dram_simlet can run
// alongside it without colliding).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "pipe_comm.h"
#include "ssd_protocol.h"

namespace {

constexpr uint64_t kPageSizeBytes = 4096;

// Number of kPageSizeBytes-sized pages spanned by [addr, addr + nbytes).
uint64_t pages_touched(uint64_t addr, uint64_t nbytes) {
  if (nbytes == 0) return 0;
  uint64_t start_page = addr / kPageSizeBytes;
  uint64_t end_page = (addr + nbytes - 1) / kPageSizeBytes;
  return end_page - start_page + 1;
}

uint64_t compute_latency_ns(uint64_t addr, uint64_t nbytes, double bandwidth_gbps, double base_latency_ns) {
  uint64_t effective_bytes = pages_touched(addr, nbytes) * kPageSizeBytes;
  double transfer_ns = bandwidth_gbps > 0.0 ? static_cast<double>(effective_bytes) / bandwidth_gbps : 0.0;
  double total_ns = base_latency_ns + transfer_ns;
  return static_cast<uint64_t>(std::ceil(total_ns));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: dram_simlet <self_x> <self_y> <peer_x> <peer_y> "
              << "[bandwidth_gbps=800] [base_latency_ns=15]" << std::endl;
    return 1;
  }
  const int self_x = std::atoi(argv[1]);
  const int self_y = std::atoi(argv[2]);
  const int peer_x = std::atoi(argv[3]);
  const int peer_y = std::atoi(argv[4]);
  const double bandwidth_gbps = argc > 5 ? std::atof(argv[5]) : 76.0;
  const double base_latency_ns = argc > 6 ? std::atof(argv[6]) : 15.0;

  InterChiplet::PipeComm pipe_comm;

  // Current known simulated time for this chiplet, threaded through
  // readSync/writeSync's cycle argument -- same role as DDR.cpp/HBM.cpp's
  // local `timeNow`. This is also what gives every core sharing this one
  // process correct FCFS queueing on the shared channel for free -- see
  // the file comment.
  InterChiplet::TimeType timeNow = 1;

  while (true) {
    // Receive one request from TOGSim.
    std::string req_file = InterChiplet::receiveSync(peer_x, peer_y, self_x, self_y);
    SsdLatencyRequest req{};
    pipe_comm.read_data(req_file.c_str(), &req, sizeof(req));
    InterChiplet::TimeType time_end =
        InterChiplet::readSync(timeNow, peer_x, peer_y, self_x, self_y, sizeof(req), 0);

    SsdLatencyResponse resp{};
    if (!req.terminate) {
      resp.latency_ns = compute_latency_ns(req.addr, req.nbytes, bandwidth_gbps, base_latency_ns);
      std::cout << "[dram_simlet] addr=0x" << std::hex << req.addr << std::dec
                << " nbytes=" << req.nbytes << " inst_id=" << req.inst_id
                << " addr_name=" << req.addr_name
                << " -> latency_ns=" << resp.latency_ns << std::endl;

      // Advance timeNow past the modeled DRAM access latency, added on top
      // of wherever the request actually landed (time_end -- informed by
      // phase-2 NoC delay when a real NoC simlet is plugged in, and already
      // clamped by interchiplet to be no earlier than the channel's own
      // previous timeNow -- see the file comment). Mirrors DDR.cpp/HBM.cpp's
      // `timeNow = true_time + time_end`. latency_ns is used directly as a
      // cycle count here, the same placeholder convention DDR.cpp uses for
      // its own fixed constant; this internal timeline is independent of
      // the ns-to-core-cycle conversion DMA.cc applies to resp.latency_ns
      // on the TOGSim side.
      timeNow = static_cast<InterChiplet::TimeType>(resp.latency_ns) + time_end;
    } else {
      resp.latency_ns = 0;
      timeNow = time_end;
    }

    // Report our own advancing cycle to interchiplet's top-level "Benchmark
    // elapses N cycle."/convergence bookkeeping (interchiplet.cpp's
    // round_cycle, fed by SyncClockStruct::update()'s running max) -- fire-
    // and-forget, no reply to wait for (unlike readSync/writeSync above):
    // handle_cycle_cmd (cmd_handler.cpp) never sends a SYNC response for a
    // CYCLE command, so InterChiplet::cycleSync() (which blocks waiting for
    // one) would hang here forever. sendCycleCmd() is the correct,
    // documented (docs/docs/04-import-sim/index.md) non-blocking call.
    InterChiplet::sendCycleCmd(timeNow);

    // Send the response back (also used to ack the terminate sentinel).
    std::string resp_file = InterChiplet::sendSync(self_x, self_y, peer_x, peer_y);
    pipe_comm.write_data(resp_file.c_str(), &resp, sizeof(resp));
    InterChiplet::writeSync(timeNow, self_x, self_y, peer_x, peer_y, sizeof(resp), 0);

    if (req.terminate) {
      std::cout << "[dram_simlet] received terminate sentinel, exiting." << std::endl;
      break;
    }
  }

  return 0;
}
