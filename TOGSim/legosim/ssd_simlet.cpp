// LegoSim simlet standing in for a real SSD/DRAM timing model. Answers
// latency queries from TOGSim's DMA engine (see SsdLegoSimLink in
// TOGSim/include/SsdLegoSimLink.h) over the interchiplet sync protocol.
//
// Model: the SSD only transfers whole 4KB pages, so a request is first
// rounded up to the pages it touches (addr-aligned, not just size-aligned --
// an unaligned nbytes-byte request can span one more page than
// ceil(nbytes / page_size) would suggest), then:
// latency_ns = base_latency_ns + ceil(pages * page_size / bandwidth_GBps)
// (1 GB/s == 1 byte/ns, so bytes / bandwidth_GBps is already in ns).
// This is a placeholder -- swap compute_latency_ns() for a real SSD/DRAM
// simulator's estimate to get actual modeled numbers.
//
// Like artifact/HBM_DDR/DDR.cpp/HBM.cpp and dram_simlet.cpp, this simlet
// tracks a running `timeNow` across requests instead of always reporting
// cycle 0 -- this is what lets the phase-2 NoC simlet (always plugged in
// whenever SsdLegoSimLink is enabled -- see _build_legosim_yaml's use_ssd
// handling in Simulator/simulator.py) actually influence the read/write
// pairing (interchiplet's getEndCycle()) instead of being computed against
// a fixed baseline every time.
//
// argv: <self_x> <self_y> <peer_x> <peer_y> [bandwidth_gbps] [base_latency_ns]
// Defaults match the (0,0)=NPU / (1,0)=DRAM convention used elsewhere in
// this integration (e.g. tests/Llama/test_legosim_integration.py).

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
    std::cerr << "usage: ssd_simlet <self_x> <self_y> <peer_x> <peer_y> "
              << "[bandwidth_gbps=256] [base_latency_ns=100]" << std::endl;
    return 1;
  }
  const int self_x = std::atoi(argv[1]);
  const int self_y = std::atoi(argv[2]);
  const int peer_x = std::atoi(argv[3]);
  const int peer_y = std::atoi(argv[4]);
  const double bandwidth_gbps = argc > 5 ? std::atof(argv[5]) : 256.0;
  const double base_latency_ns = argc > 6 ? std::atof(argv[6]) : 100.0;

  InterChiplet::PipeComm pipe_comm;

  // Current known simulated time for this chiplet, threaded through
  // readSync/writeSync's cycle argument -- same role as
  // DDR.cpp/HBM.cpp/dram_simlet.cpp's local `timeNow`.
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
      std::cout << "[ssd_simlet] addr=0x" << std::hex << req.addr << std::dec
                << " nbytes=" << req.nbytes << " inst_id=" << req.inst_id
                << " addr_name=" << req.addr_name
                << " -> latency_ns=" << resp.latency_ns << std::endl;

      // Advance timeNow past the modeled latency, added on top of wherever
      // the request actually landed (time_end -- informed by the phase-2
      // NoC delay). Mirrors dram_simlet.cpp's identical update.
      timeNow = static_cast<InterChiplet::TimeType>(resp.latency_ns) + time_end;
    } else {
      resp.latency_ns = 0;
      timeNow = time_end;
    }

    // Report our own advancing cycle to interchiplet's top-level "Benchmark
    // elapses N cycle."/convergence bookkeeping -- fire-and-forget, no
    // reply to wait for. See dram_simlet.cpp's identical call for why this
    // is sendCycleCmd() and not InterChiplet::cycleSync() (which would
    // block forever: handle_cycle_cmd never sends a SYNC response for a
    // CYCLE command).
    InterChiplet::sendCycleCmd(timeNow);

    // Send the response back (also used to ack the terminate sentinel).
    std::string resp_file = InterChiplet::sendSync(self_x, self_y, peer_x, peer_y);
    pipe_comm.write_data(resp_file.c_str(), &resp, sizeof(resp));
    InterChiplet::writeSync(timeNow, self_x, self_y, peer_x, peer_y, sizeof(resp), 0);

    if (req.terminate) {
      std::cout << "[ssd_simlet] received terminate sentinel, exiting." << std::endl;
      break;
    }
  }

  return 0;
}
