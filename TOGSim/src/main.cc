#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <atomic>

#include "DramLegoSimLink.h"
#include "Simulator.h"
#include "SsdLegoSimLink.h"
#include "TileGraphParser.h"
#include "helper/CommandLineParser.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;


void launchKernel(Simulator* simulator, unsigned int kernel_id, std::string onnx_path, std::string attribute_path, const YAML::Node& config_yaml, cycle_type request_time=0, int partition_id=0, int device_id=0) {
  auto graph_praser = TileGraphParser(onnx_path, attribute_path, config_yaml);
  std::unique_ptr<TileGraph>& tile_graph = graph_praser.get_tile_graph();
  tile_graph->set_arrival_time(request_time ? request_time : simulator->get_core_cycle());
  tile_graph->set_kernel_id(kernel_id);
  spdlog::info("[Scheduler {}] Enqueued kernel_id: {}, tog_path: {}, operation: {}, request_time_cycles: {}",
               partition_id, kernel_id, onnx_path, tile_graph->get_name(), request_time);
  simulator->enqueue_graph(partition_id, std::move(tile_graph));
}

void process_trace_file(Simulator* simulator, std::string trace_file_path, const YAML::Node& config_yaml) {
  // Open trace file (can be FIFO or regular file)
  std::ifstream trace_file;
  trace_file.open(trace_file_path);
  if (!trace_file.is_open()) {
    spdlog::error("[TOGSim] Failed to open trace file: {}", trace_file_path);
    return;
  }
  spdlog::info("[TOGSim] Reading trace file: {}", trace_file_path);

  // Read all available commands and process them
  std::string line;
  while (std::getline(trace_file, line)) {
    if (line.empty()) {
      continue;
    }

    // Skip comment lines starting with #
    if (line[0] == '#') {
      continue;
    }

    // Parse command: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(iss, token, ',')) {
      tokens.push_back(token);
    }

    if (tokens.size() != 7) {
      spdlog::error("[TOGSim] Invalid command format. Expected: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp. Got: {} ({} tokens)", line, tokens.size());
      continue;
    }

    std::string command_type = tokens[0];
    unsigned int kernel_id = std::stoul(tokens[1]);
    int device_index = std::stoi(tokens[2]);
    int stream_index = std::stoi(tokens[3]);
    std::string tog_path = tokens[4];
    std::string attribute_path = tokens[5];
    int timestamp = std::stoi(tokens[6]);
    // timestamp (tokens[6]) is available but not used in current implementation

    try {
      if (command_type == "LAUNCH_KERNEL") {
        launchKernel(simulator, kernel_id, tog_path, attribute_path, config_yaml, timestamp, stream_index, device_index);
      } else if (command_type == "DEVICE_SYNC") {
        simulator->cycle();
        spdlog::info("[Device {}] Device synchronization completed", device_index);
      } else {
        spdlog::error("[TOGSim] Unknown command type: {}", command_type);
      }
    } catch (const std::exception& e) {
      spdlog::error("[TOGSim] Error processing command {} (type: {}): {}", kernel_id, command_type, e.what());
    }
  }
  trace_file.close();
  simulator->cycle();
}

Simulator* create_simulator(const std::string& config_path) {
  YAML::Node config_yaml;
  if (!loadConfig(config_path, config_yaml)) {
    return nullptr;
  }
  SimulationConfig config = initialize_config(config_yaml, config_path);
  return new Simulator(config, std::move(config_yaml));
}

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file (.yml)");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the trace file (.trace)");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "checkpoint_in", "Optional: path to a checkpoint written by a previous "
      "batch's --checkpoint_out (see Simulator::load_checkpoint) -- restores "
      "the running cycle counters instead of starting from 0.");
  cmd_parser.add_command_line_option<std::string>(
      "checkpoint_out", "Optional: path to write this run's final cycle "
      "counters to (see Simulator::save_checkpoint), for a later batch's "
      "--checkpoint_in to pick up.");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parsing error captured. Error message: {}",
        e.what());
    std::cerr << std::endl;
    cmd_parser.print_help_message();
    exit(1);
  }
  
  // Check if help was requested
  cmd_parser.print_help_message_if_required();

  // Dump full command for copy-paste re-run
  std::ostringstream cmd_oss;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) cmd_oss << " ";
    cmd_oss << argv[i];
  }
  spdlog::info("[TOGSim] Command line: {}", cmd_oss.str());

  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  std::string trace_file_path;

  /* Create simulator */
  cmd_parser.set_if_defined("config", &config_path);

  auto simulator = create_simulator(config_path);
  if (!simulator) {
    spdlog::error("[TOGSim] Failed to load config file: {}", config_path);
    exit(1);
  }

  std::string checkpoint_in;
  cmd_parser.set_if_defined("checkpoint_in", &checkpoint_in);
  if (!checkpoint_in.empty()) {
    simulator->load_checkpoint(checkpoint_in);
  }

  // Get trace file path
  cmd_parser.set_if_defined("models_list", &trace_file_path);

  if (!trace_file_path.empty()) {
    // Process trace file (unified mode: supports both FIFO and regular file)
    process_trace_file(simulator, trace_file_path,
                       simulator->get_hardware_config_yaml());
    spdlog::info("Simulation finished");
    simulator->print_core_stat();

    std::string checkpoint_out;
    cmd_parser.set_if_defined("checkpoint_out", &checkpoint_out);
    if (!checkpoint_out.empty()) {
      simulator->save_checkpoint(checkpoint_out);
    }
  } else {
    spdlog::error("No trace file provided. Use --models_list to specify trace file path.");
    exit(1);
  }
  // If the live LegoSim SSD/DRAM path was used, tell the simlet(s) to stop
  // waiting for more requests so interchiplet's phase1 wait can complete.
  SsdLegoSimLink::instance().shutdown();
  DramLegoSimLink::instance().shutdown();
  delete simulator;

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Wall-clock time for simulation: {:2f} seconds", duration.count());
  return 0;
}
