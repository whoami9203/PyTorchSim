#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 TRACE_NAME [TOGSIM_LEGOSIM_SSD] [TOGSIM_LEGOSIM_DRAM] [TOGSIM_CONFIG_PATH]"
	exit 1
fi

# bash cleanup_results.sh

TRACE_NAME=$1
TOGSIM_LEGOSIM_SSD=${2:-0}
TOGSIM_LEGOSIM_DRAM=${3:-0}
TOGSIM_CONFIG_PATH=${4:-}
# A real popnet always runs in phase2 (instead of the default no-op filler)
# whenever either simlet above is enabled, so interchiplet's two-phase
# fixed-point loop always runs for real testing -- no separate NoC toggle.
# See Simulator/simulator.py's _build_legosim_yaml and
# TOGSim/legosim/{ssd,dram}_simlet.cpp's timeNow tracking.

if [[ "${TOGSIM_LEGOSIM_SSD}" != "0" && "${TOGSIM_LEGOSIM_SSD}" != "1" ]]; then
	echo "Error: TOGSIM_LEGOSIM_SSD must be 0 or 1 (got '${TOGSIM_LEGOSIM_SSD}')"
	exit 1
fi

if [[ "${TOGSIM_LEGOSIM_DRAM}" != "0" && "${TOGSIM_LEGOSIM_DRAM}" != "1" ]]; then
	echo "Error: TOGSIM_LEGOSIM_DRAM must be 0 or 1 (got '${TOGSIM_LEGOSIM_DRAM}')"
	exit 1
fi

if [[ -n "${TOGSIM_CONFIG_PATH}" && ! -f "${TOGSIM_CONFIG_PATH}" ]]; then
	echo "Error: TOGSIM_CONFIG_PATH '${TOGSIM_CONFIG_PATH}' does not exist"
	exit 1
fi

export TORCHSIM_DIR=/workspace/legomerged/eclab_legosim/PyTorchSim
export PYTORCHSIM_ROOT_PATH=${TORCHSIM_DIR}
export TOGSIM_DEBUG_LEVEL=info
export TOGSIM_SSD_TRACE_NAME=${TRACE_NAME}
export TOGSIM_LEGOSIM_SSD=${TOGSIM_LEGOSIM_SSD}
export TOGSIM_LEGOSIM_DRAM=${TOGSIM_LEGOSIM_DRAM}

mkdir -p ${TORCHSIM_DIR}/ssd_traces
mkdir -p ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}
mkdir -p ${TORCHSIM_DIR}/togsim_results
mkdir -p ${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TOGSIM_SSD_TRACE_DIR=${TORCHSIM_DIR}/ssd_traces

LOG_DIR=${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TORCHSIM_LOG_PATH=${LOG_DIR}
export TOGSIM_CONFIG=${TORCHSIM_DIR}/${TOGSIM_CONFIG_PATH}

export TORCHSIM_DEBUG_MODE=0
