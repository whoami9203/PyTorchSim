TRACE_NAME="${TOGSIM_SSD_TRACE_NAME:?TOGSIM_SSD_TRACE_NAME is not set}"
TORCHSIM_DIR="${TORCHSIM_DIR:?TORCHSIM_DIR is not set}"

# Whether to also remove outputs/*. Pass 1 to remove, 0 (default) to keep.
REMOVE_OUTPUTS=${1:-0}

if [[ "${REMOVE_OUTPUTS}" != "0" && "${REMOVE_OUTPUTS}" != "1" ]]; then
	echo "Usage: $0 [REMOVE_OUTPUTS]   (1=remove outputs/*, 0=keep; default 0)"
	exit 1
fi

shopt -s dotglob
rm -rf ${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}/*
if [[ "${REMOVE_OUTPUTS}" == "1" ]]; then
	rm -rf ${TORCHSIM_DIR}/outputs/*
fi
rm -rf ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}/*
rm -rf ${TORCHSIM_DIR}/validation/${TRACE_NAME}/gemm_candidates/*.txt
shopt -u dotglob