#!/bin/bash
# GPU1-only quantized Qwen 8 meta-action suite with ClassicCV image enhancement.
# Designed to coexist with a separate GPU0 CARLA run by using separate ports,
# lock file, and no broad CARLA cleanup between runs.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export SUITE_ROOT=${SUITE_ROOT:-/mnt/2/carla_metric_result/qwen_8meta_classiccv_quant_gpu1_${RUN_STAMP}}
export SUITE_LOCK=${SUITE_LOCK:-/tmp/qwen_8meta_classiccv_quant_gpu1.lock}

# Use only physical GPU 1. Inside the process this is cuda:0.
export CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-1}
export GPU_RANK=${GPU_RANK:-0}
export TRANSFORMER_QWEN_DEVICE=${TRANSFORMER_QWEN_DEVICE:-cuda:0}
export VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
export VLLM_PHYSICAL_GPU_INDEX=${VLLM_PHYSICAL_GPU_INDEX:-1}

# Separate ports from the GPU0 220 run.
export BASE_PORT=${BASE_PORT:-30144}
export BASE_TM_PORT=${BASE_TM_PORT:-50144}
export VLLM_PORT_BASE=${VLLM_PORT_BASE:-8130}

# Do not kill an unrelated CARLA/leaderboard process that may be running on GPU0.
export CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-0}
export MAX_RETRIES=${MAX_RETRIES:-999}
export RESTART_WAIT=${RESTART_WAIT:-30}
export RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE:-0}
export RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD:-20}

# Eunsu-style image-enhanced 8meta VLA.
export QWEN_8META_IMAGE_ENHANCER=${QWEN_8META_IMAGE_ENHANCER:-classic_cv}
export QWEN_DASHBOARD_REAR=0
export QWEN_EMERGENCY_PULL_OVER=0
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=0
export QWEN_SAVE_DASHBOARD=${QWEN_SAVE_DASHBOARD:-0}
export QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}

# Default to dev10 for quick quant comparison. Override ROUTES for 220.
export ROUTES=${ROUTES:-${SCRIPT_DIR}/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment : Qwen 8meta-action + ClassicCV + quant suite"
echo "GPU        : physical GPU1 only (CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST})"
echo "Routes     : ${ROUTES}"
echo "Suite root : ${SUITE_ROOT}"
echo "Ports      : CARLA=${BASE_PORT}, TM=${BASE_TM_PORT}, vLLM base=${VLLM_PORT_BASE}"
echo "Cleanup    : CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN}, MAX_RETRIES=${MAX_RETRIES}, RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD}"
echo "Enhancer   : QWEN_8META_IMAGE_ENHANCER=${QWEN_8META_IMAGE_ENHANCER}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec "${SCRIPT_DIR}/run_qwen_8meta_action_quant_suite.sh" "$@"
