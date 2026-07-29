#!/bin/bash
# QwenSensorAgent dev10 quantization suite with ClassicCV image enhancement.
# This keeps the previously successful front-only clean QwenSensorAgent setup
# and only turns on QWEN_IMAGE_ENHANCER=classic_cv for rgb_front.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export SUITE_ROOT=${SUITE_ROOT:-/mnt/2/carla_metric_result/qwen_dev10_quant_suite_frontonly_classiccv_${RUN_STAMP}}
export SUITE_LOCK=${SUITE_LOCK:-/tmp/qwen_dev10_quant_suite_frontonly_classiccv.lock}

export ROUTES=${ROUTES:-${SCRIPT_DIR}/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml}
export RUNS=${RUNS:-bf16_transformers,bf16_vllm,w8a8_int8_vllm,awq_w4a16_n64_vllm,gptq_w4a16_n64_vllm}

# Same front-only clean switches as qwen_dev10_quant_suite_frontonly_clean.
export QWEN_DASHBOARD_REAR=0
export QWEN_EMERGENCY_PULL_OVER=0
export QWEN_EMERGENCY_REAR_PROBE_STEPS=999999
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=0
export QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}

# Image enhancement: apply ClassicCV only to the front camera used by Qwen.
export QWEN_IMAGE_ENHANCER=${QWEN_IMAGE_ENHANCER:-classic_cv}
export QWEN_IMAGE_ENHANCE_TARGETS=${QWEN_IMAGE_ENHANCE_TARGETS:-rgb_front}
export QWEN_IMAGE_ENHANCE_SAVE_COMPARE=${QWEN_IMAGE_ENHANCE_SAVE_COMPARE:-0}
export QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL=${QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL:-20}

# Keep the quant/runtime defaults from the clean run unless the caller overrides.
export CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-0,1}
export GPU_RANK=${GPU_RANK:-0}
export TRANSFORMER_QWEN_DEVICE=${TRANSFORMER_QWEN_DEVICE:-cuda:1}
export VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
export VLLM_PHYSICAL_GPU_INDEX=${VLLM_PHYSICAL_GPU_INDEX:-${VLLM_CUDA_VISIBLE_DEVICES%%,*}}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment : QwenSensorAgent + ClassicCV + dev10 quant suite"
echo "Routes     : ${ROUTES}"
echo "Runs       : ${RUNS}"
echo "Suite root : ${SUITE_ROOT}"
echo "Enhancer   : ${QWEN_IMAGE_ENHANCER}, targets=${QWEN_IMAGE_ENHANCE_TARGETS}, compare=${QWEN_IMAGE_ENHANCE_SAVE_COMPARE}"
echo "Front-only : rear=${QWEN_DASHBOARD_REAR}, emergency=${QWEN_EMERGENCY_PULL_OVER}, force_lane=${QWEN_FORCE_EMERGENCY_LANE_CHANGE}"
echo "Debug viz  : DEBUG_CHALLENGE=${DEBUG_CHALLENGE}"
echo "CUDA vis   : ${CUDA_VISIBLE_DEVICES_LIST}; TF++ gpu-rank=${GPU_RANK}; vLLM GPU=${VLLM_CUDA_VISIBLE_DEVICES}"
echo "Ports      : CARLA=${BASE_PORT:-30000}, TM=${BASE_TM_PORT:-50000}, vLLM base=${VLLM_PORT_BASE:-8010}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec "${SCRIPT_DIR}/run_qwen_dev10_quant_suite.sh" "$@"
