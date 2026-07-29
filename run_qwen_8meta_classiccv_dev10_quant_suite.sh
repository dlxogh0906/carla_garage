#!/bin/bash
# Qwen 8 meta-action quantization suite on the 10-route dev set with ClassicCV
# image enhancement enabled.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export SUITE_ROOT=${SUITE_ROOT:-/mnt/2/carla_metric_result/qwen_8meta_classiccv_dev10_quant_suite_${RUN_STAMP}}
export SUITE_LOCK=${SUITE_LOCK:-/tmp/qwen_8meta_classiccv_dev10_quant_suite.lock}

export ROUTES=${ROUTES:-${SCRIPT_DIR}/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml}
export RUNS=${RUNS:-bf16_transformers,bf16_vllm,w8a8_int8_vllm,awq_w4a16_n64_vllm,gptq_w4a16_n64_vllm}

# Use GPU0 for TF++/CARLA and GPU1 for Qwen/vLLM when both A6000s are visible.
export CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-0,1}
export GPU_RANK=${GPU_RANK:-0}
export TRANSFORMER_QWEN_DEVICE=${TRANSFORMER_QWEN_DEVICE:-cuda:1}
export VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
export VLLM_PHYSICAL_GPU_INDEX=${VLLM_PHYSICAL_GPU_INDEX:-${VLLM_CUDA_VISIBLE_DEVICES%%,*}}

export BASE_PORT=${BASE_PORT:-30000}
export BASE_TM_PORT=${BASE_TM_PORT:-50000}
export VLLM_PORT_BASE=${VLLM_PORT_BASE:-8030}

export QWEN_8META_IMAGE_ENHANCER=${QWEN_8META_IMAGE_ENHANCER:-classic_cv}
export TEAM_AGENT=${TEAM_AGENT:-${SCRIPT_DIR}/team_code/eunsu_sensor_agent_meta_action_classic_cv.py}
export QWEN_MODEL_LABEL=${QWEN_MODEL_LABEL:-Qwen3-VL-8B-Eunsu8MetaClassicCV}
export QWEN_METHOD=${QWEN_METHOD:-eunsu_8meta_action_classiccv_front_only}
export SUITE_METRICS_NAME=${SUITE_METRICS_NAME:-eunsu_8meta_classiccv_quant_metrics}
export USE_CLASSIC_CV=${USE_CLASSIC_CV:-1}
export META_TTC_THRESHOLD=${META_TTC_THRESHOLD:-${QWEN_TTC_THRESHOLD:-3.0}}
export QWEN_TTC_THRESHOLD=${QWEN_TTC_THRESHOLD:-${META_TTC_THRESHOLD}}
export META_EVERY_N_STEPS=${META_EVERY_N_STEPS:-${QWEN_8META_EVERY_N_STEPS:-20}}
export QWEN_8META_EVERY_N_STEPS=${QWEN_8META_EVERY_N_STEPS:-${META_EVERY_N_STEPS}}
export ENH_VIS_MAX_ROUTES=${ENH_VIS_MAX_ROUTES:-10}
export QWEN_DASHBOARD_REAR=0
export QWEN_EMERGENCY_PULL_OVER=0
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=0
export QWEN_SAVE_DASHBOARD=${QWEN_SAVE_DASHBOARD:-0}
export QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}
export QWEN_BENCHMARK_INFER=${QWEN_BENCHMARK_INFER:-1}
export QWEN_8META_EVERY_N_STEPS=${QWEN_8META_EVERY_N_STEPS:-20}

# Crash restart only. Do not rerun legitimate low-score routes in the benchmark.
export MAX_RETRIES=${MAX_RETRIES:-999}
export RESTART_WAIT=${RESTART_WAIT:-30}
export RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE:-1}
export RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD:--1}
export CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-1}
export SUITE_FRESH=${SUITE_FRESH:-1}

export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment : Eunsu 8meta-action + ClassicCV + dev10 quant suite"
echo "Routes     : ${ROUTES}"
echo "Runs       : ${RUNS}"
echo "Suite root : ${SUITE_ROOT}"
echo "Enhancer   : QWEN_8META_IMAGE_ENHANCER=${QWEN_8META_IMAGE_ENHANCER}"
echo "Agent      : ${TEAM_AGENT}"
echo "CUDA vis   : ${CUDA_VISIBLE_DEVICES_LIST}; TF++ gpu-rank=${GPU_RANK}; vLLM GPU=${VLLM_CUDA_VISIBLE_DEVICES}"
echo "Ports      : CARLA=${BASE_PORT}, TM=${BASE_TM_PORT}, vLLM base=${VLLM_PORT_BASE}"
echo "Retry      : MAX_RETRIES=${MAX_RETRIES}, RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE}, LOW_SCORE=${RETRY_LOW_SCORE_THRESHOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec "${SCRIPT_DIR}/run_qwen_8meta_action_quant_suite.sh" "$@"
