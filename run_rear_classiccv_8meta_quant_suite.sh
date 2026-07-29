#!/bin/bash
# Run TF++ + rear ClassicCV meta-action VLA on dev10 for three table rows:
#   1. Raw Qwen3-VL, transformers
#   2. VQA-LoRA merged checkpoint, transformers
#   3. GPTQ-W4A16 checkpoint, vLLM

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEADERBOARD_PYTHON=${LEADERBOARD_PYTHON:-/home/kwy00/anaconda3/envs/garage_2/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/kwy00/anaconda3/envs/qwen_quant/bin/python}

export CARLA_ROOT=${CARLA_ROOT:-/mnt/2/carla}
export WORK_DIR=${WORK_DIR:-${SCRIPT_DIR}/Bench2Drive}
export SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT:-${WORK_DIR}/scenario_runner}
export LEADERBOARD_ROOT=${LEADERBOARD_ROOT:-${WORK_DIR}/leaderboard}
export PYTHONPATH="${SCRIPT_DIR}/team_code:${SCRIPT_DIR}/Bench2Drive:${SCRIPT_DIR}:${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}
TEAM_AGENT=${TEAM_AGENT:-${SCRIPT_DIR}/team_code/sensor_agent_meta_action_rear_classic_cv.py}
TEAM_CONFIG=${TEAM_CONFIG:-/mnt/2/pretrained_models/all_towns}
RESULT_ROOT=${RESULT_ROOT:-/mnt/2/carla_metric_result2}
SUITE_ROOT=${SUITE_ROOT:-${RESULT_ROOT}/rear_classiccv_8meta_quant_suite_$(date +%Y%m%d_%H%M%S)}
RUNS=${RUNS:-raw_transformers,lora_transformers,gptq_w4a16_vllm}
SUITE_LOCK=${SUITE_LOCK:-/tmp/rear_classiccv_8meta_quant_suite.lock}

MODEL_RAW=${MODEL_RAW:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}
MODEL_LORA=${MODEL_LORA:-/mnt/2/pretrained_models/Qwen3-VL-8B-SimLingo-ckpt30000-merged}
MODEL_GPTQ=${MODEL_GPTQ:-/mnt/2/pretrained_models/Qwen3-VL-8B-SimLingo-ckpt30000-GPTQ-W4A16-n64-s64}

CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-0,1}
GPU_RANK=${GPU_RANK:-0}
TRANSFORMER_META_DEVICE=${TRANSFORMER_META_DEVICE:-cuda:1}
VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
VLLM_PHYSICAL_GPU_INDEX=${VLLM_PHYSICAL_GPU_INDEX:-${VLLM_CUDA_VISIBLE_DEVICES%%,*}}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8024}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}
VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-1073741824}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-900}

BASE_PORT=${BASE_PORT:-30036}
BASE_TM_PORT=${BASE_TM_PORT:-50036}
MAX_RETRIES=${MAX_RETRIES:-30}
RESTART_WAIT=${RESTART_WAIT:-30}
CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-1}
SUITE_FRESH=${SUITE_FRESH:-1}
QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}
SAVE_META_DASHBOARD=${SAVE_META_DASHBOARD:-1}
LEADERBOARD_DEBUG=${LEADERBOARD_DEBUG:-1}

export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-1}
export IS_BENCH2DRIVE=True
export CARLA_QUALITY_LEVEL=${CARLA_QUALITY_LEVEL:-Epic}
export USE_CLASSIC_CV=${USE_CLASSIC_CV:-1}
export META_TTC_THRESHOLD=${META_TTC_THRESHOLD:-3.0}
export META_EVERY_N_STEPS=${META_EVERY_N_STEPS:-20}
export QWEN_BENCHMARK_INFER=${QWEN_BENCHMARK_INFER:-1}
export QWEN_MAX_NEW_TOKENS=${QWEN_MAX_NEW_TOKENS:-10}
export QWEN_VLM_DTYPE=${QWEN_VLM_DTYPE:-$VLLM_DTYPE}

VLLM_PID=""
CURRENT_VLLM_PORT=""

exec 9>"$SUITE_LOCK"
if ! flock -n 9; then
  echo "[lock] another rear ClassicCV 8meta suite is running: ${SUITE_LOCK}" >&2
  exit 2
fi

mkdir -p "$SUITE_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

contains_run() {
  local needle="$1" item
  IFS=',' read -ra items <<< "$RUNS"
  for item in "${items[@]}"; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

gpu_mem_gib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'NR==1 { printf "%.6f", $1 / 1024.0 }'
}

server_ready() {
  local url="$1"
  "$VLLM_PYTHON" - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as resp:
    raise SystemExit(0 if resp.status == 200 else 1)
PY
}

wait_for_server() {
  local models_endpoint="$1"
  local server_log="$2"
  local waited=0
  while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
    if server_ready "$models_endpoint"; then
      return 0
    fi
    if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log "[vLLM] server exited before ready"
      tail -120 "$server_log" 2>/dev/null || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    [ $((waited % 30)) -eq 0 ] && log "[vLLM] waiting... ${waited}s"
  done
  log "[vLLM] timeout after ${VLLM_STARTUP_TIMEOUT}s"
  tail -120 "$server_log" 2>/dev/null || true
  return 1
}

stop_vllm() {
  if [ -n "$VLLM_PID" ]; then
    log "[vLLM] stopping pid=${VLLM_PID}"
    kill -- "-${VLLM_PID}" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
  if [ -n "$CURRENT_VLLM_PORT" ]; then
    pkill -f "[v]llm.entrypoints.openai.api_server.*--port ${CURRENT_VLLM_PORT}" 2>/dev/null || true
    CURRENT_VLLM_PORT=""
  fi
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 3
}

cleanup_all() {
  stop_vllm
}
trap cleanup_all EXIT

cleanup_carla() {
  [ "$CLEAN_BEFORE_RUN" = "1" ] || return 0
  log "[cleanup] CARLA/leaderboard"
  pkill -9 -f "[C]arlaUE4" 2>/dev/null || true
  pkill -9 -f "[l]eaderboard_evaluator" 2>/dev/null || true
  pkill -9 -f "[s]cenario_manager" 2>/dev/null || true
  sleep 5
}

prepare_out_dir() {
  local out_dir="$1"
  if [ "$SUITE_FRESH" = "1" ] && [ -e "$out_dir" ]; then
    mv "$out_dir" "${out_dir}.bak_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$out_dir/viz"
}

write_metadata() {
  local out_dir="$1"
  local run_id="$2"
  local quant_label="$3"
  local backend="$4"
  local model_path="$5"
  "$LEADERBOARD_PYTHON" - "$out_dir/run_metadata.json" "$run_id" "$quant_label" "$backend" "$model_path" "$ROUTES" <<'PY'
import json
import os
import sys
from pathlib import Path
path, run_id, quant_label, backend, model_path, routes = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "model_label": "Qwen3-VL-8B",
    "quant_label": quant_label,
    "backend": backend,
    "model_path": model_path,
    "routes": routes,
    "method": "TF++ + Meta-Action VLA + ClassicCV(front+rear)",
    "classic_cv": os.environ.get("USE_CLASSIC_CV", "1"),
    "ttc_threshold": os.environ.get("META_TTC_THRESHOLD", "3.0"),
    "every_n_steps": os.environ.get("META_EVERY_N_STEPS", "20"),
}, indent=2) + "\n")
PY
}

summarize_one() {
  local out_dir="$1"
  local quant_label="$2"
  "$LEADERBOARD_PYTHON" "${SCRIPT_DIR}/tools/summarize_qwen_runtime.py" \
    --input "${out_dir}/viz" \
    --output-prefix "${out_dir}/qwen_runtime" \
    --model-name "Qwen3-VL-8B" \
    --quant "$quant_label"
}

run_eval() {
  local out_dir="$1"
  local log_file="${out_dir}/eval.log"
  local checkpoint="${out_dir}/eval.json"
  local attempt=0

  export SAVE_PATH="${out_dir}/viz"
  if [ "$SAVE_META_DASHBOARD" = "1" ]; then
    export META_DASHBOARD_PATH="${out_dir}/meta_dashboard"
    mkdir -p "$META_DASHBOARD_PATH"
  else
    unset META_DASHBOARD_PATH
  fi
  mkdir -p "$SAVE_PATH"
  touch "$log_file"

  while [ $attempt -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    log "[run] attempt ${attempt}/${MAX_RETRIES}: ${out_dir}"
    set +e
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST} "$LEADERBOARD_PYTHON" \
      "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
      --routes="${ROUTES}" \
      --repetitions=1 \
      --track=SENSORS \
      --checkpoint="${checkpoint}" \
      --agent="${TEAM_AGENT}" \
      --agent-config="${TEAM_CONFIG}" \
      --debug="${LEADERBOARD_DEBUG}" \
      --resume=True \
      --port="${BASE_PORT}" \
      --traffic-manager-port="${BASE_TM_PORT}" \
      --gpu-rank="${GPU_RANK}" \
      2>&1 | tee -a "$log_file"
    exit_code=${PIPESTATUS[0]}
    summarize_one "$out_dir" "${QWEN_QUANT:-unknown}" >>"$log_file" 2>&1 || true
    set -e
    [ $exit_code -eq 0 ] && return 0
    [ $attempt -ge "$MAX_RETRIES" ] && return "$exit_code"
    log "[run] crashed exit=${exit_code}; restart after ${RESTART_WAIT}s"
    cleanup_carla
    sleep "$RESTART_WAIT"
  done
}

run_transformers_case() {
  local run_id="$1"
  local quant_label="$2"
  local runtime_quant="$3"
  local model_path="$4"
  contains_run "$run_id" || return 0
  local out_dir="${SUITE_ROOT}/${run_id}"

  prepare_out_dir "$out_dir"
  write_metadata "$out_dir" "$run_id" "$quant_label" "transformers" "$model_path"
  cleanup_carla

  export META_MODEL="$model_path"
  export META_DEVICE="$TRANSFORMER_META_DEVICE"
  export QWEN_VLM_BACKEND=transformers
  export QWEN_QUANT="$quant_label"
  export QWEN_RUNTIME_QUANT="$runtime_quant"
  unset QWEN_VLLM_ENDPOINT QWEN_VLLM_MODEL_NAME QWEN_VLLM_GPU_INDEX
  unset QWEN_VLLM_BASELINE_GPU_MEM_GIB QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Run      : ${run_id}"
  log "Method   : rear ClassicCV 8meta"
  log "Backend  : transformers"
  log "Model    : ${model_path}"
  log "OUT_DIR  : ${out_dir}"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  run_eval "$out_dir"
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local quantization="$3"
  local out_dir="$4"
  local server_log="${out_dir}/vllm_server.log"
  local endpoint="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
  local models_endpoint="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"

  stop_vllm
  if server_ready "$models_endpoint"; then
    echo "[config error] vLLM port already in use: ${VLLM_PORT}" >&2
    exit 2
  fi

  local baseline_gpu_mem_gib
  baseline_gpu_mem_gib=$(gpu_mem_gib "$VLLM_PHYSICAL_GPU_INDEX")

  log "[vLLM] starting ${model_path}"
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
  setsid \
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --dtype "$VLLM_DTYPE" \
    --quantization "$quantization" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt '{"image": 2}' \
    --disable-log-requests \
    >"$server_log" 2>&1 &
  VLLM_PID=$!
  CURRENT_VLLM_PORT="$VLLM_PORT"
  wait_for_server "$models_endpoint" "$server_log"

  local after_load_gpu_mem_gib
  after_load_gpu_mem_gib=$(gpu_mem_gib "$VLLM_PHYSICAL_GPU_INDEX")

  export QWEN_VLM_BACKEND=vllm_openai
  export QWEN_VLLM_ENDPOINT="$endpoint"
  export QWEN_VLLM_MODEL_NAME="$served_name"
  export QWEN_VLLM_GPU_INDEX="$VLLM_PHYSICAL_GPU_INDEX"
  export QWEN_VLLM_BASELINE_GPU_MEM_GIB="$baseline_gpu_mem_gib"
  export QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB="$after_load_gpu_mem_gib"
  export META_DEVICE=remote_vllm
}

run_vllm_case() {
  local run_id="$1"
  local quant_label="$2"
  local runtime_quant="$3"
  local model_path="$4"
  local served_name="$5"
  local quantization="$6"
  contains_run "$run_id" || return 0
  local out_dir="${SUITE_ROOT}/${run_id}"

  prepare_out_dir "$out_dir"
  write_metadata "$out_dir" "$run_id" "$quant_label" "vllm" "$model_path"
  cleanup_carla

  export META_MODEL="$model_path"
  export QWEN_QUANT="$quant_label"
  export QWEN_RUNTIME_QUANT="$runtime_quant"
  start_vllm "$model_path" "$served_name" "$quantization" "$out_dir"

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Run      : ${run_id}"
  log "Method   : rear ClassicCV 8meta"
  log "Backend  : vLLM"
  log "Model    : ${model_path}"
  log "OUT_DIR  : ${out_dir}"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  run_eval "$out_dir"
  stop_vllm
}

log "Suite root : ${SUITE_ROOT}"
log "Runs       : ${RUNS}"
log "Routes     : ${ROUTES}"
log "Agent      : ${TEAM_AGENT}"

run_transformers_case "raw_transformers" "Raw-Qwen3VL-transformers" "bf16" "$MODEL_RAW"
run_transformers_case "lora_transformers" "VQA-LoRA-transformers" "bf16" "$MODEL_LORA"
run_vllm_case \
  "gptq_w4a16_vllm" \
  "GPTQ-W4A16-vLLM" \
  "gptq-w4a16" \
  "$MODEL_GPTQ" \
  "Qwen3-VL-8B-SimLingo-GPTQ-W4A16" \
  "compressed-tensors"

"$LEADERBOARD_PYTHON" "${SCRIPT_DIR}/tools/summarize_qwen_dev10_quant_suite.py" \
  --suite-root "$SUITE_ROOT" \
  --output-prefix "${SUITE_ROOT}/rear_classiccv_8meta_quant_metrics"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Done"
log "CSV: ${SUITE_ROOT}/rear_classiccv_8meta_quant_metrics.csv"
log "MD : ${SUITE_ROOT}/rear_classiccv_8meta_quant_metrics.md"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
