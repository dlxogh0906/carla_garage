#!/bin/bash
# Run Qwen3-VL-8B dev10 quantization suite sequentially and summarize metrics.
#
# Default runs:
#   1. BF16 transformers
#   2. BF16 vLLM
#   3. W8A8-INT8 vLLM
#   4. AWQ W4A16 n64 vLLM
#   5. GPTQ W4A16 n64 vLLM

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${LEADERBOARD_PYTHON:-/home/kwy00/anaconda3/envs/garage_2/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/kwy00/anaconda3/envs/qwen_quant/bin/python}

ROUTES=${ROUTES:-${SCRIPT_DIR}/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml}
SUITE_ROOT=${SUITE_ROOT:-/mnt/2/carla_metric_result/qwen_dev10_quant_suite_$(date +%Y%m%d_%H%M%S)}
SUITE_LOCK=${SUITE_LOCK:-/tmp/qwen_dev10_quant_suite.lock}
RUNS=${RUNS:-bf16_transformers,bf16_vllm,w8a8_int8_vllm,awq_w4a16_n64_vllm,gptq_w4a16_n64_vllm}

CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-0,1}
GPU_RANK=${GPU_RANK:-0}
TRANSFORMER_QWEN_DEVICE=${TRANSFORMER_QWEN_DEVICE:-cuda:1}
VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
VLLM_PHYSICAL_GPU_INDEX=${VLLM_PHYSICAL_GPU_INDEX:-${VLLM_CUDA_VISIBLE_DEVICES%%,*}}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT_BASE=${VLLM_PORT_BASE:-8010}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-1073741824}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-900}

BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}
MAX_RETRIES=${MAX_RETRIES:-30}
RESTART_WAIT=${RESTART_WAIT:-30}
SUITE_FRESH=${SUITE_FRESH:-1}
CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-1}
QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}
QWEN_BENCHMARK_INFER=${QWEN_BENCHMARK_INFER:-1}
QWEN_MAX_NEW_TOKENS=${QWEN_MAX_NEW_TOKENS:-220}
QWEN_DASHBOARD_REAR=${QWEN_DASHBOARD_REAR:-0}
QWEN_EMERGENCY_PULL_OVER=${QWEN_EMERGENCY_PULL_OVER:-0}
QWEN_EMERGENCY_REAR_PROBE_STEPS=${QWEN_EMERGENCY_REAR_PROBE_STEPS:-999999}
QWEN_FORCE_EMERGENCY_LANE_CHANGE=${QWEN_FORCE_EMERGENCY_LANE_CHANGE:-0}

MODEL_BF16=${MODEL_BF16:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}
MODEL_W8A8=${MODEL_W8A8:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012}
MODEL_AWQ=${MODEL_AWQ:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n64}
MODEL_GPTQ=${MODEL_GPTQ:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16-n64}

VLLM_PID=""
CURRENT_VLLM_PORT=""

exec 9>"$SUITE_LOCK"
if ! flock -n 9; then
  echo "[lock] Another qwen dev10 quant suite is already running. Lock: ${SUITE_LOCK}" >&2
  echo "[lock] Stop the existing run first, or set SUITE_LOCK=/tmp/another.lock if you intentionally need parallel runs on different ports." >&2
  exit 2
fi

mkdir -p "$SUITE_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

contains_run() {
  local needle="$1"
  local item
  IFS=',' read -ra items <<< "$RUNS"
  for item in "${items[@]}"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
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
      log "[vLLM] server process exited before becoming ready"
      log "[vLLM] log: ${server_log}"
      tail -120 "$server_log" 2>/dev/null || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
      log "[vLLM] waiting... ${waited}s"
    fi
  done
  log "[vLLM] server did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
  log "[vLLM] log: ${server_log}"
  tail -80 "$server_log" 2>/dev/null || true
  return 1
}

stop_vllm() {
  if [ -n "$VLLM_PID" ]; then
    log "[vLLM] stopping server pid=${VLLM_PID}"
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
  pkill -9 -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
}

cleanup_all() {
  stop_vllm
}
trap cleanup_all EXIT

cleanup_carla() {
  if [ "$CLEAN_BEFORE_RUN" = "1" ]; then
    log "[cleanup] CARLA/leaderboard 잔존 프로세스 정리"
    pkill -9 -f "[C]arlaUE4" 2>/dev/null || true
    pkill -9 -f "[l]eaderboard_evaluator" 2>/dev/null || true
    pkill -9 -f "[s]cenario_manager" 2>/dev/null || true
    sleep 5
  fi
}

prepare_out_dir() {
  local out_dir="$1"
  if [ "$SUITE_FRESH" = "1" ] && [ -e "$out_dir" ]; then
    local backup="${out_dir}.bak_$(date +%Y%m%d_%H%M%S)"
    log "[fresh] moving existing OUT_DIR to ${backup}"
    mv "$out_dir" "$backup"
  fi
  mkdir -p "$out_dir"
}

write_metadata() {
  local out_dir="$1"
  local run_id="$2"
  local quant_label="$3"
  local backend="$4"
  local model_path="$5"
  "$PYTHON_BIN" - "$out_dir/run_metadata.json" "$run_id" "$quant_label" "$backend" "$model_path" "$ROUTES" <<'PY'
import json
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
    "image_enhancer": __import__("os").environ.get("QWEN_IMAGE_ENHANCER", "off") or "off",
    "image_enhance_targets": __import__("os").environ.get("QWEN_IMAGE_ENHANCE_TARGETS", "rgb_front"),
    "front_only": {
        "dashboard_rear": __import__("os").environ.get("QWEN_DASHBOARD_REAR", ""),
        "emergency_pull_over": __import__("os").environ.get("QWEN_EMERGENCY_PULL_OVER", ""),
        "emergency_rear_probe_steps": __import__("os").environ.get("QWEN_EMERGENCY_REAR_PROBE_STEPS", ""),
        "force_emergency_lane_change": __import__("os").environ.get("QWEN_FORCE_EMERGENCY_LANE_CHANGE", ""),
    },
}, indent=2) + "\n")
PY
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local quantization="$3"
  local port="$4"
  local out_dir="$5"
  local server_log="${out_dir}/vllm_server.log"
  local endpoint="http://${VLLM_HOST}:${port}/v1/chat/completions"
  local models_endpoint="http://${VLLM_HOST}:${port}/v1/models"

  if [ ! -d "$model_path" ]; then
    echo "[config error] VLLM model not found: ${model_path}" >&2
    exit 2
  fi

  stop_vllm
  if server_ready "$models_endpoint"; then
    echo "[config error] vLLM port ${port} is already serving something at ${models_endpoint}" >&2
    echo "Use another VLLM_PORT_BASE or stop the stale server first." >&2
    exit 2
  fi

  local quant_args=()
  if [ -n "$quantization" ] && [ "$quantization" != "none" ]; then
    quant_args=(--quantization "$quantization")
  fi

  local baseline_gpu_mem_gib
  baseline_gpu_mem_gib=$(gpu_mem_gib "$VLLM_PHYSICAL_GPU_INDEX")

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Starting vLLM server"
  log "Model      : ${model_path}"
  log "Served name: ${served_name}"
  log "Quant      : ${quantization}"
  log "Endpoint   : ${endpoint}"
  log "GPU        : physical ${VLLM_PHYSICAL_GPU_INDEX} (CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES})"
  log "Log        : ${server_log}"
  log "Base mem   : ${baseline_gpu_mem_gib} GiB"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
  setsid \
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --host "$VLLM_HOST" \
    --port "$port" \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --dtype "$VLLM_DTYPE" \
    "${quant_args[@]}" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt '{"image": 1}' \
    --disable-log-requests \
    >"$server_log" 2>&1 &
  VLLM_PID=$!
  CURRENT_VLLM_PORT="$port"
  wait_for_server "$models_endpoint" "$server_log"

  local after_load_gpu_mem_gib
  after_load_gpu_mem_gib=$(gpu_mem_gib "$VLLM_PHYSICAL_GPU_INDEX")
  log "[vLLM] ready. after-load mem=${after_load_gpu_mem_gib} GiB"

  export QWEN_VLM_BACKEND=vllm_openai
  export QWEN_VLLM_ENDPOINT="$endpoint"
  export QWEN_VLLM_MODEL_NAME="$served_name"
  export QWEN_VLLM_GPU_INDEX="$VLLM_PHYSICAL_GPU_INDEX"
  export QWEN_VLLM_BASELINE_GPU_MEM_GIB="$baseline_gpu_mem_gib"
  export QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB="$after_load_gpu_mem_gib"
  export QWEN_VLM_DEVICE=remote_vllm
}

summarize_one() {
  local out_dir="$1"
  local quant_label="$2"
  "$PYTHON_BIN" "${SCRIPT_DIR}/tools/summarize_qwen_runtime.py" \
    --input "${out_dir}/viz" \
    --output-prefix "${out_dir}/qwen_runtime" \
    --model-name "Qwen3-VL-8B" \
    --quant "$quant_label"
}

run_dev10() {
  local run_id="$1"
  local quant_label="$2"
  local backend="$3"
  local model_path="$4"
  local prepared="${5:-0}"
  local out_dir="${SUITE_ROOT}/${run_id}"

  if [ "$prepared" != "1" ]; then
    prepare_out_dir "$out_dir"
    write_metadata "$out_dir" "$run_id" "$quant_label" "$backend" "$model_path"
    cleanup_carla
  fi

  export OUT_DIR="$out_dir"
  export ROUTES
  export BASE_PORT
  export BASE_TM_PORT
  export CUDA_VISIBLE_DEVICES_LIST
  export GPU_RANK
  export MAX_RETRIES
  export RESTART_WAIT
  export QWEN_SKIP_VIDEO
  export QWEN_BENCHMARK_INFER
  export QWEN_MAX_NEW_TOKENS
  export QWEN_DASHBOARD_REAR
  export QWEN_EMERGENCY_PULL_OVER
  export QWEN_EMERGENCY_REAR_PROBE_STEPS
  export QWEN_FORCE_EMERGENCY_LANE_CHANGE
  export QWEN_MODEL="$model_path"
  export QWEN_MODEL_LABEL="Qwen3-VL-8B"
  export QWEN_QUANT="$quant_label"
  export QWEN_VLM_DTYPE="$VLLM_DTYPE"

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Run        : ${run_id}"
  log "Quant      : ${quant_label}"
  log "Backend    : ${backend}"
  log "Model      : ${model_path}"
  log "ROUTES     : ${ROUTES}"
  log "OUT_DIR    : ${out_dir}"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  bash "${SCRIPT_DIR}/../eval/run_qwen_dev10.sh"
  summarize_one "$out_dir" "$quant_label"
}

run_transformers_bf16() {
  local run_id="bf16_transformers"
  contains_run "$run_id" || return 0

  unset QWEN_VLM_BACKEND QWEN_VLLM_ENDPOINT QWEN_VLLM_MODEL_NAME
  unset QWEN_VLLM_GPU_INDEX QWEN_VLLM_BASELINE_GPU_MEM_GIB QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB
  export QWEN_RUNTIME_QUANT=none
  export QWEN_VLM_DEVICE="$TRANSFORMER_QWEN_DEVICE"

  run_dev10 "$run_id" "BF16-transformers" "transformers" "$MODEL_BF16"
}

run_vllm_case() {
  local run_id="$1"
  local quant_label="$2"
  local model_path="$3"
  local served_name="$4"
  local quantization="$5"
  local runtime_quant="$6"
  local port="$7"

  contains_run "$run_id" || return 0

  export QWEN_RUNTIME_QUANT="$runtime_quant"
  local out_dir="${SUITE_ROOT}/${run_id}"
  prepare_out_dir "$out_dir"
  write_metadata "$out_dir" "$run_id" "$quant_label" "vllm" "$model_path"
  cleanup_carla
  start_vllm "$model_path" "$served_name" "$quantization" "$port" "$out_dir"

  run_dev10 "$run_id" "$quant_label" "vllm" "$model_path" "1"
  stop_vllm
}

log "Suite root : ${SUITE_ROOT}"
log "Runs       : ${RUNS}"
log "Routes     : ${ROUTES}"
log "CUDA vis   : ${CUDA_VISIBLE_DEVICES_LIST}; TF++ cuda:${GPU_RANK}; transformer Qwen=${TRANSFORMER_QWEN_DEVICE}; vLLM physical GPU=${VLLM_PHYSICAL_GPU_INDEX}"

run_transformers_bf16
run_vllm_case \
  "bf16_vllm" \
  "BF16-vLLM" \
  "$MODEL_BF16" \
  "Qwen3-VL-8B-BF16-vLLM" \
  "none" \
  "vllm-bf16" \
  "$((VLLM_PORT_BASE + 0))"

run_vllm_case \
  "w8a8_int8_vllm" \
  "W8A8-INT8-vLLM" \
  "$MODEL_W8A8" \
  "Qwen3-VL-8B-W8A8-INT8-vLLM" \
  "compressed-tensors" \
  "vllm-w8a8" \
  "$((VLLM_PORT_BASE + 2))"

run_vllm_case \
  "awq_w4a16_n64_vllm" \
  "AWQ-W4A16-n64-vLLM" \
  "$MODEL_AWQ" \
  "Qwen3-VL-8B-AWQ-W4A16-n64-vLLM" \
  "compressed-tensors" \
  "vllm-awq" \
  "$((VLLM_PORT_BASE + 4))"

run_vllm_case \
  "gptq_w4a16_n64_vllm" \
  "GPTQ-W4A16-n64-vLLM" \
  "$MODEL_GPTQ" \
  "Qwen3-VL-8B-GPTQ-W4A16-n64-vLLM" \
  "compressed-tensors" \
  "vllm-gptq" \
  "$((VLLM_PORT_BASE + 6))"

"$PYTHON_BIN" "${SCRIPT_DIR}/tools/summarize_qwen_dev10_quant_suite.py" \
  --suite-root "$SUITE_ROOT" \
  --output-prefix "${SUITE_ROOT}/qwen_dev10_quant_metrics"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Done"
log "Metrics CSV : ${SUITE_ROOT}/qwen_dev10_quant_metrics.csv"
log "Metrics MD  : ${SUITE_ROOT}/qwen_dev10_quant_metrics.md"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
