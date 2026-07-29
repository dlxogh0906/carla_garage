#!/bin/bash
# Quantize Qwen3-VL-30B-A3B-Instruct into W8A8, AWQ, and GPTQ checkpoints.
#
# Examples:
#   ./run_quantize_qwen3vl_30b_a3b.sh                  # AWQ only by default
#   RUNS=w8a8 ./run_quantize_qwen3vl_30b_a3b.sh
#   RUNS=gptq NUM_SAMPLES=128 CUDA_VISIBLE_DEVICES=1 ./run_quantize_qwen3vl_30b_a3b.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/home/kwy00/anaconda3/envs/qwen_quant/bin/python}

MODEL=${MODEL:-/mnt/2/pretrained_models/Qwen3-VL-30B-A3B-Instruct}
CALIB_JSONL=${CALIB_JSONL:-/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl}
OUT_ROOT=${OUT_ROOT:-/mnt/2/pretrained_models}

OUT_W8A8=${OUT_W8A8:-${OUT_ROOT}/Qwen3-VL-30B-A3B-Instruct-W8A8-INT8}
OUT_AWQ=${OUT_AWQ:-${OUT_ROOT}/Qwen3-VL-30B-A3B-Instruct-AWQ-W4A16-n64}
OUT_GPTQ=${OUT_GPTQ:-${OUT_ROOT}/Qwen3-VL-30B-A3B-Instruct-GPTQ-W4A16-n64}

RUNS=${RUNS:-awq}
NUM_SAMPLES=${NUM_SAMPLES:-256}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}
MAX_MEMORY=${MAX_MEMORY:-}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-/mnt/2/qwen30b_quant_offload}
OVERWRITE=${OVERWRITE:-0}
NO_SAVE=${NO_SAVE:-0}

GPTQ_BLOCK_SIZE=${GPTQ_BLOCK_SIZE:-128}
GPTQ_DAMPENING_FRAC=${GPTQ_DAMPENING_FRAC:-0.01}
GPTQ_ACTORDER=${GPTQ_ACTORDER:-static}
GPTQ_OFFLOAD_HESSIANS=${GPTQ_OFFLOAD_HESSIANS:-1}

AWQ_DUO_SCALING=${AWQ_DUO_SCALING:-true}
AWQ_N_GRID=${AWQ_N_GRID:-20}

contains_run() {
  local needle="$1" item
  IFS=',' read -ra items <<< "$RUNS"
  for item in "${items[@]}"; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

common_args=(
  --model "$MODEL"
  --calib-jsonl "$CALIB_JSONL"
  --num-samples "$NUM_SAMPLES"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
)

if [ "$OVERWRITE" = "1" ]; then
  common_args+=(--overwrite)
fi
if [ "$NO_SAVE" = "1" ]; then
  common_args+=(--no-save)
fi
if [ -n "$MAX_MEMORY" ]; then
  common_args+=(--max-memory "$MAX_MEMORY")
  common_args+=(--offload-folder "$OFFLOAD_FOLDER")
  mkdir -p "$OFFLOAD_FOLDER"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Qwen3-VL-30B-A3B quantization suite"
echo "model      : $MODEL"
echo "calib      : $CALIB_JSONL"
echo "runs       : $RUNS"
echo "samples    : $NUM_SAMPLES"
echo "device_map : $DEVICE_MAP"
echo "dtype      : $DTYPE"
echo "max_memory : ${MAX_MEMORY:-default}"
echo "offload    : ${MAX_MEMORY:+$OFFLOAD_FOLDER}"
echo "CUDA vis   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "outputs    :"
echo "  W8A8: $OUT_W8A8"
echo "  AWQ : $OUT_AWQ"
echo "  GPTQ: $OUT_GPTQ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if contains_run w8a8 || contains_run int8; then
  "$PYTHON_BIN" "${SCRIPT_DIR}/tools/quantize_qwen3vl_w8a8.py" \
    "${common_args[@]}" \
    --output-dir "$OUT_W8A8"
fi

if contains_run awq; then
  "$PYTHON_BIN" "${SCRIPT_DIR}/tools/quantize_qwen3vl_w4a16.py" \
    --method awq \
    "${common_args[@]}" \
    --output-dir "$OUT_AWQ" \
    --awq-duo-scaling "$AWQ_DUO_SCALING" \
    --awq-n-grid "$AWQ_N_GRID"
fi

if contains_run gptq; then
  gptq_args=()
  if [ "$GPTQ_OFFLOAD_HESSIANS" = "1" ]; then
    gptq_args+=(--offload-hessians)
  fi
  "$PYTHON_BIN" "${SCRIPT_DIR}/tools/quantize_qwen3vl_w4a16.py" \
    --method gptq \
    "${common_args[@]}" \
    --output-dir "$OUT_GPTQ" \
    --block-size "$GPTQ_BLOCK_SIZE" \
    --dampening-frac "$GPTQ_DAMPENING_FRAC" \
    --actorder "$GPTQ_ACTORDER" \
    "${gptq_args[@]}"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
