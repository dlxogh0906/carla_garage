#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-/home/kwy00/anaconda3/envs/qwen_quant/bin/python}

BASE_MODEL=${BASE_MODEL:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}
LORA_DIR=${LORA_DIR:-/mnt/2/pretrained_models/qwen3vl_vqa30000_stage2B_dreamerALL_vqa10_checkpoint-15000}

MERGED_MODEL=${MERGED_MODEL:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-vqa30000-stage2B-dreamerALL-vqa10-merged-bf16}
OUT_GPTQ=${OUT_GPTQ:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-vqa30000-stage2B-dreamerALL-vqa10-GPTQ-W4A16-n64}

CALIB_JSONL=${CALIB_JSONL:-/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-64}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GPTQ_BLOCK_SIZE=${GPTQ_BLOCK_SIZE:-128}
GPTQ_DAMPENING_FRAC=${GPTQ_DAMPENING_FRAC:-0.01}
GPTQ_ACTORDER=${GPTQ_ACTORDER:-static}
GPTQ_OFFLOAD_HESSIANS=${GPTQ_OFFLOAD_HESSIANS:-1}

SKIP_MERGE=${SKIP_MERGE:-0}
SKIP_GPTQ=${SKIP_GPTQ:-0}
OVERWRITE_MERGED=${OVERWRITE_MERGED:-0}
OVERWRITE_GPTQ=${OVERWRITE_GPTQ:-0}

need_dir() {
  if [ ! -d "$1" ]; then
    echo "[error] missing directory: $1" >&2
    exit 2
  fi
}

need_file() {
  if [ ! -f "$1" ]; then
    echo "[error] missing file: $1" >&2
    exit 2
  fi
}

dir_nonempty() {
  [ -d "$1" ] && find "$1" -mindepth 1 -maxdepth 1 | read -r _
}

need_dir "$BASE_MODEL"
need_dir "$LORA_DIR"
need_file "$LORA_DIR/adapter_model.safetensors"
need_file "$LORA_DIR/adapter_config.json"
need_file "$CALIB_JSONL"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Merge Qwen3-VL LoRA, then GPTQ W4A16 n64"
echo "Base      : $BASE_MODEL"
echo "LoRA      : $LORA_DIR"
echo "Merged    : $MERGED_MODEL"
echo "GPTQ out  : $OUT_GPTQ"
echo "Calib     : $CALIB_JSONL"
echo "Samples   : $NUM_SAMPLES"
echo "CUDA vis  : $CUDA_VISIBLE_DEVICES"
echo "Skip merge: $SKIP_MERGE"
echo "Skip GPTQ : $SKIP_GPTQ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$SKIP_MERGE" != "1" ]; then
  merge_args=()
  if [ "$OVERWRITE_MERGED" = "1" ]; then
    merge_args+=(--overwrite)
  elif dir_nonempty "$MERGED_MODEL"; then
    echo "[error] merged output exists and is not empty: $MERGED_MODEL" >&2
    echo "Set OVERWRITE_MERGED=1 only if you want to replace that output copy." >&2
    exit 2
  fi

  "$PYTHON_BIN" "$SCRIPT_DIR/tools/merge_qwen3vl_lora_safetensors.py" \
    --base-model "$BASE_MODEL" \
    --lora-dir "$LORA_DIR" \
    --output-dir "$MERGED_MODEL" \
    "${merge_args[@]}"
else
  need_dir "$MERGED_MODEL"
fi

if [ "$SKIP_GPTQ" = "1" ]; then
  echo "[done] merge complete; GPTQ skipped."
  exit 0
fi

if [ "$OVERWRITE_GPTQ" != "1" ] && dir_nonempty "$OUT_GPTQ"; then
  echo "[error] GPTQ output exists and is not empty: $OUT_GPTQ" >&2
  echo "Set OVERWRITE_GPTQ=1 only if you want to replace that output copy." >&2
  exit 2
fi

gptq_args=()
if [ "$GPTQ_OFFLOAD_HESSIANS" = "1" ]; then
  gptq_args+=(--offload-hessians)
fi
if [ "$OVERWRITE_GPTQ" = "1" ]; then
  gptq_args+=(--overwrite)
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$PYTHON_BIN" "$SCRIPT_DIR/tools/quantize_qwen3vl_w4a16.py" \
  --method gptq \
  --model "$MERGED_MODEL" \
  --calib-jsonl "$CALIB_JSONL" \
  --output-dir "$OUT_GPTQ" \
  --num-samples "$NUM_SAMPLES" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --block-size "$GPTQ_BLOCK_SIZE" \
  --dampening-frac "$GPTQ_DAMPENING_FRAC" \
  --actorder "$GPTQ_ACTORDER" \
  "${gptq_args[@]}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done"
echo "Merged : $MERGED_MODEL"
echo "GPTQ   : $OUT_GPTQ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
