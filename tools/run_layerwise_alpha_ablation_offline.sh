#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kwy00/anaconda3/envs/garage_2/bin/python}"
BASE_MODEL="${BASE_MODEL:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}"
LORA_DIR="${LORA_DIR:-/mnt/2/pretrained_models/qwen3vl_simlingo_checkpoint-30000_nonquant}"
VAL_JSONL="${VAL_JSONL:-/mnt/2/carla_metric_result2/layerwise_lora_analysis/front_classiccv_pseudoval_vlmonly.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/2/pretrained_models}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/2/carla_metric_result2/layerwise_lora_analysis/offline_ablation}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"
LIMIT="${LIMIT:-}"
DELETE_MERGED="${DELETE_MERGED:-0}"

mkdir -p "${RESULT_ROOT}"

configs=(
  "naive:early=1.0,middle=1.0,late=1.0,other=1.0"
  "base_preserve:early=0.25,middle=0.75,late=1.0,other=1.0"
  "conservative:early=0.5,middle=0.75,late=1.0,other=1.0"
  "driving_focused:early=0.5,middle=1.0,late=1.25,other=1.0"
  "strong_adapt:early=0.75,middle=1.0,late=1.25,other=1.0"
  "late_only:early=0.0,middle=0.0,late=1.0,other=1.0"
  "early_off:early=0.0,middle=1.0,late=1.0,other=1.0"
)

for item in "${configs[@]}"; do
  name="${item%%:*}"
  alphas="${item#*:}"
  model_dir="${MODEL_ROOT}/Qwen3-VL-8B-SimLingo-ckpt30000-layerwise-${name}"
  result_dir="${RESULT_ROOT}/${name}"
  mkdir -p "${result_dir}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "[ablation] ${name} ${alphas}"
  echo "model_dir=${model_dir}"
  echo "result_dir=${result_dir}"

  if [[ ! -d "${model_dir}" || ! -f "${model_dir}/merge_lora_layerwise_metadata.json" ]]; then
    "${PYTHON_BIN}" tools/merge_qwen3vl_lora_layerwise.py \
      --base-model "${BASE_MODEL}" \
      --lora-dir "${LORA_DIR}" \
      --output-dir "${model_dir}" \
      --group-alphas "${alphas}" \
      --overwrite
  else
    echo "[skip merge] existing merged model found"
  fi

  eval_args=()
  if [[ -n "${LIMIT}" ]]; then
    eval_args+=(--limit "${LIMIT}")
  fi

  "${PYTHON_BIN}" tools/evaluate_qwen_action_offline.py \
    --validation-jsonl "${VAL_JSONL}" \
    --model "${model_dir}" \
    --output-jsonl "${result_dir}/predictions.jsonl" \
    --summary-json "${result_dir}/summary.json" \
    --backend transformers \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-new-tokens 2 \
    "${eval_args[@]}"

  if [[ "${DELETE_MERGED}" == "1" ]]; then
    echo "[delete merged] ${model_dir}"
    rm -rf "${model_dir}"
  fi
done

echo "done: ${RESULT_ROOT}"
