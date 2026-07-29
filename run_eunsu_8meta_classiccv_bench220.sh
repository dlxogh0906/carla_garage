#!/bin/bash
# TF++ + Eunsu-style 8 meta-action VLA + ClassicCV image enhancement.
# Full Bench2Drive 220 routes. Front camera only, no rear-camera logic.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/home/kwy00/anaconda3/envs/garage_2/bin/python}

export CARLA_ROOT=${CARLA_ROOT:-/mnt/2/carla}
export WORK_DIR=${WORK_DIR:-/mnt/2/carla_garage/Bench2Drive}
export SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT:-${WORK_DIR}/scenario_runner}
export LEADERBOARD_ROOT=${LEADERBOARD_ROOT:-${WORK_DIR}/leaderboard}
export LEADERBOARD_PYTHON=${LEADERBOARD_PYTHON:-/home/kwy00/anaconda3/envs/garage_2/bin/python}

export PYTHONPATH="${SCRIPT_DIR}/team_code:${SCRIPT_DIR}:${SCRIPT_DIR}/src:${SCRIPT_DIR}/Bench2Drive:${SCRIPT_DIR}/src/garage_ext/visualization:${PYTHONPATH}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# Full Bench2Drive 220 route file. Override ROUTES for a split XML.
export ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/bench2drive220.xml}

# Clear, searchable output folder name.
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/eunsu_8meta_classiccv_bench220_${RUN_STAMP}}
export CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-${OUT_DIR}/eval.json}
export SAVE_PATH=${SAVE_PATH:-${OUT_DIR}/viz}
export LOG_FILE=${LOG_FILE:-${OUT_DIR}/eval.log}

# Eunsu-style 8 meta-action VLA settings.
export META_MODEL=${META_MODEL:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}
export META_TTC_THRESHOLD=${META_TTC_THRESHOLD:-3.0}
export META_EVERY_N_STEPS=${META_EVERY_N_STEPS:-20}
export META_DASHBOARD_PATH=${META_DASHBOARD_PATH:-${OUT_DIR}/meta_dashboard}

# ClassicCV image enhancement. The agent applies this before TF++ and before
# the image is cached for the VLA prompt.
export USE_CLASSIC_CV=${USE_CLASSIC_CV:-1}
export ENH_VIS_PATH=${ENH_VIS_PATH:-${OUT_DIR}/classiccv_compare}
export ENH_VIS_MAX_ROUTES=${ENH_VIS_MAX_ROUTES:-10}

# Avoid giant videos by default on 220 routes.
export QWEN_SKIP_VIDEO=${QWEN_SKIP_VIDEO:-1}

export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-1}
export IS_BENCH2DRIVE=${IS_BENCH2DRIVE:-True}
export CARLA_QUALITY_LEVEL=${CARLA_QUALITY_LEVEL:-Epic}

BASE_PORT=${BASE_PORT:-30044}
BASE_TM_PORT=${BASE_TM_PORT:-50044}
GPU_RANK=${GPU_RANK:-0}

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
GPU_COUNT=${GPU_COUNT//[[:space:]]/}
if [ -z "${CUDA_VISIBLE_DEVICES_LIST:-}" ]; then
  if [ "${GPU_COUNT:-0}" -ge 2 ]; then
    CUDA_VISIBLE_DEVICES_LIST="0,1"
  else
    CUDA_VISIBLE_DEVICES_LIST="${GPU_RANK}"
  fi
fi
export CUDA_VISIBLE_DEVICES_LIST

TEAM_AGENT=${TEAM_AGENT:-${SCRIPT_DIR}/team_code/eunsu_sensor_agent_meta_action_classic_cv.py}
TEAM_CONFIG=${TEAM_CONFIG:-/mnt/2/pretrained_models/all_towns}

MAX_RETRIES=${MAX_RETRIES:-30}
RESTART_WAIT=${RESTART_WAIT:-30}
RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE:-0}
RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD:--1}
RUNTIME_SUMMARY_INTERVAL=${RUNTIME_SUMMARY_INTERVAL:-60}
RUNTIME_SUMMARY_PID=""

kill_carla() {
  echo "[cleanup] CARLA / leaderboard cleanup for port ${BASE_PORT}"
  pkill -9 -f "[C]arlaUE4.*carla-rpc-port=${BASE_PORT}" 2>/dev/null || true
  pkill -9 -f "[l]eaderboard_evaluator.py.*--port=${BASE_PORT}" 2>/dev/null || true
  pkill -9 -f "[l]eaderboard_evaluator.py.*--checkpoint=${CHECKPOINT_ENDPOINT}" 2>/dev/null || true
  sleep 5
}

rewind_retry_route() {
  [ -f "${CHECKPOINT_ENDPOINT}" ] || return 0
  "$LEADERBOARD_PYTHON" - "$CHECKPOINT_ENDPOINT" "$RETRY_FAILED_ROUTE" "$RETRY_LOW_SCORE_THRESHOLD" <<'PY' || true
import json
import shutil
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
retry_failed = sys.argv[2] == "1"
low_threshold = float(sys.argv[3])
data = json.loads(path.read_text())
ck = data.get("_checkpoint", {})
records = ck.get("records", [])
if not records:
    raise SystemExit(0)
last = records[-1]
status = str(last.get("status", ""))
idx = int(last.get("index", -1))
score = float(last.get("scores", {}).get("score_composed", 101.0))
reason = None
if retry_failed and status.startswith("Failed"):
    reason = f"failed_status:{status}"
elif low_threshold >= 0 and score <= low_threshold:
    reason = f"low_score:{score:.3f}<=threshold:{low_threshold:.3f}"
if idx < 0 or reason is None:
    raise SystemExit(0)
backup = path.with_suffix(path.suffix + f".bak_rewind_{idx}_{time.strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(path, backup)
ck["records"] = [r for r in records if int(r.get("index", -1)) < idx]
total = ck.get("progress", [idx, 220])[1]
ck["progress"] = [idx, total]
data["entry_status"] = "Started"
data["eligible"] = False
data["values"] = []
data["labels"] = []
path.write_text(json.dumps(data, indent=4) + "\n")
print(f"[restart] rewound route index={idx}; reason={reason}; backup={backup}")
PY
}

start_crash_watchdog() {
  local start_line="$1"
  (
    while true; do
      sleep "${CRASH_WATCHDOG_INTERVAL:-30}"
      [ -f "$LOG_FILE" ] || continue
      if tail -n +"$((start_line + 1))" "$LOG_FILE" | grep -E -q "Signal 11|please restart|The scenario could not be loaded|time-out of 600000ms while waiting for the simulator|couldn't spawn the ego vehicle"; then
        echo "[watchdog] crash pattern detected; terminating evaluator on port ${BASE_PORT}" | tee -a "$LOG_FILE"
        pkill -TERM -f "[l]eaderboard_evaluator.py.*--port=${BASE_PORT}" 2>/dev/null || true
        pkill -TERM -f "[l]eaderboard_evaluator.py.*--checkpoint=${CHECKPOINT_ENDPOINT}" 2>/dev/null || true
        sleep 10
        pkill -KILL -f "[l]eaderboard_evaluator.py.*--port=${BASE_PORT}" 2>/dev/null || true
        pkill -KILL -f "[l]eaderboard_evaluator.py.*--checkpoint=${CHECKPOINT_ENDPOINT}" 2>/dev/null || true
        exit 0
      fi
    done
  ) >>"$LOG_FILE" 2>&1 &
  WATCHDOG_PID=$!
}

summarize_runtime_once() {
  "$PYTHON_BIN" "${SCRIPT_DIR}/tools/summarize_qwen_runtime.py" \
    --input "${SAVE_PATH}" \
    --output-prefix "${OUT_DIR}/qwen_runtime" \
    --model-name "Qwen3-VL-8B-Eunsu8MetaClassicCV" \
    --quant "${QWEN_QUANT:-BF16-transformers}" \
    >>"$LOG_FILE" 2>&1 || true
}

start_runtime_summarizer() {
  (
    while true; do
      sleep "$RUNTIME_SUMMARY_INTERVAL"
      summarize_runtime_once
    done
  ) &
  RUNTIME_SUMMARY_PID=$!
}

stop_runtime_summarizer() {
  if [ -n "$RUNTIME_SUMMARY_PID" ]; then
    kill "$RUNTIME_SUMMARY_PID" 2>/dev/null || true
    wait "$RUNTIME_SUMMARY_PID" 2>/dev/null || true
    RUNTIME_SUMMARY_PID=""
  fi
}

mkdir -p "$OUT_DIR" "$SAVE_PATH" "$META_DASHBOARD_PATH" "$ENH_VIS_PATH" "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment : Eunsu 8meta action + ClassicCV image enhancement"
echo "Routes     : ${ROUTES}"
echo "Agent      : ${TEAM_AGENT}"
echo "TF++ ckpt  : ${TEAM_CONFIG}"
echo "Qwen model : ${META_MODEL}"
echo "VLA        : one digit 0-7, TTC < ${META_TTC_THRESHOLD}s, every ${META_EVERY_N_STEPS} steps"
echo "ClassicCV  : USE_CLASSIC_CV=${USE_CLASSIC_CV}, compare=${ENH_VIS_PATH}, max_routes=${ENH_VIS_MAX_ROUTES}"
echo "CUDA vis   : ${CUDA_VISIBLE_DEVICES_LIST}; TF++ gpu-rank=${GPU_RANK}"
echo "Retry      : MAX_RETRIES=${MAX_RETRIES}, RESTART_WAIT=${RESTART_WAIT}, RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE}, RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD}"
echo "Runtime    : update ${OUT_DIR}/qwen_runtime_*.csv every ${RUNTIME_SUMMARY_INTERVAL}s"
echo "Output     : ${OUT_DIR}"
echo "Eval JSON  : ${CHECKPOINT_ENDPOINT}"
echo "Viz        : ${SAVE_PATH}"
echo "Meta dash  : ${META_DASHBOARD_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

start_runtime_summarizer
trap stop_runtime_summarizer EXIT

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
  attempt=$((attempt + 1))
  echo ""
  echo "[restart] attempt ${attempt}/${MAX_RETRIES}"

  set +e
  LOG_START_LINE=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
  start_crash_watchdog "$LOG_START_LINE"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST} "$LEADERBOARD_PYTHON" \
    "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --routes="${ROUTES}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${TEAM_CONFIG}" \
    --debug=1 \
    --resume=True \
    --port="${BASE_PORT}" \
    --traffic-manager-port="${BASE_TM_PORT}" \
    --gpu-rank="${GPU_RANK}" \
    2>&1 | tee -a "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
  summarize_runtime_once
  set -e

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[restart] evaluation complete"
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "[restart] max retries exceeded: ${MAX_RETRIES}"
    break
  fi

  echo "[restart] crashed with exit=${EXIT_CODE}; retrying after ${RESTART_WAIT}s"
  kill_carla
  rewind_retry_route
  sleep "$RESTART_WAIT"
done

stop_runtime_summarizer
summarize_runtime_once

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Result     : ${CHECKPOINT_ENDPOINT}"
echo "Viz        : ${SAVE_PATH}"
echo "Meta dash  : ${META_DASHBOARD_PATH}"
echo "ClassicCV  : ${ENH_VIS_PATH}"
echo "calls      : ${OUT_DIR}/qwen_runtime_calls.csv"
echo "summary    : ${OUT_DIR}/qwen_runtime_summary.csv"
echo "paper      : ${OUT_DIR}/qwen_runtime_paper_table.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${QWEN_SKIP_VIDEO:-1}" = "1" ]; then
  echo "[video] QWEN_SKIP_VIDEO=1, skipping MP4 generation"
  exit 0
fi

FFMPEG=${FFMPEG:-/home/kwy00/anaconda3/envs/cogs/bin/ffmpeg}
VIDEO_DIR="${OUT_DIR}/videos"
mkdir -p "$VIDEO_DIR"

find "$SAVE_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r route_dir; do
  route_name=$(basename "$route_dir")
  png_count=$(find "$route_dir" -maxdepth 1 -name "*.png" | wc -l)
  [ "$png_count" -eq 0 ] && continue
  "$FFMPEG" -y -framerate 10 \
    -pattern_type glob -i "${route_dir}/*.png" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
    "${VIDEO_DIR}/${route_name}.mp4" >> "$LOG_FILE" 2>&1 || true
done
