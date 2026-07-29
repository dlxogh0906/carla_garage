#!/bin/bash
# TF++ + teammate-style Qwen 8 meta-action VLA — dev10 routes.
# Rear-camera/emergency-yield logic is intentionally excluded. ClassicCV image
# enhancement can be enabled with QWEN_8META_IMAGE_ENHANCER=classic_cv.

set -eo pipefail

export CARLA_ROOT=/mnt/2/carla
export WORK_DIR=/mnt/2/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export LEADERBOARD_PYTHON=${LEADERBOARD_PYTHON:-/home/kwy00/anaconda3/envs/garage_2/bin/python}
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/team_code
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/src
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export QWEN_VLM_ENABLED=${QWEN_VLM_ENABLED:-1}
export QWEN_MODEL=${QWEN_MODEL:-/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct}
export QWEN_TTC_THRESHOLD=${QWEN_TTC_THRESHOLD:-3.0}
export QWEN_8META_EVERY_N_STEPS=${QWEN_8META_EVERY_N_STEPS:-20}
export QWEN_MAX_NEW_TOKENS=${QWEN_MAX_NEW_TOKENS:-10}
export QWEN_VLM_THINKING=${QWEN_VLM_THINKING:-0}
export QWEN_LATERAL_THRESH=${QWEN_LATERAL_THRESH:-2.0}
export QWEN_FRONT_MAX_DISTANCE=${QWEN_FRONT_MAX_DISTANCE:-80.0}
export QWEN_BENCHMARK_INFER=${QWEN_BENCHMARK_INFER:-1}
export QWEN_SAVE_DASHBOARD=${QWEN_SAVE_DASHBOARD:-1}

# Explicitly keep excluded features off, even if inherited from a previous shell.
unset QWEN_IMAGE_ENHANCER
export QWEN_DASHBOARD_REAR=0
export QWEN_EMERGENCY_PULL_OVER=0
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=0

export DEBUG_CHALLENGE=1
export IS_BENCH2DRIVE=True
export CARLA_QUALITY_LEVEL=Epic
export EXT_DASHBOARD_INTERVAL=${EXT_DASHBOARD_INTERVAL:-4}

BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}
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

VISIBLE_GPU_COUNT=$(python3 - <<PY
items = [x.strip() for x in "${CUDA_VISIBLE_DEVICES_LIST}".split(",") if x.strip()]
print(len(items))
PY
)

if [ -z "${QWEN_VLM_DEVICE+x}" ]; then
  if [ "$VISIBLE_GPU_COUNT" -ge 2 ]; then
    export QWEN_VLM_DEVICE="cuda:1"
  else
    export QWEN_VLM_DEVICE="cuda:0"
  fi
fi

if [[ "$QWEN_VLM_DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  QWEN_DEVICE_INDEX="${BASH_REMATCH[1]}"
  if [ "$QWEN_DEVICE_INDEX" -ge "$VISIBLE_GPU_COUNT" ]; then
    echo "[config error] QWEN_VLM_DEVICE=$QWEN_VLM_DEVICE but CUDA_VISIBLE_DEVICES_LIST=$CUDA_VISIBLE_DEVICES_LIST exposes only $VISIBLE_GPU_COUNT device(s)."
    exit 2
  fi
fi

ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}
TEAM_AGENT=${TEAM_AGENT:-/mnt/2/carla_garage/src/garage_ext/agents/qwen_8meta_action_agent.py}
TEAM_CONFIG=${TEAM_CONFIG:-/mnt/2/pretrained_models/all_towns}

OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/qwen_8meta_action_dev10}
CHECKPOINT_ENDPOINT=${OUT_DIR}/eval.json
SAVE_PATH=${OUT_DIR}/viz
LOG_FILE=${OUT_DIR}/eval.log
export SAVE_PATH

MAX_RETRIES=${MAX_RETRIES:-30}
RESTART_WAIT=${RESTART_WAIT:-30}
RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE:-0}
RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD:--1}
RUNTIME_SUMMARY_INTERVAL=${RUNTIME_SUMMARY_INTERVAL:-60}
RUNTIME_SUMMARY_PID=""

kill_carla() {
  if [ "${DISABLE_PROCESS_KILL:-0}" = "1" ]; then
    echo "[restart] process cleanup disabled; skip pkill for port ${BASE_PORT}"
    return 0
  fi
  echo "[restart] cleanup CARLA / leaderboard for port ${BASE_PORT}"
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
        if [ "${DISABLE_PROCESS_KILL:-0}" = "1" ]; then
          echo "[watchdog] process kill disabled; leaving evaluator untouched" | tee -a "$LOG_FILE"
          exit 0
        fi
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
  "$LEADERBOARD_PYTHON" /mnt/2/carla_garage/tools/summarize_qwen_runtime.py \
    --input "${SAVE_PATH}" \
    --output-prefix "${OUT_DIR}/qwen_runtime" \
    --model-name "${QWEN_MODEL_LABEL:-Qwen3-VL-8B-Team8MetaAction}" \
    --quant "${QWEN_QUANT:-${QWEN_RUNTIME_QUANT:-unknown}}" \
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

mkdir -p "$OUT_DIR" "$SAVE_PATH"
touch "$LOG_FILE"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AGENT      : ${TEAM_AGENT}"
echo "Method     : teammate 8 meta-action digit VLA"
echo "Input      : front-only, enhancer=${QWEN_8META_IMAGE_ENHANCER:-off}, no rear camera"
echo "VLM        : ${QWEN_MODEL} enabled=${QWEN_VLM_ENABLED} backend=${QWEN_VLM_BACKEND:-transformers}"
echo "TTC        : danger < ${QWEN_TTC_THRESHOLD}s; every ${QWEN_8META_EVERY_N_STEPS} steps"
echo "CUDA vis   : ${CUDA_VISIBLE_DEVICES_LIST} (TF++ cuda:${GPU_RANK})"
echo "VLM device : ${QWEN_VLM_DEVICE}"
echo "Retry      : MAX_RETRIES=${MAX_RETRIES}, RESTART_WAIT=${RESTART_WAIT}, RETRY_FAILED_ROUTE=${RETRY_FAILED_ROUTE}, RETRY_LOW_SCORE_THRESHOLD=${RETRY_LOW_SCORE_THRESHOLD}"
echo "Runtime    : update ${OUT_DIR}/qwen_runtime_*.csv every ${RUNTIME_SUMMARY_INTERVAL}s"
echo "ROUTES     : ${ROUTES}"
echo "OUT_DIR    : ${OUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

start_runtime_summarizer
trap stop_runtime_summarizer EXIT

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
  attempt=$((attempt + 1))
  echo ""
  echo "[restart] ▶ attempt $attempt / $MAX_RETRIES"

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

  if [ $EXIT_CODE -eq 0 ]; then
    echo "[restart] evaluation complete"
    break
  fi
  if [ $attempt -ge $MAX_RETRIES ]; then
    echo "[restart] max retries exceeded"
    break
  fi
  echo "[restart] crash exit=${EXIT_CODE}; restart after ${RESTART_WAIT}s"
  kill_carla
  rewind_retry_route
  sleep "$RESTART_WAIT"
done

stop_runtime_summarizer
summarize_runtime_once

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Result     : $CHECKPOINT_ENDPOINT"
echo "JSONL logs : $SAVE_PATH/*/qwen_intervention.jsonl"
echo "calls      : ${OUT_DIR}/qwen_runtime_calls.csv"
echo "summary    : ${OUT_DIR}/qwen_runtime_summary.csv"
echo "paper      : ${OUT_DIR}/qwen_runtime_paper_table.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${QWEN_SKIP_VIDEO:-0}" = "1" ]; then
  echo "[video] QWEN_SKIP_VIDEO=1, skipping MP4 generation"
  exit 0
fi

FFMPEG=/home/kwy00/anaconda3/envs/cogs/bin/ffmpeg
VIDEO_DIR="${OUT_DIR}/dashboard_videos"
mkdir -p "$VIDEO_DIR"

find "$SAVE_PATH" -mindepth 2 -maxdepth 2 -type d -name "dashboard" | sort | while read -r dash_dir; do
  route_name=$(basename "$(dirname "$dash_dir")")
  png_count=$(find "$dash_dir" -name "*.png" | wc -l)
  [ "$png_count" -eq 0 ] && echo "  [skip] $route_name dashboard" && continue
  out_mp4="${VIDEO_DIR}/${route_name}.mp4"
  "$FFMPEG" -y -framerate 10 \
    -pattern_type glob -i "${dash_dir}/*.png" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
    "$out_mp4" >> "$LOG_FILE" 2>&1 \
    && echo "  done $route_name dashboard" || echo "  ffmpeg failed: $route_name"
done
