#!/bin/bash
# TF++ + classic_cv image enhancement only — dev10 routes
# Qwen/VLM 없이 upstream TF++ 입력 이미지에만 기존 classic_cv 보정을 적용한다.

set -eo pipefail

# ──────────────────────────────────────────────
# 환경 변수
# ──────────────────────────────────────────────
export CARLA_ROOT=/mnt/2/carla
export WORK_DIR=/mnt/2/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export LEADERBOARD_PYTHON=${LEADERBOARD_PYTHON:-/home/kwy00/anaconda3/envs/garage_2/bin/python}
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/team_code
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/src
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# TF++ original behavior + image enhancer only.
# classic_enhance.yaml extends base.yaml where vlm=null, risk=noop, safety=noop.
export GARAGE_EXT_CONFIG=${GARAGE_EXT_CONFIG:-/mnt/2/carla_garage/configs/experiments/classic_enhance.yaml}

export DEBUG_CHALLENGE=1
export IS_BENCH2DRIVE=True
export CARLA_QUALITY_LEVEL=Epic
export EXT_DASHBOARD_INTERVAL=${EXT_DASHBOARD_INTERVAL:-4}
export EXT_ENHANCE_COMPARE_REAR=${EXT_ENHANCE_COMPARE_REAR:-1}

BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}
GPU_RANK=${GPU_RANK:-0}

ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}

TEAM_AGENT=/mnt/2/carla_garage/src/garage_ext/agents/ext_sensor_agent.py
TEAM_CONFIG=/mnt/2/pretrained_models/all_towns

OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/tfpp_enhance_dev10}
CHECKPOINT_ENDPOINT=${OUT_DIR}/eval.json
SAVE_PATH=${OUT_DIR}/viz
LOG_FILE=${OUT_DIR}/eval.log

export SAVE_PATH

MAX_RETRIES=${MAX_RETRIES:-30}
RESTART_WAIT=${RESTART_WAIT:-30}

kill_carla() {
    echo "[restart] 잔존 CARLA / leaderboard 프로세스 정리..."
    pkill -9 -f "CarlaUE4"                2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator"   2>/dev/null || true
    pkill -9 -f "scenario_manager"        2>/dev/null || true
    sleep 5
}

mkdir -p "$OUT_DIR" "$SAVE_PATH"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AGENT      : ExtSensorAgent"
echo "MODE       : TF++ + image enhancement only"
echo "ENHANCE    : $GARAGE_EXT_CONFIG"
echo "Qwen/VLM   : disabled"
echo "Rear input : compare-only=${EXT_ENHANCE_COMPARE_REAR}"
echo "ROUTES     : $ROUTES"
echo "OUT_DIR    : $OUT_DIR"
echo "SAVE_PATH  : $SAVE_PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "[restart] ▶ 시도 $attempt / $MAX_RETRIES"

    set +e
    CUDA_VISIBLE_DEVICES=${GPU_RANK} "$LEADERBOARD_PYTHON" \
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
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[restart] ✅ 평가 완료"
        break
    fi

    if [ $attempt -ge $MAX_RETRIES ]; then
        echo "[restart] ❌ 최대 재시도($MAX_RETRIES) 초과"
        break
    fi

    echo "[restart] ⚠️  크래시 감지 (exit=$EXIT_CODE) — ${RESTART_WAIT}초 후 재시작"
    kill_carla
    sleep $RESTART_WAIT
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "결과       : $CHECKPOINT_ENDPOINT"
echo "시각화     : $SAVE_PATH"
echo "보정 비교  : $SAVE_PATH/<route>/enhance_compare"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
