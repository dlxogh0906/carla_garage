#!/bin/bash
# TF++ 단독 평가 — dev10 라우트 / CARLA Low 품질 / RGB 시각화
# run_evaluation.sh 은 DEBUG_CHALLENGE=0 으로 덮어쓰므로 evaluator 직접 호출

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

# ---- 시각화 스위치 (RGB 주행 장면 저장) ----
export DEBUG_CHALLENGE=1
export IS_BENCH2DRIVE=True

# ---- 그래픽 품질: Low는 RenderOffScreen과 조합 시 특정 씬에서 Segfault 발생 ----
export CARLA_QUALITY_LEVEL=Epic

BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}

ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}

# TF++ 원본 에이전트 (Alpamayo/ExtPipeline 없음)
TEAM_AGENT=/mnt/2/carla_garage/team_code/sensor_agent.py
TEAM_CONFIG=/mnt/2/pretrained_models/all_towns
PLANNER_TYPE=traj
GPU_RANK=${GPU_RANK:-0}

OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/tfpp_dev10_epic}
CHECKPOINT_ENDPOINT=${OUT_DIR}/eval.json
SAVE_PATH=${OUT_DIR}/viz
LOG_FILE=${OUT_DIR}/eval.log

export SAVE_PATH

MAX_RETRIES=30
RESTART_WAIT=30

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
kill_carla() {
    echo "[restart] 잔존 CARLA / leaderboard 프로세스 정리..."
    pkill -9 -f "CarlaUE4"                2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator"   2>/dev/null || true
    pkill -9 -f "scenario_manager"        2>/dev/null || true
    sleep 5
}

# ──────────────────────────────────────────────
# 평가 루프
# ──────────────────────────────────────────────
mkdir -p "$OUT_DIR" "$SAVE_PATH"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TF++ ONLY  : dev10 routes / Low quality"
echo "ROUTES     : $ROUTES"
echo "CHECKPOINT : $CHECKPOINT_ENDPOINT"
echo "SAVE_PATH  : $SAVE_PATH"
echo "LOG        : $LOG_FILE"
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
echo "결과: $CHECKPOINT_ENDPOINT"
echo "시각화: $SAVE_PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── PNG 시퀀스 → MP4 영상 변환 ─────────────────────────────────────────────
VIDEO_DIR="${OUT_DIR}/videos"
mkdir -p "$VIDEO_DIR"
echo "[video] 루트별 MP4 생성 중..."

find "$SAVE_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r route_dir; do
    route_name=$(basename "$route_dir")
    first_frame=$(ls "$route_dir"/*.png 2>/dev/null | head -1)
    if [ -z "$first_frame" ]; then
        echo "  [skip] $route_name — 이미지 없음"
        continue
    fi
    out_mp4="${VIDEO_DIR}/${route_name}.mp4"
    echo "  [ffmpeg] $route_name → $(basename $out_mp4)"
    ffmpeg -y -framerate 10 \
        -pattern_type glob -i "${route_dir}/*.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
        "$out_mp4" \
        >> "$LOG_FILE" 2>&1 \
        && echo "    ✅ 완료: $out_mp4" \
        || echo "    ⚠️  ffmpeg 실패: $route_name"
done

echo "[video] 완료 → $VIDEO_DIR"
