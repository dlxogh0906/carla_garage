#!/bin/bash
# TF++ + Meta-action VLA (Qwen3-VL-8B-Instruct) — dev10 routes
# 팀원 코드(sensor_agent_meta_action.py, meta_action_vla.py) 기반 별도 실험 스크립트
# Qwen 스크립트(run_qwen_dev10.sh)와 완전히 독립 운영

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
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/Bench2Drive
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/src/garage_ext/visualization
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


# ──────────────────────────────────────────────
# Meta-action VLA 설정 (meta_action_vla.py 의 env var)
# ──────────────────────────────────────────────
# TTC < META_TTC_THRESHOLD(초) 일 때만 VLM 개입 (TF++ 주행 최대 보존)
export META_TTC_THRESHOLD=${META_TTC_THRESHOLD:-3.0}

# 동일 상황에서 최소 재추론 간격 (스텝 수, 기본 20스텝 = 2s at 10Hz)
export META_EVERY_N_STEPS=${META_EVERY_N_STEPS:-30}

# 신호등/정지표지판 rule critic은 speed critic과 별도 간격으로 호출
export META_RULE_EVERY_N_STEPS=${META_RULE_EVERY_N_STEPS:-5}
export META_RULE_CONFIDENCE_THRESH=${META_RULE_CONFIDENCE_THRESH:-0.75}
export META_RULE_MAX_AGE_STEPS=${META_RULE_MAX_AGE_STEPS:-20}
export META_RULE_ENABLE_STOP_SIGN=${META_RULE_ENABLE_STOP_SIGN:-0}
export META_RULE_HOLD_ENABLED=${META_RULE_HOLD_ENABLED:-1}
export META_RULE_HOLD_POLL_STEPS=${META_RULE_HOLD_POLL_STEPS:-5}
export META_RULE_HOLD_GREEN_CONFIRMATIONS=${META_RULE_HOLD_GREEN_CONFIRMATIONS:-2}
export META_RULE_HOLD_MAX_STEPS=${META_RULE_HOLD_MAX_STEPS:-0}
export META_RULE_HOLD_NO_STOP_VOTES=${META_RULE_HOLD_NO_STOP_VOTES:-2}
export META_RULE_HOLD_SAFETY_STEPS=${META_RULE_HOLD_SAFETY_STEPS:-80}
export META_RULE_HOLD_POST_RELEASE_COOLDOWN=${META_RULE_HOLD_POST_RELEASE_COOLDOWN:-30}
export META_STOP_SIGN_RELEASE_STEPS=${META_STOP_SIGN_RELEASE_STEPS:-60}
export META_TL_PRESTOP_ENABLED=${META_TL_PRESTOP_ENABLED:-1}
export META_TL_PRESTOP_SCALE=${META_TL_PRESTOP_SCALE:-0.45}
export META_TL_PRESTOP_DISTANCE=${META_TL_PRESTOP_DISTANCE:-45.0}

# 비신호 교차로 gap critic: 맞은편/측면 차량이 보이면 Qwen이 진입 타이밍을 판단
export META_GAP_CRITIC_ENABLED=${META_GAP_CRITIC_ENABLED:-1}
export META_GAP_ROUTE_ONLY=${META_GAP_ROUTE_ONLY:-1}
export META_GAP_EVERY_N_STEPS=${META_GAP_EVERY_N_STEPS:-8}
export META_GAP_MAX_AGE_STEPS=${META_GAP_MAX_AGE_STEPS:-8}
export META_GAP_CONFIDENCE_THRESH=${META_GAP_CONFIDENCE_THRESH:-0.75}
export META_GAP_STRONG_CONFIDENCE_THRESH=${META_GAP_STRONG_CONFIDENCE_THRESH:-0.82}
export META_GAP_STOP_CONFIDENCE_THRESH=${META_GAP_STOP_CONFIDENCE_THRESH:-0.92}
export META_GAP_MIN_SCALE=${META_GAP_MIN_SCALE:-0.85}
export META_GAP_LOOKAHEAD_DISTANCE=${META_GAP_LOOKAHEAD_DISTANCE:-30.0}
export META_GAP_SIDE_DISTANCE=${META_GAP_SIDE_DISTANCE:-10.0}
export META_GAP_IMMEDIATE_X=${META_GAP_IMMEDIATE_X:-12.0}
export META_GAP_IMMEDIATE_Y=${META_GAP_IMMEDIATE_Y:-6.0}
export META_GAP_VISUAL_PROBE=${META_GAP_VISUAL_PROBE:-0}
export META_GAP_RELEASE_NO_CANDIDATE_STEPS=${META_GAP_RELEASE_NO_CANDIDATE_STEPS:-8}
export META_GAP_STOP_RELEASE_STEPS=${META_GAP_STOP_RELEASE_STEPS:-35}
export META_GAP_STOP_RELEASE_SCALE=${META_GAP_STOP_RELEASE_SCALE:-0.85}

# 일반 speed critic은 실제 경로 차단 + TTC 위험일 때만 적용
export META_SPEED_SEMANTIC_ENABLED=${META_SPEED_SEMANTIC_ENABLED:-1}
export META_SPEED_MAX_AGE_STEPS=${META_SPEED_MAX_AGE_STEPS:-12}

# 충돌/끼임 이후 목적지까지 계속 가도록 하는 저속 recovery
export META_RECOVERY_ENABLED=${META_RECOVERY_ENABLED:-1}
export META_RECOVERY_STUCK_STEPS=${META_RECOVERY_STUCK_STEPS:-25}
export META_RECOVERY_DURATION_STEPS=${META_RECOVERY_DURATION_STEPS:-35}
export META_RECOVERY_TARGET_SPEED=${META_RECOVERY_TARGET_SPEED:-1.5}
export META_RECOVERY_THROTTLE=${META_RECOVERY_THROTTLE:-0.45}
export META_RECOVERY_REQUIRES_MOTION=${META_RECOVERY_REQUIRES_MOTION:-1}
export META_RECOVERY_MOTION_FRAMES=${META_RECOVERY_MOTION_FRAMES:-5}

# 좌회전/교차로 진입은 TF++ target이 튀어도 천천히 들어가게 제한
export META_TURN_CAUTION_ENABLED=${META_TURN_CAUTION_ENABLED:-1}
export META_TURN_CAUTION_SPEED=${META_TURN_CAUTION_SPEED:-3.0}
export META_TURN_CAUTION_LATERAL=${META_TURN_CAUTION_LATERAL:-2.5}

# 전진 recovery로도 못 빠져나오면 짧게 후진해서 울타리/차량 접촉에서 탈출
export META_ESCAPE_REVERSE_ENABLED=${META_ESCAPE_REVERSE_ENABLED:-1}
export META_ESCAPE_STUCK_STEPS=${META_ESCAPE_STUCK_STEPS:-90}
export META_ESCAPE_REVERSE_STEPS=${META_ESCAPE_REVERSE_STEPS:-14}
export META_ESCAPE_THROTTLE=${META_ESCAPE_THROTTLE:-0.35}

# Qwen meta-action input enrichment
export META_LATERAL_THRESH=${META_LATERAL_THRESH:-2.0}
export META_FRONT_MAX_DISTANCE=${META_FRONT_MAX_DISTANCE:-80.0}
export META_MAX_OBJECTS=${META_MAX_OBJECTS:-8}
export META_SAVE_INPUTS=${META_SAVE_INPUTS:-1}
export EXT_DASHBOARD_INTERVAL=${EXT_DASHBOARD_INTERVAL:-4}

# AllWeatherNet 이미지 향상 (팀원 코드 옵션 — 기본 끄기)
export USE_ALLWEATHERNET=${USE_ALLWEATHERNET:-0}

# ──────────────────────────────────────────────
# 일반 설정
# ──────────────────────────────────────────────
export DEBUG_CHALLENGE=1          # RGB 시각화 저장
export IS_BENCH2DRIVE=True
export CARLA_QUALITY_LEVEL=Epic   # Low는 특정 씬에서 Segfault 발생

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

#ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10_3routes_no25424.xml}
ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/drivetransformer_bench2drive_dev10.xml}

# 팀원 에이전트 (수정 금지 — 원본 그대로 사용)
TEAM_AGENT=/mnt/2/carla_garage/team_code/sensor_agent_meta_action.py
TEAM_CONFIG=/mnt/2/pretrained_models/all_towns

# ── 출력 경로 ────────────────────────────────
OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/meta_action_dev10_1}
CHECKPOINT_ENDPOINT=${OUT_DIR}/eval.json
SAVE_PATH=${OUT_DIR}/viz
LOG_FILE=${OUT_DIR}/eval.log

# Meta-action 대시보드 저장 경로 (sensor_agent_meta_action.py 의 META_DASHBOARD_PATH)
export META_DASHBOARD_PATH=${OUT_DIR}/meta_dashboard
export SAVE_PATH

MAX_RETRIES=30
RESTART_WAIT=30

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
kill_carla() {
    echo "[restart] 잔존 CARLA / leaderboard 프로세스 정리..."
    pkill -9 -f "CarlaUE4"              2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true
    pkill -9 -f "scenario_manager"      2>/dev/null || true
    sleep 5
}

# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
mkdir -p "$OUT_DIR" "$SAVE_PATH" "$META_DASHBOARD_PATH"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AGENT      : SensorAgent (sensor_agent_meta_action.py)"
echo "VLA        : meta_action_vla.py  (Qwen3-VL-8B-Instruct)"
echo "TTC thr    : ${META_TTC_THRESHOLD}s"
echo "Infer gap  : every ${META_GAP_EVERY_N_STEPS} steps"
echo "Rule critic: every ${META_RULE_EVERY_N_STEPS} steps, stop_sign=${META_RULE_ENABLE_STOP_SIGN}, hold=${META_RULE_HOLD_ENABLED}, green_confirm=${META_RULE_HOLD_GREEN_CONFIRMATIONS}, no_stop_votes=${META_RULE_HOLD_NO_STOP_VOTES}"
echo "TL pre-stop: enabled=${META_TL_PRESTOP_ENABLED}, scale=${META_TL_PRESTOP_SCALE}, distance=${META_TL_PRESTOP_DISTANCE}m"
echo "Gap critic : enabled=${META_GAP_CRITIC_ENABLED}, route_only=${META_GAP_ROUTE_ONLY}, every=${META_GAP_EVERY_N_STEPS}, age=${META_GAP_MAX_AGE_STEPS}, conf=${META_GAP_CONFIDENCE_THRESH}, strong=${META_GAP_STRONG_CONFIDENCE_THRESH}, stop_conf=${META_GAP_STOP_CONFIDENCE_THRESH}"
echo "Gap filter : min_scale=${META_GAP_MIN_SCALE}, visual_probe=${META_GAP_VISUAL_PROBE}, lookahead=${META_GAP_LOOKAHEAD_DISTANCE}m side=${META_GAP_SIDE_DISTANCE}m immediate=${META_GAP_IMMEDIATE_X}x${META_GAP_IMMEDIATE_Y}m"
echo "Gap release: no_candidate=${META_GAP_RELEASE_NO_CANDIDATE_STEPS}, stop_hold=${META_GAP_STOP_RELEASE_STEPS}, stop_scale=${META_GAP_STOP_RELEASE_SCALE}"
echo "Speed gate : every=${META_EVERY_N_STEPS}, semantic=${META_SPEED_SEMANTIC_ENABLED}, age=${META_SPEED_MAX_AGE_STEPS}"
echo "Recovery   : enabled=${META_RECOVERY_ENABLED}, stuck=${META_RECOVERY_STUCK_STEPS}, duration=${META_RECOVERY_DURATION_STEPS}, target=${META_RECOVERY_TARGET_SPEED}m/s throttle=${META_RECOVERY_THROTTLE}, requires_motion=${META_RECOVERY_REQUIRES_MOTION}"
echo "Turn cap   : enabled=${META_TURN_CAUTION_ENABLED}, speed=${META_TURN_CAUTION_SPEED}m/s lateral=${META_TURN_CAUTION_LATERAL}m"
echo "Escape rev : enabled=${META_ESCAPE_REVERSE_ENABLED}, stuck=${META_ESCAPE_STUCK_STEPS}, steps=${META_ESCAPE_REVERSE_STEPS}, throttle=${META_ESCAPE_THROTTLE}"
echo "Meta input : lateral=${META_LATERAL_THRESH}m front_max=${META_FRONT_MAX_DISTANCE}m max_objects=${META_MAX_OBJECTS} save=${META_SAVE_INPUTS}"
echo "Qwen dash  : every ${EXT_DASHBOARD_INTERVAL} steps -> ${SAVE_PATH}/<route>/dashboard"
echo "AllWeather : ${USE_ALLWEATHERNET}"
echo "CUDA vis   : $CUDA_VISIBLE_DEVICES_LIST  (TF++ cuda:${GPU_RANK})"
echo "ROUTES     : $ROUTES"
echo "OUT_DIR    : $OUT_DIR"
echo "META_DASH  : $META_DASHBOARD_PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "[restart] ▶ 시도 $attempt / $MAX_RETRIES"

    set +e
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
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[restart] ✅ 평가 완료"
        break
    fi

    if [ $attempt -ge $MAX_RETRIES ]; then
        echo "[restart] ❌ 최대 재시도($MAX_RETRIES) 초과"
        break
    fi

    echo "[restart] ⚠️  크래시 (exit=$EXIT_CODE) — ${RESTART_WAIT}초 후 재시작"
    kill_carla
    sleep $RESTART_WAIT
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "결과       : $CHECKPOINT_ENDPOINT"
echo "시각화     : $SAVE_PATH"
echo "메타대시보드: $META_DASHBOARD_PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── MP4 변환 ──────────────────────────────────────────────────────────
FFMPEG=/home/kwy00/anaconda3/envs/cogs/bin/ffmpeg
VIDEO_DIR="${OUT_DIR}/videos"
DASHBOARD_VIDEO_DIR="${OUT_DIR}/dashboard_videos"
META_DASH_VIDEO_DIR="${OUT_DIR}/meta_dashboard_videos"
mkdir -p "$VIDEO_DIR" "$DASHBOARD_VIDEO_DIR" "$META_DASH_VIDEO_DIR"

echo "[video] RGB 루트별 MP4 생성..."
for route_dir in "$SAVE_PATH"/*/; do
    route_name=$(basename "$route_dir")
    png_count=$(find "$route_dir" -maxdepth 1 -name "*.png" | wc -l)
    [ "$png_count" -eq 0 ] && echo "  [skip] $route_name" && continue
    out_mp4="${VIDEO_DIR}/${route_name}.mp4"
    "$FFMPEG" -y -framerate 10 \
        -pattern_type glob -i "${route_dir}*.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
        "$out_mp4" >> "$LOG_FILE" 2>&1 \
        && echo "  ✅ $route_name" || echo "  ⚠️  ffmpeg 실패: $route_name"
done

echo "[video] Dashboard MP4 생성..."
find "$SAVE_PATH" -mindepth 2 -maxdepth 2 -type d -name "dashboard" | sort | while read -r dash_dir; do
    route_name=$(basename "$(dirname "$dash_dir")")
    png_count=$(find "$dash_dir" -name "*.png" | wc -l)
    [ "$png_count" -eq 0 ] && echo "  [skip] $route_name dashboard" && continue
    out_mp4="${DASHBOARD_VIDEO_DIR}/${route_name}.mp4"
    "$FFMPEG" -y -framerate 10 \
        -pattern_type glob -i "${dash_dir}/*.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
        "$out_mp4" >> "$LOG_FILE" 2>&1 \
        && echo "  ✅ $route_name dashboard" || echo "  ⚠️  ffmpeg 실패: $route_name"
done

echo "[video] Meta-dashboard MP4 생성..."
find "$META_DASHBOARD_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dash_dir; do
    route_name=$(basename "$dash_dir")
    png_count=$(find "$dash_dir" -name "*.png" | wc -l)
    [ "$png_count" -eq 0 ] && echo "  [skip] $route_name meta_dashboard" && continue
    out_mp4="${META_DASH_VIDEO_DIR}/${route_name}.mp4"
    "$FFMPEG" -y -framerate 10 \
        -pattern_type glob -i "${dash_dir}/*.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
        "$out_mp4" >> "$LOG_FILE" 2>&1 \
        && echo "  ✅ $route_name meta_dashboard" || echo "  ⚠️  ffmpeg 실패: $route_name"
done

echo "[video] 완료"
echo "  RGB          → $VIDEO_DIR"
echo "  Dashboard    → $DASHBOARD_VIDEO_DIR"
echo "  Meta-dash    → $META_DASH_VIDEO_DIR"
