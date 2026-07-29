#!/bin/bash
# TF++ + image enhancement + rear camera + Qwen VLM — dev10 routes
# 전방/후방 이미지를 classic_cv로 보정하고, Qwen speed/rule/emergency-rear critic을 함께 테스트한다.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Image enhancement. TF++ uses rgb_front, while Qwen uses rgb_front for
# speed/rule prompts and rgb_rear for emergency-rear prompts.
export QWEN_IMAGE_ENHANCER=${QWEN_IMAGE_ENHANCER:-classic_cv}
export QWEN_IMAGE_ENHANCE_TARGETS=${QWEN_IMAGE_ENHANCE_TARGETS:-rgb_front,rgb_rear}
export QWEN_IMAGE_ENHANCE_SAVE_COMPARE=${QWEN_IMAGE_ENHANCE_SAVE_COMPARE:-1}
export QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL=${QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL:-4}

# Rear camera and rear VLM path enabled.
export QWEN_DASHBOARD_REAR=${QWEN_DASHBOARD_REAR:-1}
export QWEN_EMERGENCY_PULL_OVER=${QWEN_EMERGENCY_PULL_OVER:-1}
export QWEN_EMERGENCY_CONTROL_OVERRIDE=${QWEN_EMERGENCY_CONTROL_OVERRIDE:-1}
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=${QWEN_FORCE_EMERGENCY_LANE_CHANGE:-0}

export OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/qwen_enhance_rear_dev10}

exec "${SCRIPT_DIR}/run_qwen_dev10.sh" "$@"
