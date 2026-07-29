#!/bin/bash
# TF++ + front-only image enhancement + Qwen VLM — dev10 routes
# 후방 카메라 / emergency-rear prompt 없이 front 보정 이미지로 TF++와 Qwen을 함께 테스트한다.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Image enhancement: applied before TF++ sees the frame, so Qwen also receives
# the enhanced front image when it builds the annotated VLM input.
export QWEN_IMAGE_ENHANCER=${QWEN_IMAGE_ENHANCER:-classic_cv}
export QWEN_IMAGE_ENHANCE_TARGETS=${QWEN_IMAGE_ENHANCE_TARGETS:-rgb_front}
export QWEN_IMAGE_ENHANCE_SAVE_COMPARE=${QWEN_IMAGE_ENHANCE_SAVE_COMPARE:-1}
export QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL=${QWEN_IMAGE_ENHANCE_COMPARE_INTERVAL:-4}

# Front-only VLM experiment: do not add rear sensor or emergency-rear prompt.
export QWEN_DASHBOARD_REAR=0
export QWEN_EMERGENCY_PULL_OVER=0
export QWEN_EMERGENCY_CONTROL_OVERRIDE=0
export QWEN_FORCE_EMERGENCY_LANE_CHANGE=0

export OUT_DIR=${OUT_DIR:-/mnt/2/carla_metric_result/drivetransformer_bench2drive_dev10}

exec "${SCRIPT_DIR}/run_qwen_dev10.sh" "$@"
