# Qwen VLM Intervention Work Summary

This document summarizes the Qwen-related work added on top of TF++ in
`carla_garage`.

## Goal

Use Qwen as a visual safety critic for TF++ without replacing TF++ planning.
TF++ remains the primary driving agent. Qwen receives front-camera visual
context plus selected TF++ internal signals and can only reduce the target
speed through `speed_scale`.

The intended contribution is:

- TF++ provides fast perception/planning priors.
- Qwen verifies visually ambiguous safety/rule situations.
- The final controller applies bounded speed-only intervention.
- Logs keep the decision auditable step by step.

## Main Files

- `src/garage_ext/agents/qwen_sensor_agent.py`
  - TF++ wrapper agent.
  - Collects TF++ speed/path/bbox context.
  - Sends async Qwen requests.
  - Applies speed scaling.
  - Writes JSONL logs and dashboard frames.

- `src/garage_ext/vlm_intervention/qwen_client.py`
  - Async Qwen model loader/inference worker.
  - Supports speed critic and traffic-rule critic prompt modes.
  - Parses Qwen JSON responses.

- `src/garage_ext/vlm_intervention/qwen_input.py`
  - Builds Qwen object tables.
  - Draws annotated front camera input with ego corridor and BEV inset.
  - Summarizes raw TF++ bbox classes.

- `src/garage_ext/vlm_intervention/ttc.py`
  - Computes bbox TTC.
  - Computes sensitive planner-proxy TTC from TF++ braking intent.

- `src/garage_ext/vlm_intervention/logger.py`
  - Writes per-step `qwen_intervention.jsonl`.

- `src/garage_ext/visualization/qwen_dashboard.py`
  - Renders dashboard PNGs.

- `run_qwen_dev10.sh`
  - Main dev10 evaluation script.

- `make_video_from_folder.py`
  - Converts route visualization folders to videos.

## Current Qwen Inputs

Each Qwen request can include:

- Front RGB image.
- Approximate ego corridor overlay.
- BEV inset of TF++ detected objects.
- Ego speed.
- TF++ target speed before intervention.
- TF++ predicted checkpoints.
- Front distance and TTC, when available.
- TTC source:
  - `bbox`
  - `planner_proxy`
  - `none`
- Object table from TF++ bbox output.
- Raw TF++ bbox class summary:
  - `bbox_vehicle_count`
  - `bbox_pedestrian_count`
  - `bbox_traffic_light_count`
  - `bbox_stop_sign_count`
  - `bbox_emergency_vehicle_count`
  - nearest traffic light / stop sign id and approximate ego-coordinate position.

## Critic Modes

### 1. Speed Safety Critic

Used for collision-style speed intervention.

It answers:

- Is the ego corridor blocked?
- Is there a visible vehicle/pedestrian/cyclist/obstacle risk?
- Should TF++ target speed be reduced?

Output fields include:

```json
{
  "intervene": true,
  "risk_level": "high",
  "speed_scale": 0.5,
  "primary_hazard_id": 1,
  "path_blocked": true,
  "tfpp_plan_safe": false,
  "hazard_type": "vehicle",
  "reason": "..."
}
```

### 2. Traffic-Rule Critic

Used for traffic light and stop sign intervention.

It answers:

- Is a red/yellow light relevant to the ego lane/route?
- Is a stop sign relevant to the ego lane/route?
- Should the vehicle stop for the rule object?

Output fields include:

```json
{
  "rule_intervene": true,
  "rule_type": "red_light",
  "traffic_light_state": "red",
  "stop_sign_visible": false,
  "relevant_to_ego": true,
  "confidence": 0.82,
  "speed_scale": 0.3,
  "reason": "..."
}
```

The traffic-rule critic is active by default, but intervention is gated.

## Active Rule Intervention Gate

Rule intervention applies only if all are true:

- `QWEN_RULE_ACTIVE=1`
- `rule_intervene=true`
- `relevant_to_ego=true`
- `rule_type` is one of:
  - `red_light`
  - `yellow_light`
  - `stop_sign`
- `rule_confidence >= QWEN_RULE_CONFIDENCE_THRESH`
- `rule_speed_scale < 0.95`
- result age is within `QWEN_RULE_MAX_AGE_STEPS`

Default gate values:

```bash
QWEN_RULE_CONFIDENCE_THRESH=0.75
QWEN_RULE_MAX_AGE_STEPS=20
QWEN_RULE_PERIODIC_STEPS=20
QWEN_STOP_SIGN_RELEASE_STEPS=60
```

Stop signs have a release guard: once the ego nearly stops, stop-sign rule
intervention is suppressed for `QWEN_STOP_SIGN_RELEASE_STEPS` so the vehicle can
move again.

## TTC Evolution

### Original

TTC was computed only as:

```text
front_distance / ego_speed
```

This failed when TF++ bbox coordinates were unavailable or noisy.

### Sensitive TTC

`compute_sensitive_ttc()` now uses:

- bbox TTC when valid.
- planner-proxy TTC when TF++ strongly wants to slow down and bbox TTC is
  unavailable.

The aggressive version caused too much braking. It was later softened:

- brake-only cues are ignored if TF++ target speed is not actually lower than
  ego speed.
- planner-proxy guard is limited to softer speed scales.

Current proxy guard:

```text
planner_proxy TTC <= 1.0s -> guard scale 0.55
planner_proxy TTC <  2.0s -> guard scale 0.75
otherwise                 -> guard scale 0.90
```

## Important Lessons From dev10

### Bicycle Scenario

Qwen helped because the failure type matched the current speed critic:

```text
visible vulnerable road user -> reduce speed
```

### Red Light / Stop Sign / Lane Change Scenarios

These did not improve reliably with a generic front-risk critic.

Observed failure types:

- Red light violation:
  - requires traffic-rule reasoning, not generic obstacle reasoning.

- Stop sign / intersection collision:
  - requires rule relevance and stop-state handling.

- Lane-change collision:
  - requires target-lane / lateral-path reasoning.
  - speed-only can help but is not enough by itself.

## Why Giving TF++ Outputs To Qwen Is Still Meaningful

The contribution is not "Qwen repeats TF++".

The contribution is a structured arbitration layer:

```text
TF++ perception/planning prior
        +
front camera visual evidence
        ↓
Qwen critic verifies semantic/rule risk
        ↓
bounded speed-only intervention
```

Qwen is used to validate and reinterpret TF++ signals, especially when:

- TF++ slows down but the reason is unclear.
- TF++ bbox context is noisy.
- a traffic rule object may be visually relevant.
- a visible hazard is not represented well by TTC.

## Current Logging Additions

`qwen_intervention.jsonl` now includes:

- VLM state:
  - `vlm_ready`
  - `vlm_called`
  - `vlm_trigger`
  - `vlm_load_error`

- TTC:
  - `ttc`
  - `ttc_source`
  - `is_risky`
  - `guard_scale`

- Speed critic:
  - `qwen_intervene`
  - `qwen_requested_scale`
  - `qwen_path_blocked`
  - `qwen_hazard_type`
  - `qwen_raw_response`

- Traffic-rule critic:
  - `rule_active`
  - `rule_type`
  - `rule_confidence`
  - `rule_relevant`
  - `rule_speed_scale`
  - `traffic_light_state`
  - `stop_sign_visible`
  - `rule_reason`

- Raw TF++ bbox class counts:
  - `bbox_vehicle_count`
  - `bbox_pedestrian_count`
  - `bbox_traffic_light_count`
  - `bbox_stop_sign_count`
  - `bbox_emergency_vehicle_count`
  - nearest traffic light / stop sign coordinates.

## Useful Commands

Run dev10:

```bash
OUT_DIR=/mnt/2/carla_metric_result/qwen_dev10_2 \
CUDA_VISIBLE_DEVICES_LIST=0,1 \
GPU_RANK=0 \
QWEN_VLM_DEVICE=cuda:1 \
bash /mnt/2/carla_garage/scripts/eval/run_qwen_dev10.sh
```

Make dashboard videos:

```bash
python /mnt/2/carla_garage/tools/make_video_from_folder.py \
  /mnt/2/carla_metric_result/qwen_dev10_2/viz \
  -o /mnt/2/carla_metric_result/qwen_dev10_2/dashboard_video \
  --subdir dashboard
```

Make raw viz videos:

```bash
python /mnt/2/carla_garage/tools/make_video_from_folder.py \
  /mnt/2/carla_metric_result/qwen_dev10_2/viz \
  -o /mnt/2/carla_metric_result/qwen_dev10_2/video
```

## Next Checks

After the next run, inspect:

- Does `bbox_traffic_light_count` become nonzero before red-light infractions?
- Does `bbox_stop_sign_count` become nonzero before stop-sign scenarios?
- Does `rule_active=true` happen only near relevant red/yellow lights or stop signs?
- Are there false stops from irrelevant lights/signs?
- Does bicycle improvement remain?
- Does `planner_proxy` stop causing long unnecessary braking?

The most important question is whether TF++ rule candidates are present early
enough. If not, Qwen rule critic must rely more on image crops/zoom rather than
bbox counts.
