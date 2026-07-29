# Qwen VLM Speed Intervention

TF++ 주행 위에 Qwen VLM 기반 속도 개입 모듈을 결합한 실험 구현.
Alpamayo와 완전히 독립된 별도 코드베이스.

---

## 파이프라인

```
TF++ run_step
  │
  ├─ 1. TF++ 기본 예측 (perception + planning, 변경 없음)
  │       pred_checkpoints, pred_target_speed_scalar, bb_buffer 생성
  │
  ├─ 2. 전방 객체 거리 추출
  │       get_front_distance(bb_buffer[-1])
  │       bb[0]=x (forward), bb[1]=y (lateral)
  │       |y| < 2.0m → 동일 차선 영역 필터
  │
  ├─ 3. Simplified TTC 계산
  │       ttc = front_distance / ego_speed
  │       ego_speed < 0.5 → ttc = 999 (정차 시 div-by-zero 방지)
  │
  ├─ 4. 위험 판단
  │       ttc < QWEN_TTC_THRESHOLD (기본 3.0s) → risky=True
  │
  ├─ 5. Qwen VLM 호출 (비동기)
  │       조건:
  │         - risky=True
  │         - 또는 front_distance < QWEN_VLM_PROBE_DISTANCE
  │         - 또는 QWEN_VLM_PERIODIC_STEPS 주기 도달
  │       입력:
  │         - ego corridor + BEV inset이 그려진 front image
  │         - ego_speed, front_distance, ttc
  │         - TF++ target speed / planned checkpoints
  │         - top-k bbox object table
  │         - recent TTC history
  │       출력: {intervene, risk_level, speed_scale, primary_hazard_id,
  │              path_blocked, tfpp_plan_safe, hazard_type, reason}
  │       → 백그라운드 스레드에서 처리, 캐시에 저장
  │
  ├─ 6. speed_scale 적용
  │       risky=False → speed_scale = 1.0 (자동 해제)
  │       risky=True → min(Qwen speed_scale, TTC guard scale)
  │       control_pid_direct 패치로 pred_target_speed에 곱해짐
  │       steer/waypoint 변경 없음
  │
  └─ 7. 로그 + 대시보드 저장
```

---

## 수정한 파일

| 파일 | 역할 |
|------|------|
| `src/garage_ext/agents/qwen_sensor_agent.py` | 메인 에이전트 (신규) |
| `src/garage_ext/vlm_intervention/ttc.py` | TTC 계산 유틸 (신규) |
| `src/garage_ext/vlm_intervention/qwen_input.py` | Qwen 입력 이미지/객체 테이블 생성 (신규) |
| `src/garage_ext/vlm_intervention/qwen_client.py` | 비동기 Qwen 클라이언트 (신규) |
| `src/garage_ext/vlm_intervention/logger.py` | JSONL 스텝 로거 (신규) |
| `run_qwen_dev10.sh` | 실행 스크립트 (신규) |

**기존 파일 변경 없음** (sensor_agent.py, ext_sensor_agent.py, Alpamayo 코드 등 무수정)

---

## 추가한 함수

### `ttc.py`

| 함수 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `get_front_distance(pred_bboxes, lateral_thresh=2.0)` | BB 리스트 | float (m) | 전방 동일차선 최근접 객체 거리 |
| `compute_simplified_ttc(front_distance, ego_speed)` | float, float | float (s) | 앞차 정지 가정 TTC |
| `is_risky_ttc(ttc, threshold=3.0)` | float, float | bool | 위험 여부 판단 |

### `qwen_client.py`

| 함수 | 설명 |
|------|------|
| `QwenVLMClient.__init__(model_name, device, enable_thinking)` | 모델 로드 (백그라운드 스레드) |
| `request(image, ego_speed, front_distance, ttc, context)` | 비동기 추론 요청, True=큐 삽입 성공 |
| `get_latest()` | 가장 최근 캐시된 결과 반환 |

### `qwen_input.py`

| 함수 | 설명 |
|------|------|
| `build_object_context(pred_bboxes)` | bbox top-k 객체 표 + primary hazard id 생성 |
| `annotate_qwen_image(...)` | front RGB에 ego corridor 힌트와 BEV inset 추가 |
| `format_object_table(...)` | Qwen 프롬프트용 객체 테이블 생성 |
| `format_ttc_history(...)` | 최근 TTC 변화 요약 |
| `format_path_summary(...)` | TF++ checkpoint 요약 |

### `qwen_sensor_agent.py`

| 함수 | 설명 |
|------|------|
| `setup(path_to_conf_file, ...)` | 에이전트 초기화, Qwen 클라이언트 생성, 패치 적용 |
| `run_step(input_data, timestamp, sensors)` | 메인 루프 (TTC + VLM + 로그) |
| `_patch_speed_scale()` | `control_pid_direct`를 래핑해서 speed_scale 적용 |
| `get_front_distance(pred_bboxes)` | TTC용 전방 거리 추출 |

---

## run_step / control 변경 위치

```python
# sensor_agent.py 내부에서 생성된 값들
pred_target_speed_scalar  # TF++ 목표 속도 (m/s)
pred_checkpoints          # TF++ 계획 checkpoint
bbs_vehicle_coordinate_system  # 라인 562 → bb_buffer[-1] — 예측 바운딩박스
gt_velocity  # input_data["speed"][1]["speed"]에서 읽은 ego 속도

# 패치 위치
# QwenSensorAgent._patch_speed_scale() 에서
# net.control_pid_direct 를 래핑
# pred_target_speed * self._qwen_speed_scale 로 스케일링 후 원본 호출

# final_target_speed = tfpp_target_speed * speed_scale
# → control_pid_direct의 PID가 이 속도를 목표로 throttle/brake 계산
# steer는 절대 수정하지 않음
```

---

## Qwen 프롬프트

```
You are a conservative speed-only safety critic for an autonomous vehicle.

Input includes:
- annotated front image with ego corridor and BEV inset
- ego_speed / front_distance / TTC
- TF++ target speed and planned checkpoints
- top-k detected object table
- recent TTC history

Qwen acts as a TF++ safety critic. It only decides speed reduction.

Respond ONLY with a single JSON object — no markdown, no explanation:
{
  "intervene": <true or false>,
  "risk_level": "<low|medium|high|critical>",
  "speed_scale": <float 0.0 to 1.0>,
  "primary_hazard_id": <integer or null>,
  "path_blocked": <true or false>,
  "tfpp_plan_safe": <true or false>,
  "hazard_type": "<none|vehicle|pedestrian|traffic_light|stop_sign|obstacle|unknown>",
  "reason": "<one brief sentence>"
}

Rules:
- speed_scale 1.0 = keep current speed, 0.0 = full stop
- If intervene is false → speed_scale must be 1.0
- Do NOT suggest steering changes — only adjust speed
/no_think
```

---

## Fallback 처리

| 상황 | 동작 |
|------|------|
| VLM 비활성화 (`QWEN_VLM_ENABLED=0`) | speed_scale=1.0 고정, TF++ 그대로 |
| Qwen 모델 로드 실패 | 에러 로그 후 Qwen 결과 fallback. VLM enabled 상태에서는 TTC guard가 안전망으로 작동 |
| JSON 파싱 실패 | speed_scale=1.0 fallback, raw_response 로그에 기록 |
| VLM 추론 중 예외 | speed_scale=1.0 fallback |
| TTC 안전 구간 진입 | 즉시 speed_scale=1.0으로 해제 (다음 스텝부터) |
| VLM 응답 아직 없음 (초기) | VLM enabled + risky이면 TTC guard scale 적용 |

---

## 실행 방법

### 필요 패키지

```bash
pip install transformers accelerate qwen-vl-utils pillow
```

### 기본 실행 (VLM 개입 활성화)

```bash
bash /mnt/2/carla_garage/scripts/eval/run_qwen_dev10.sh
```

### 옵션 오버라이드

```bash
# TF++ baseline만 (VLM 없음, 비교용)
QWEN_VLM_ENABLED=0 bash run_qwen_dev10.sh

# TTC 임계값 변경 (2초)
QWEN_TTC_THRESHOLD=2.0 bash run_qwen_dev10.sh

# 다른 출력 경로
OUT_DIR=/mnt/2/carla_metric_result/qwen_ttc2 bash run_qwen_dev10.sh

# GPU 지정
# CUDA_VISIBLE_DEVICES_LIST는 물리 GPU 목록, GPU_RANK/QWEN_VLM_DEVICE는 그 안의 cuda index
CUDA_VISIBLE_DEVICES_LIST=0,1 GPU_RANK=0 QWEN_VLM_DEVICE=cuda:1 bash run_qwen_dev10.sh

# GPU 하나만 보이게 실행한다면 Qwen도 cuda:0이어야 함
CUDA_VISIBLE_DEVICES_LIST=0 GPU_RANK=0 QWEN_VLM_DEVICE=cuda:0 bash run_qwen_dev10.sh

# Qwen3 thinking 모드 (느리지만 더 상세한 reasoning)
QWEN_VLM_THINKING=1 bash run_qwen_dev10.sh

# Qwen 입력 이미지 저장 끄기
QWEN_SAVE_INPUTS=0 bash run_qwen_dev10.sh

# TTC 위험 전에도 25m 이내 객체가 있으면 Qwen critic 호출
QWEN_VLM_PROBE_DISTANCE=25.0 bash run_qwen_dev10.sh

# 50 step마다 shadow/probe 호출
QWEN_VLM_PERIODIC_STEPS=50 bash run_qwen_dev10.sh

# TTC guard 끄기 (Qwen 출력만 실험)
QWEN_TTC_GUARD_ENABLED=0 bash run_qwen_dev10.sh
```

### 결과 위치

```
/mnt/2/carla_metric_result/qwen_dev10/
├── eval.json                          # Bench2Drive 점수
├── eval.log                           # 전체 실행 로그
├── viz/
│   ├── RouteScenario_XXXX_rep0/
│   │   ├── 0001.png ...               # TF++ RGB 시각화
│   │   ├── dashboard/
│   │   │   └── 00001.png ...          # Qwen 대시보드
│   │   ├── qwen_input/
│   │   │   └── 00042.png ...          # Qwen에 실제 투입된 annotated image
│   │   └── qwen_intervention.jsonl    # 스텝별 TTC/VLM 로그
├── videos/                            # 루트별 MP4
└── dashboard_videos/                  # 대시보드 MP4
```

---

## JSONL 로그 형식

```json
{
  "step": 42,
  "ego_speed": 12.3,
  "front_distance": 18.5,
  "ttc": 1.50,
  "is_risky": true,
  "vlm_called": true,
  "vlm_ready": true,
  "vlm_trigger": "ttc",
  "speed_scale": 0.6,
  "guard_scale": 0.8,
  "risk_level": "high",
  "object_count": 4,
  "primary_object_id": 1,
  "reason": "Vehicle ahead braking, collision risk",
  "tfpp_target_speed": 13.88,
  "final_target_speed": 8.33
}
```

---

## 확인해야 할 점

1. **BB format 확인**
   - `center_net.py` 라인 230: `[x, y, w, h, yaw, velocity, brake, class, score]`
   - `bb[0]=x` (전방 거리), `bb[1]=y` (측면 거리) — ego 좌표계
   - `x > 0` = ego 전방

2. **target_speed 반영 위치 확인**
   - `sensor_agent.py:620` `control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)`
   - 패치가 적용된 후: `pred_target_speed_scalar * speed_scale`이 실제로 PID에 전달됨

3. **baseline 동일성 확인**
   ```bash
   # 동일 조건에서 QWEN_VLM_ENABLED=0으로 실행 시 TF++ baseline과 결과 동일해야 함
   QWEN_VLM_ENABLED=0 bash run_qwen_dev10.sh
   ```

4. **GPU 메모리 확인**
   - `CUDA_VISIBLE_DEVICES_LIST=0,1`이면 TF++는 기본 `cuda:0`, Qwen은 기본 `cuda:1`
   - TF++: 약 4-6GB
   - Qwen 8B: 약 16GB fp16
   - 단일 GPU라면 `CUDA_VISIBLE_DEVICES_LIST=0 QWEN_VLM_DEVICE=cuda:0` + 메모리 여유 확인 필요
