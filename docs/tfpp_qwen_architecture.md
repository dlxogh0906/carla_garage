# TF++ + Qwen VLM 결합 구조

> **대상**: 팀 내부 소개용  
> **작성 기준**: `qwen_sensor_agent.py`, `qwen_client.py`, `qwen_input.py` 코드 기준  

---

## 1. 전체 구조 개요

```
CARLA 시뮬레이터 (10 Hz)
    │
    ▼  센서 데이터 (RGB 카메라 4개, LiDAR, GPS/IMU, 속도계)
┌───────────────────────────────────────────────────────────────┐
│  QwenSensorAgent  (qwen_sensor_agent.py)                      │
│                                                               │
│  ① TF++ 실행  ────────────────────────────────────────────►  │
│     • 3D bbox 감지  (bb_buffer)                               │
│     • 경로 계획    (pred_checkpoints)                          │
│     • 목표속도 계산 (pred_target_speed)                        │
│     • 제어 출력   (throttle / brake / steer)                   │
│                                                               │
│  ② 보조 계산  (매 스텝)                                         │
│     • 전방 거리, TTC, 곡률, 신호등 bbox 수 등                   │
│                                                               │
│  ③ Qwen 호출 조건 판단 (트리거)  ─────────────────────────►    │
│     (조건 충족 시 background thread에 요청 전송)                │
│                                                               │
│  ④ 최신 Qwen 결과 적용 (non-blocking)                          │
│     • speed_scale 계산                                        │
│     • TF++ pred_target_speed × speed_scale 적용               │
│                                                               │
│  ⑤ 로그 + 대시보드 저장                                        │
└───────────────────────────────────────────────────────────────┘
    │
    ▼  최종 제어 명령 → CARLA

────────────────────────────────────────────────────────────────
                                        ▲  결과 캐시 (비동기)
                                        │
                                ┌───────────────┐
                                │  Qwen 백그라운드 │
                                │  daemon thread  │
                                │  (1 ~ 5 초)     │
                                └───────────────┘
```

**핵심 설계 원칙**: Qwen 추론(1~5초)이 10Hz CARLA 제어 루프를 절대 블록하지 않는다.  
TF++는 매 스텝 정상 실행되고, Qwen 결과는 완료된 최신 캐시에서만 읽는다.

---

## 2. TF++ 입출력

### 2-1. TF++ 입력 (CARLA 센서)

| 센서 키 | 타입 | 설명 |
|---------|------|------|
| `rgb_front` | `np.ndarray (H×W×4, BGRA)` | 전방 카메라 (주요 입력) |
| `rgb_left`, `rgb_right`, `rgb_back` | `np.ndarray` | 측면·후방 카메라 |
| `lidar` | point cloud | 3D 포인트 클라우드 |
| `gps` | `[lat, lon, alt]` | GPS 위치 |
| `imu` | acceleration + gyro | 관성 측정 |
| `speed` / `speedometer` | `{"speed": float}` | 현재 속도 (m/s) |
| route waypoints | XML route config | 목표 경로 웨이포인트 |

### 2-2. TF++ 출력 (QwenSensorAgent에서 사용)

| 출력 | 변수명 | 타입 | 설명 |
|------|--------|------|------|
| 제어 명령 | `control` | `CarlaEgoVehicleControl` | throttle / brake / steer (0~1, -1~1) |
| 3D 바운딩박스 | `bb_buffer[-1]` | `List[array]` | 감지된 객체들 (자세한 포맷은 아래) |
| 경로 체크포인트 | `_captured_tfpp_checkpoints` | `np.ndarray (N×2)` | 계획된 경로 (x\_앞쪽m, y\_오른쪽m, 자아 좌표계) |
| 목표 속도 | `_captured_tfpp_target_speed` | `float` | TF++가 요청한 속도 (m/s), Qwen 개입 전 원본 |
| 최종 속도 | `_captured_final_target_speed` | `float` | speed\_scale 적용 후 실제 사용 속도 |

#### TF++ 3D bbox 포맷 (bb\[i\])

```
인덱스  값
 [0]   x        — 자아 기준 전방 거리 (m, 양수=앞)
 [1]   y        — 자아 기준 우측 거리 (m, 양수=오른쪽)
 [2]   width    — 폭 (m)
 [3]   length   — 길이 (m)
 [4]   yaw      — 방향각 (rad)
 [5]   speed    — 객체 속도 (m/s)
 [6]   (미사용)
 [7]   class_id — 0:vehicle 1:pedestrian 2:traffic_light 3:stop_sign 4:emergency_vehicle
 [8]   score    — 감지 신뢰도 (0~1)
```

---

## 3. TF++ → Qwen 데이터 흐름

Qwen 호출 시 TF++ 출력에서 다음 6가지를 추출·가공해서 전달한다.

```
TF++ 출력                          가공 함수                Qwen 입력
─────────────────────────────────────────────────────────────────────
bb_buffer[-1] (전체 bbox)
  ├─ x, y, class_id, score  ──► build_object_context()  ──► object_table  (텍스트)
  │                              format_object_table()
  │
  ├─ traffic_light / stop_sign ► summarize_bbox_classes() ─► rule_context  (텍스트)
  │   개수 + 위치                 format_rule_context()
  │
  └─ (모든 bbox)  ──────────────► annotate_qwen_image()  ──► 전방 카메라 이미지
       + rgb_front 카메라              BEV inset 합성            (어노테이션됨)

pred_checkpoints  ───────────────► format_path_summary()  ──► path_summary  (텍스트)

pred_target_speed ───────────────► _format_speed()        ──► tfpp_target_speed (텍스트)

ego_speed (속도계)
front_distance (bbox에서 계산)  ─► compute_sensitive_ttc() ─► ego_speed, front_distance,
TTC (추론)                                                      ttc, ttc_source (수치)

ttc_history (deque 20개)  ───────► format_ttc_history()   ──► ttc_history  (텍스트)
```

---

## 4. Qwen 입력 상세

### 4-1. 이미지 입력

`rgb_front` 카메라에 3가지 레이어를 오버레이한 단일 RGB 이미지:

| 레이어 | 내용 |
|--------|------|
| **원본** | CARLA 전방 카메라 (BGRA→RGB 변환) |
| **자아 주행 코리도 힌트** | 반투명 황금색 삼각형 — TF++가 주행할 예상 영역 |
| **헤더 텍스트** | 현재 속도 / 전방 거리 / TTC / TF++ 목표속도 / 신호등·정지표지 개수 |
| **BEV inset** (우하단) | 객체들을 하향 시점 좌표계로 표시 (빨강=주 위험물, 노랑=같은 차선, 회색=기타) |

### 4-2. 텍스트 프롬프트 구성 요소

```
항목                  내용 예시
──────────────────────────────────────────────────────────────
ego_speed_text        "12.3 m/s (44.3 km/h)"
front_distance_text   "18.50 m"
ttc_text              "1.50 s"
ttc_source            "bbox" | "planner_proxy" | "none"
tfpp_target_speed     "8.20 m/s (29.5 km/h)"

object_table          id | type | x_front_m | y_right_m | same_lane | primary | speed_mps | score
                       2 | vehicle | 18.5 | 0.3 | yes | yes | 0.00 | 0.92
                       5 | traffic_light | 32.1 | -1.2 | no | no | n/a | 0.85

rule_context          TF++ rule-object summary:
                      - traffic_light_count: 1
                      - stop_sign_count: 0
                      - nearest_traffic_light: id=5, x=32.10m, y=-1.20m, score=0.85

path_summary          checkpoint | x_front_m | y_right_m
                      0 | 5.2 | 0.1
                      1 | 10.8 | 0.3
                      2 | 16.5 | 0.8
                      ...

ttc_history           step | ego_speed_mps | front_dist_m | ttc_s | source
                      53 | 12.6 | 18.5 | 1.47 | bbox
                      54 | 12.8 | 16.2 | 1.27 | bbox
```

### 4-3. 프롬프트 모드 두 가지

| 모드 | 언제 사용 | 판단 대상 |
|------|----------|---------|
| `speed` | TTC 위험, 전방 물체 근접, 일반 장애물 | 속도 줄여야 하는가? |
| `traffic_rule` | 신호등/정지표지 감지, 교차로 진입, 주기적 rule 체크 | 빨간불/정지표지 위반인가? |

---

## 5. Qwen 호출 트리거 조건

매 스텝 아래 조건 중 하나라도 참이면 Qwen에 요청을 전송한다.  
단, 이전 추론이 진행 중이거나 `min_interval` 미경과 시 skip.

| 트리거 이름 | 조건 | 프롬프트 모드 | min_interval |
|------------|------|-------------|-------------|
| `traffic_rule` | bbox에 신호등/정지표지 존재, 또는 주기적 rule 체크 (매 20스텝), 또는 rule\_hold 중 폴링 | `traffic_rule` | 0.5s (hold) / **0.8s** (TL visible) / 2.0s (일반) |
| `ttc` | TTC < 3.0s (risky) | `speed` | 2.0s |
| `proximity` | 전방 거리 < 35m | `speed` | 2.0s |
| `object` | 감지된 객체 수 > 0 & 속도 > 0.5m/s | `speed` | 2.0s |
| `curve` | 경로 곡률 ≥ 0.25 rad (교차로/회전 예상) | `traffic_rule` | 2.0s |
| `periodic` | 매 N스텝 (기본값: 비활성화) | `speed` | 2.0s |

> **곡률 계산**: TF++ 체크포인트 앞 4개 점의 방향 변화량 합산.  
> 직선 ≈ 0 rad, 완만한 커브 ≈ 0.1~0.2 rad, 교차로 좌/우회전 ≈ 0.3~0.8 rad

---

## 6. Qwen 출력 상세

### 6-1. Speed 모드 응답

```json
{
  "intervene": true,
  "risk_level": "high",
  "speed_scale": 0.4,
  "primary_hazard_id": 2,
  "path_blocked": true,
  "tfpp_plan_safe": false,
  "hazard_type": "vehicle",
  "reason": "Vehicle stopped in ego corridor at 18m"
}
```

| 필드 | 타입 | 의미 |
|------|------|------|
| `intervene` | bool | 개입 여부 |
| `risk_level` | str | low / medium / high / critical |
| `speed_scale` | float 0~1 | 목표속도 배율 (1.0=유지, 0.0=정지) |
| `primary_hazard_id` | int\|null | 주요 위험 객체 id (bbox 테이블 기준) |
| `path_blocked` | bool | 주행 경로 차단 여부 |
| `hazard_type` | str | vehicle / pedestrian / traffic\_light / stop\_sign / obstacle |
| `reason` | str | 한 문장 이유 |

### 6-2. Traffic Rule 모드 응답

```json
{
  "rule_intervene": true,
  "rule_type": "red_light",
  "traffic_light_state": "red",
  "stop_sign_visible": false,
  "relevant_to_ego": true,
  "confidence": 0.92,
  "speed_scale": 0.0,
  "reason": "Red light ahead relevant to ego lane"
}
```

| 필드 | 타입 | 의미 |
|------|------|------|
| `rule_intervene` | bool | 규칙 위반 개입 여부 |
| `rule_type` | str | none / red\_light / yellow\_light / stop\_sign / unknown |
| `traffic_light_state` | str | red / yellow / green / unknown / not\_visible |
| `relevant_to_ego` | bool | 자아 차선에 해당하는 신호인지 |
| `confidence` | float 0~1 | 판단 신뢰도 |
| `speed_scale` | float | 빨간불/정지표지 → 0.0, 초록불 → 1.0 |

---

## 7. Speed Scale 적용 파이프라인

TF++의 `control_pid_direct()` 함수를 래핑(monkey-patch)하여 속도를 조절한다.

```
TF++ pred_target_speed
        │
        │  × speed_scale
        ▼
control_pid_direct(scaled_speed)  →  throttle / brake / steer

speed_scale 결정 순서 (낮을수록 우선):
  1. 기본값: 1.0
  2. VLM disabled → 1.0 고정
  3. TTC 위험 없음 + semantic 결과 없음 → 1.0
  4. TTC 위험 → ttc_guard_scale (TTC 값에 따라 0.0~0.8)
  5. Qwen speed 결과 (intervene=true) → Qwen 제안 scale
  6. rule_active (신호등/정지표지 판단) → rule_speed_scale
  7. rule_hold_active (빨간불 홀드) → 0.0 (완전 정지)
  8. tl_prestop (신호등 bbox 보이고 판단 미완료) → min(scale, 0.45)
```

---

## 8. Rule Hold 메커니즘

빨간불 판단 후 차가 멈추고, 초록불 확인 전까지 정지를 유지하는 래치 구조.

```
Qwen: "red_light, relevant_to_ego=True, conf≥0.75"
        │
        ▼
rule_hold_active = True  →  speed_scale = 0.0 (완전 정지)
        │
        ├─ 해제 조건 1: Qwen이 "green + relevant + conf≥0.75" 확인 2회 연속
        ├─ 해제 조건 2: 모호한 응답(not_visible 등) 3회 연속 (no-stop vote)
        ├─ 해제 조건 3: 정지 후 150스텝 초과 (safety fallback)
        └─ 해제 조건 4: rule_hold_max_steps 초과 (선택적)
```

Hold 중 Qwen 폴링 간격: **0.5초** (일반 2.0초보다 빠름)

---

## 9. 비동기 추론 구조

```
Control Loop (10 Hz, main thread)
    │
    ├─ run_step() 매 100ms 호출
    │       │
    │       ├─ TF++ 실행 (동기)
    │       ├─ 트리거 판단
    │       ├─ req_queue.put_nowait(request)  ← 큐가 꽉 차면 그냥 skip
    │       ├─ get_latest() → 캐시된 이전 결과 읽기 (항상 즉시 반환)
    │       └─ speed_scale 적용 후 제어 출력
    │
    └─ Qwen Worker Thread (daemon, 1개)
            │
            ├─ req_queue.get() → 대기
            ├─ 모델 추론 (1~5초)
            │       이미지 전처리 → tokenize → generate → parse JSON
            └─ self._cached 업데이트 (lock 보호)
```

**큐 크기 1**: 추론 중 새 요청은 자동 drop → 항상 최신 프레임 기준으로 판단

---

## 10. 실험 설정 요약

### 주요 환경변수 (run_qwen_dev10.sh 기준)

| 변수 | 기본값 | 의미 |
|------|--------|------|
| `QWEN_VLM_ENABLED` | 1 | VLM 개입 on/off |
| `QWEN_MODEL` | `Qwen3-VL-8B-Instruct` | 모델 경로 |
| `QWEN_VLM_DEVICE` | `cuda:1` (2GPU) | Qwen GPU (TF++와 분리) |
| `QWEN_TTC_THRESHOLD` | 3.0s | TTC 위험 임계값 |
| `QWEN_RULE_CONFIDENCE_THRESH` | 0.75 | 신호등 판단 최소 신뢰도 |
| `QWEN_RULE_HOLD_SAFETY_STEPS` | 150 | hold 최대 유지 스텝 (약 15초) |
| `QWEN_TL_PRESTOP_SCALE` | 0.45 | 신호등 bbox 보일 때 선제 속도 제한 배율 |
| `QWEN_TL_PROBE_MIN_INTERVAL_S` | 0.8s | 신호등 가시 시 빠른 폴링 간격 |
| `QWEN_CURVE_PROBE_THRESH` | 0.25 rad | 곡률 기반 교차로 선제 쿼리 임계값 |
| `QWEN_VLM_THINKING` | 0 | Qwen3 chain-of-thought 비활성화 (속도 우선) |

---

## 11. Alpamayo p3 vs Qwen 비교

| 항목 | Alpamayo p3 | Qwen (현재) |
|------|------------|------------|
| VLM 모델 | Alpamayo 1.5 (사내) | Qwen3-VL-8B-Instruct |
| 추론 방식 | ZMQ IPC (외부 서버) | 동일 프로세스 내 daemon thread |
| 교차로 쿼리 트리거 | 경로 곡률 (`curvature_thresh: 0.3`) | **곡률 프로브 구현** (`QWEN_CURVE_PROBE_THRESH: 0.25`) |
| 인접 차량 트리거 | actor 거리 15m 이내 | object probe (감지 객체 있으면 발동) |
| 측면 lane change 인식 | trajectory rollout (공간 충돌 예측) | BEV inset + object table (간접 추론) |
| 신호등 선제 감속 | `caution_throttle_scale: 0.5` | **TL prestop (scale 0.45)** |
| dev10 평균 DS | **85.2** | 진행 중 |
