# Carla Garage Extension — Architecture Document

> 근거: 코드 직접 분석. 각 주장에 파일·클래스·함수명 및 라인 번호를 명시.  
> 구분 표시: **[코드상 명확함]** / **[추론 가능]** / **[확인 필요]**

---

## 1. 프로젝트 아키텍처 한줄 요약

TF++ (TransFuser++) 기반 자율주행 에이전트(SensorAgent)에 **Alpamayo 1.5 VLM 기반 시맨틱 세이프티 레이어**를 비침습적으로 결합한 **Dual-System VLA** 구조다. TF++가 10 Hz 리얼타임 제어를 담당하고, Alpamayo는 별도 GPU·별도 프로세스에서 비동기 추론하여 "멈춰야 하는가 / 양보해야 하는가" 수준의 시맨틱 힌트를 제공한다. 두 시스템의 **경로 불일치(disagreement)** 가 크고 VLM 신뢰도가 충분할 때만 TF++의 throttle/brake를 덮어쓰며, steer는 절대 수정하지 않는다. 모든 개입은 JSONL로 기록되어 오프라인 분석에 사용된다.

---

## 2. 전체 구조

```text
┌─────────────────────────────────────────────────────────────────────┐
│  CARLA Simulator (10 Hz)                                            │
│    sensor data: rgb_front, rgb_{left,right,back}, lidar, gps, imu  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ input_data
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ExtSensorAgent  (ext_sensor_agent.py)                              │
│  └─ inherits SensorAgent  (team_code/sensor_agent.py, TF++)         │
│                                                                     │
│  [1] ImageEnhancer (optional)                                       │
│       ClassicCVEnhancer: low-light / haze / blur 보정               │
│       → enhanced input_data                                         │
│                                                                     │
│  [2] Upstream TF++ Inference  (super().run_step)                    │
│       Perception → BEV → Planning → PID                             │
│       → base_control (steer, throttle, brake)                       │
│       → pred_wp / pred_checkpoints, pred_target_speed  (captured)   │
│                                                                     │
│  [3] ExtPipeline.run(obs, plan, base_control)                       │
│       │                                                             │
│       ├─ [VLM]  AlpamayoZmqVLM.infer(obs)   (non-blocking)         │
│       │          ZMQ REQ → alpamayo_server.py (별도 프로세스)        │
│       │          → vlm_hint, vlm_hint_ts  (obs.data 에 merge)       │
│       │                                                             │
│       ├─ [RISK] ActionAwareRisk.estimate(obs, plan)                 │
│       │          P1: meta_action × dis_gate                         │
│       │          P2: traj stop/slow intent                          │
│       │          P3: trajectory disagreement                        │
│       │          P4: text fallback                                  │
│       │          → RiskReport(score ∈ [0,1])                       │
│       │                                                             │
│       ├─ [SAFETY] SemanticArbiter.filter(control, risk, obs)        │
│       │          Gate 1–5 → override throttle/brake  (steer 불변)   │
│       │          → final Control                                    │
│       │                                                             │
│       └─ [LOG] _log_intervention → $SAVE_PATH/intervention_log.jsonl│
│                                                                     │
│  [4] control.throttle / brake 갱신 후 CARLA 에 반환                 │
└─────────────────────────────────────────────────────────────────────┘

                   ┌─────────────────────────────────┐
                   │  alpamayo_server.py (별도 GPU)   │
                   │  ZMQ REP socket                  │
                   │  Alpamayo 1.5 + Cosmos VLM        │
                   │  → structured CoT + pred_xyz(64,3)│
                   └─────────────────────────────────┘
```

---

## 3. 각 모듈의 역할

### 3-1. ExtSensorAgent
- **파일:** `src/garage_ext/agents/ext_sensor_agent.py`
- **클래스:** `ExtSensorAgent(SensorAgent)` — `get_entry_point()` 반환 이름 `"ExtSensorAgent"`
- **역할:** Leaderboard 진입점. 업스트림 SensorAgent를 상속하고 Extension 파이프라인을 그 위에 얹음.
- **핵심 동작:**
  - `setup()` [L41]: 루트마다 호출. `GARAGE_EXT_CONFIG` 환경변수로 ExtConfig 로드, ExtPipeline 인스턴스 생성.
  - `_patch_direct_controller_capture()` [L67]: `net.control_pid_direct`를 monkey-patch하여 TF++의 `pred_checkpoints`와 `pred_target_speed`를 가로챔. **업스트림 코드 수정 없이** 구현. **[코드상 명확함]**
  - `run_step()` [L233]: 이미지 보정 → 업스트림 추론 → ExtPipeline 실행 → 제어값 덮어쓰기.
  - `VIZ_ROUTE_LIMIT` [L48]: class-level 카운터 `_route_setup_count`로 앞 N개 루트만 `save_path` 유지, 나머지 `None`으로 null처리.

### 3-2. ExtPipeline
- **파일:** `src/garage_ext/pipeline.py`
- **클래스:** `ExtPipeline`
- **역할:** VLM → Risk → Safety 3단계를 순서대로 실행하는 오케스트레이터.
- **핵심 동작:**
  - `run(obs, plan, base_control)` [L41]: 단계별 실행, `PipelineOutputs(control, risk, vlm_info)` 반환.
  - `_log_intervention()` [L58]: `SAVE_PATH` 환경변수가 설정된 경우 `intervention_log.jsonl`에 step별 JSONL 기록. risk.score > 0 또는 hint 있을 때만 기록. **[코드상 명확함]**

### 3-3. AlpamayoZmqVLM
- **파일:** `src/garage_ext/modules/vlm/alpamayo_zmq.py`
- **클래스:** `AlpamayoZmqVLM`, `@register("vlm", "alpamayo_zmq")`
- **역할:** Alpamayo 서버에 비동기 ZMQ 요청을 보내고 시맨틱 힌트를 캐싱.
- **핵심 동작:**
  - `infer(obs)` [L119]: 매 스텝 호출되지만 non-blocking. 캐시된 hint 반환 + 트리거 조건 체크.
  - `_check_triggers()` [L155]: 교차로 진입 / 곡률 급변(>0.3 rad) / 근거리 액터(<15 m) / 기본 5초 주기 중 하나 충족 시 ZMQ 요청 발사.
  - `_worker_loop()` [L304]: 데몬 스레드에서 ZMQ REQ socket으로 msgpack 요청/응답 처리.
  - FrameBuffer(4 프레임) + PoseBuffer(16 스텝) 유지. **[코드상 명확함]**

### 3-4. ActionAwareRisk
- **파일:** `src/garage_ext/modules/risk/action_aware_risk.py`
- **클래스:** `ActionAwareRisk`, `@register("risk", "action_aware")`
- **역할:** VLM 힌트·TF++ 예측 경로를 종합하여 [0,1] 위험 점수 산출.
- **우선순위 4단계:**

| 우선순위 | 이름 | 조건 | 출력 |
|---------|------|------|------|
| P1 | meta_action | VLM의 `meta_action` (stop/yield/slow_down/…) | `base_score × conf × gate` |
| P2 | traj intent | Alpamayo `traj_analysis.stop_intent / slow_intent` | 0.9 / 0.55 |
| P3 | disagreement | TF++ path ≠ Alpamayo pred_xyz (경로 비교) | 0~1 |
| P4 | text fallback | reasoning / cot 키워드 매칭 | max 0.5 |

- **disagreement gate** [P1]: `gate = min(1.0, dis_score / dis_gate_threshold)`. threshold=0.25이면 dis_score < 0.25일 때 meta_action 영향 줄어듦.
- **death-spiral 방지** [P2]: `tf_target_speed >= 1.5 m/s`일 때만 traj intent 활성. **[코드상 명확함]**
- **_compute_disagreement()** [L165]: 1초·2초 시점의 전진/측면 이동량 차이를 가중합산.
  - 비교 지점: `wp_gru` → `pred_wp[3], pred_wp[7]` / `checkpoint` → speed×1.0s, speed×2.0s
  - 가중치: `0.40×prog_1s + 0.30×prog_2s + 0.20×lat_1s + 0.10×lat_2s`

### 3-5. SemanticArbiter
- **파일:** `src/garage_ext/modules/safety/semantic_arbiter.py`
- **클래스:** `SemanticArbiter`, `@register("safety", "semantic_arbiter")`
- **역할:** 위험 점수 기반 마지막 개입 게이트. throttle/brake만 수정, steer 불변.
- **5단계 게이트:**

```
Gate 1: vlm_hint 존재
Gate 2: hint_age < stale_s (기본 8초)
Gate 3: confidence >= min_confidence (기본 0.4)
Gate 4: consistency_check — stop/yield이면 traj.fwd_10 > 3m일 때 fail
Gate 5: risk.score >= risk_threshold (기본 0.45)
    ↓ 모두 통과
meta_action → 제어 개입
```

- **액션 매핑:**
  - `stop` → brake=1.0, throttle=0
  - `yield` → brake=0.5, throttle=0 (단, lateral clearance yield는 passthrough)
  - `slow_down` → speed_cap 초과 시 brake=0.2; 아니면 throttle=min(0.5, …)
  - `cautious_proceed` → speed_cap 없으면 passthrough; 있으면 throttle×0.5
- **lateral clearance yield** [L233]: "nudge/pass the/stopped vehicle" 류 표현 + static 장애물이면 → TF++ steer에 맡김, 브레이크 개입 없음.

### 3-6. ClassicCVEnhancer
- **파일:** `src/garage_ext/modules/image_enhancer/classic.py`
- **클래스:** `ClassicCVEnhancer`, `@register("image_enhancer", "classic_cv")`
- **역할:** 업스트림 TF++ 모델이 입력을 받기 전에 카메라 프레임 화질 보정.
- **동작:** brightness/saturation/sharpness 분석 → low_light/over_exposed/haze/blurry/normal 분류 → 적합한 CLAHE+Gamma+Unsharp 파이프라인 적용.

### 3-7. alpamayo_server.py
- **파일:** `tools/alpamayo_server.py`
- **역할:** 별도 GPU·별도 Python 환경(venv)에서 ZMQ REP 서버로 구동. Alpamayo 1.5 + Cosmos VLM 모델 로드 후 `structured/traj/vqa` 요청 처리.
- **핵심 함수:** `serve(ipc_path, gpu)` — 메인 루프. `run_inference(req, ...)` — 요청 타입 디스패치. `parse_structured_cot(...)` — CoT에서 JSON 또는 키워드 패턴으로 meta_action 추출.

---

## 4. 모듈 간 연결 방식

### 등록·생성 메커니즘 (`src/garage_ext/registry.py`)

```python
# 등록 (각 모듈 파일 최상단)
@register("risk", "action_aware")
class ActionAwareRisk: ...

# 생성 (pipeline.py, ext_sensor_agent.py)
risk = build("risk", cfg.risk, **cfg.risk_kwargs)
# → _REGISTRY["risk"]["action_aware"](**kwargs) 호출
```

**[코드상 명확함]** `build(kind, name)` 함수가 `_REGISTRY` dict 룩업 후 인스턴스 반환.

### 설정 전달 경로

```
환경변수 GARAGE_EXT_CONFIG
  └─ ext_sensor_agent.py:_load_ext_config()
      └─ config/ext_config.py:load_experiment_config(path)
          └─ YAML extends 체인 로드
              └─ ExtConfig.apply_overlay(data)
                  └─ ExtPipeline(cfg)
                      └─ build("vlm", cfg.vlm, **cfg.vlm_kwargs)
                         build("risk", cfg.risk, **cfg.risk_kwargs)
                         build("safety", cfg.safety, **cfg.safety_kwargs)
```

### 데이터 연결: obs.data 공유 dict

모든 모듈은 `Observation.data` dict를 통해 데이터를 공유한다. **[코드상 명확함]**

```python
# ext_sensor_agent.py:run_step() — obs 생성
obs = Observation(data={
    "input_data": input_data,           # 원본 센서 데이터
    "timestamp": timestamp,
    "agent": self,                      # self 참조 (속도, 상태 등 접근용)
    "tf_pred_wp": ...,                  # TF++ GRU 경로
    "tf_pred_checkpoints": ...,         # TF++ checkpoint 경로
    "tf_pred_target_speed_mps": ...,    # TF++ 목표 속도
    "tf_pred_path": ...,                # 선택된 경로
    "tf_pred_path_kind": ...,           # "wp_gru" or "checkpoint"
})

# pipeline.py:run() — VLM 결과 merge
obs.data.update(vlm_info)  # vlm_hint, vlm_hint_ts, vlm_step 추가

# risk/action_aware_risk.py:estimate() — 읽기
hint = obs.data.get("vlm_hint", {})
tf_pred_path = obs.data.get("tf_pred_path")

# safety/semantic_arbiter.py:filter() — 읽기
hint = obs.data.get("vlm_hint", {})
hint_ts = obs.data.get("vlm_hint_ts", 0.0)
```

---

## 5. 입력/출력

### ExtSensorAgent.run_step() 전체 I/O

| 방향 | 데이터 | 타입 |
|------|--------|------|
| 입력 | `input_data` | dict — `rgb_front: (id, (H,W,4) BGRA)`, `lidar_top`, `gps`, `imu`, `carla_speedometer` |
| 입력 | `timestamp` | float (game time) |
| 출력 | `control` | `carla.VehicleControl(steer, throttle, brake)` |

### AlpamayoZmqVLM.infer() I/O

| 방향 | 데이터 |
|------|--------|
| 입력 | `obs` — `input_data` 내 카메라 프레임, 속도계; `agent` 참조로 GPS/pose |
| 내부 | FrameBuffer(N_cam, 4 JPEG) + PoseBuffer(16, xyz+rot) → ZMQ 전송 |
| 출력 | `{"vlm_hint": {...}, "vlm_hint_ts": float, "vlm_step": int}` |

### vlm_hint 구조

```python
{
    "meta_action": "proceed" | "cautious_proceed" | "slow_down" | "yield" | "stop",
    "hazard_type": "none" | "pedestrian" | "vehicle" | "construction" | "occlusion",
    "confidence": float,          # 0..1
    "expiry_sec": float,
    "target_speed_cap_mps": float | None,
    "reasoning": str,
    "cot": str,                   # chain-of-thought 전문
    "traj_analysis": {
        "stop_intent": bool,      # pred_xyz fwd_10 < 0.5 m
        "slow_intent": bool,      # pred_xyz fwd_10 < 2.0 m
        "fwd_10": float,          # 전진 거리 (m)
        "lat_10": float,          # 측면 이동 (m)
        "lateral_dir": "none" | "left" | "right",
    },
    "pred_xyz": list,             # (64, 3) Alpamayo 예측 경로 (로컬 좌표)
}
```

### ActionAwareRisk.estimate() I/O

| 방향 | 데이터 |
|------|--------|
| 입력 | `obs.data["vlm_hint"]`, `obs.data["tf_pred_path"]`, `obs.data["tf_pred_target_speed_mps"]` |
| 출력 | `RiskReport(score: float, reasons: list[str], extra: dict)` |

### SemanticArbiter.filter() I/O

| 방향 | 데이터 |
|------|--------|
| 입력 | `control` (TF++ 원본), `risk` (RiskReport), `obs` (vlm_hint 포함) |
| 출력 | `Control(steer=unchanged, throttle=?, brake=?, meta={"override_reason": ...})` |

---

## 6. 기존 시스템과의 결합 방식

### 결합 지점 1: 에이전트 진입점 교체 (비침습적)

```
기존: leaderboard --agent=team_code/sensor_agent.py
현재: leaderboard --agent=src/garage_ext/agents/ext_sensor_agent.py
```

Leaderboard가 `get_entry_point() → "ExtSensorAgent"` 호출. 업스트림 코드 수정 없음. **[코드상 명확함]**

### 결합 지점 2: monkey-patch로 TF++ 내부값 캡처

**파일:** `ext_sensor_agent.py:_patch_direct_controller_capture()` [L67–90]

```python
original = net.control_pid_direct  # TF++ 내부 메서드

def capture_wrapper(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
    agent_self._capture_tf_direct_predictions(pred_checkpoints, pred_target_speed)
    return original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)  # 원본 동작 보존

net.control_pid_direct = capture_wrapper  # 교체
```

**업스트림 코드 무수정**으로 TF++의 내부 예측값(`pred_checkpoints`, `pred_target_speed`)을 추출. **[코드상 명확함]**

### 결합 지점 3: run_step 후처리 (제어값 덮어쓰기)

**파일:** `ext_sensor_agent.py:run_step()` [L260–264]

```python
control = super().run_step(...)  # TF++ 제어값 (carla.VehicleControl)
out = self._ext_pipeline.run(obs, plan_obj, base_control)
control.steer = out.control.steer      # steer는 pipeline이 변경하지 않음
control.throttle = out.control.throttle
control.brake = out.control.brake
return control  # CARLA 시뮬레이터 반환
```

TF++의 steer 결정은 항상 유지. throttle/brake만 SemanticArbiter가 수정 가능. **[코드상 명확함]**

### 결합 지점 4: 이미지 보정 (업스트림 전처리)

**파일:** `ext_sensor_agent.py:run_step()` [L238]

```python
input_data = self._apply_image_enhancement(input_data)  # 먼저 보정
control = super().run_step(input_data, ...)              # TF++는 보정된 이미지를 봄
```

TF++가 보정된 이미지를 받도록 순서 보장. **[코드상 명확함]**

---

## 7. 개입(Intervention): 언제, 어떤 조건에서, 무엇을 바꾸는가

### 개입 전체 조건 (AND 조건)

```
① AlpamayoZmqVLM이 vlm_hint를 캐시 중
② hint_age < stale_s (기본 8초)
③ hint.confidence >= min_confidence (기본 0.4)
④ consistency_check 통과
   - stop/yield이면 traj.fwd_10 <= 3m 이어야 함
   - (Alpamayo가 "멈춰라" 하는데 TF++ 경로가 계속 전진하면 inconsistent)
⑤ risk.score >= risk_threshold (기본 0.45)
   - risk.score는 ActionAwareRisk가 산출한 [0,1] 점수
```

### risk.score 산출 메커니즘

```
P1 score = meta_action_score × confidence × dis_gate
           (dis_gate = min(1, dis_score / 0.25)
            → dis_score 낮으면 gate 줄어들어 P1 억제)

P2 score = 0.9 (stop_intent) or 0.55 (slow_intent)
           단, tf_target_speed >= 1.5 m/s 일 때만 활성

P3 score = disagreement(TF++ path, Alpamayo pred_xyz)
           0.40×prog_gap_1s + 0.30×prog_gap_2s + 0.20×lat_gap_1s + 0.10×lat_gap_2s

P4 score <= 0.5 (text keyword fallback)

final = max(P1, P2, P3, P4)
```

### 개입 시 제어값 변화

| meta_action | throttle | brake | steer |
|-------------|----------|-------|-------|
| `stop` | 0 | max(기존, **1.0**) | 불변 |
| `yield` (동적) | 0 | max(기존, **0.5**) | 불변 |
| `yield` (lateral clearance) | **passthrough** | 불변 | 불변 |
| `slow_down` (speed_cap 초과) | 0 | max(기존, **0.2**) | 불변 |
| `slow_down` (speed_cap 미만) | min(기존, **0.5**) | 불변 | 불변 |
| `cautious_proceed` (cap 있음) | 기존 × **0.5** | 불변 | 불변 |
| `cautious_proceed` (cap 없음) | **passthrough** | 불변 | 불변 |
| `proceed` / 기타 | **passthrough** | 불변 | 불변 |

### 개입이 발생하지 않는 경우 (passthrough)

- `GARAGE_EXT_CONFIG` 환경변수 미설정 → `ExtPipeline` 미생성, TF++ 단독 동작
- VLM 힌트 없음 (아직 첫 추론 전, 또는 timeout 중)
- hint 오래됨 (>8초)
- confidence 낮음 (<0.4)
- consistency fail (Alpamayo "멈춰" + TF++ fwd_10 > 3m)
- risk.score 낮음 (<0.45)
- meta_action이 `proceed`

---

## 8. 실제 실행 흐름: 데이터와 제어의 이동

### 스텝별 흐름 (10 Hz)

```
t=0  CARLA → input_data (rgb×4 + lidar + gps + imu + speedometer)
         │
t=1  ClassicCVEnhancer.enhance(bgr) per camera
         │ enhanced input_data
t=2  SensorAgent.run_step(enhanced)   [TF++ BEV perception + waypoint GRU + PID]
         │ → pred_wp (8,2) or pred_checkpoints (N,2)
         │ → pred_target_speed (float)
         │ → base_control (steer, throttle, brake)
         │
         ├─ [capture] net.control_pid_direct wrapper
         │    → self._tf_pred_checkpoints, self._tf_pred_target_speed_mps
         │
t=3  obs = Observation({input_data, timestamp, agent, tf_pred_*, ...})
         │
t=4  AlpamayoZmqVLM.infer(obs)   [non-blocking]
         │ 트리거 조건 충족 시:
         │   FrameBuffer.update() → (2 cam, 4 JPEG)
         │   PoseBuffer.update()  → (16, xyz+rot)
         │   ZMQ REQ → alpamayo_server
         │     → Alpamayo 1.5 structured inference
         │     → parse_structured_cot → meta_action
         │     → analyze_trajectory → stop/slow_intent, fwd_10
         │   ZMQ REP ← {meta_action, pred_xyz, traj_analysis, confidence, ...}
         │ → 캐시 업데이트, obs.data에 vlm_hint merge
         │
t=5  ActionAwareRisk.estimate(obs, plan)
         │ hint = obs.data["vlm_hint"]
         │ tf_path = obs.data["tf_pred_path"]
         │ alp_path = hint["pred_xyz"]
         │
         │ P3: _compute_disagreement(tf_path, alp_path)
         │     → prog_gap_1s, prog_gap_2s, lat_gap_1s, lat_gap_2s
         │     → dis_score (가중합)
         │
         │ P1: base_score × conf × min(1, dis_score/0.25)
         │ P2: 0.9 if stop_intent (and tf_speed >= 1.5)
         │ P4: keyword score (cap 0.5)
         │
         │ → RiskReport(score=max(P1,P2,P3,P4), reasons=[...])
         │
t=6  SemanticArbiter.filter(base_control, risk, obs)
         │ Gate 1–5 체크
         │ 모두 통과 시: meta_action → throttle/brake 수정
         │ → Control(steer=unchanged, throttle=?, brake=?, meta={override_reason})
         │
t=7  control.throttle = out.control.throttle
     control.brake    = out.control.brake
     control.steer    = out.control.steer   (TF++ 값 그대로)
         │
t=8  ExtPipeline._log_intervention()
         │ → $SAVE_PATH/intervention_log.jsonl  (risk.score > 0 or hint 있을 때)
         │
t=9  return control → CARLA Simulator
```

### VLM 추론 타이밍 (비동기)

```
CARLA step 0  → _check_triggers() → fire_event.set()
                   ↓  (daemon thread)
                 ZMQ REQ 전송 (msgpack)
                   ↓  (~0.5~3 s 소요, 시뮬레이터 블로킹 없음)
                 ZMQ REP 수신
                 hint 캐시 업데이트

CARLA step 1  → infer() → 캐시된 hint 반환 (이전 결과)
CARLA step 2  → infer() → 캐시된 hint 반환
...
CARLA step N  → 캐시 갱신된 hint 반환 (새 추론 결과)
```

VLM 추론이 느려도 시뮬레이터 루프는 10 Hz 유지. 힌트는 `stale_s=8초` 이내면 유효. **[코드상 명확함]**

---

## 9. Ablation 설정 비교

| 설정 | YAML | VLM | disagreement gate | 목적 |
|------|------|-----|------------------|------|
| **A** (baseline) | `GARAGE_EXT_CONFIG` 미설정 | 없음 | 없음 | TF++ 단독 |
| **B** (bool-only) | `alpamayo_bool_only.yaml` | alpamayo_zmq | `dis_gate_threshold=0.0` (비활성) | trajectory bool만 |
| **C** (제안 방법) | `alpamayo_structured.yaml` | alpamayo_zmq | `dis_gate_threshold=0.25` | 완전한 dual-system |

---

## 10. 로그 파일

| 파일 | 생성 위치 | 내용 |
|------|-----------|------|
| `intervention_log.jsonl` | `$SAVE_PATH/intervention_log.jsonl` | 스텝별 VLM hint, risk score, override 여부, 제어값 변화 |
| `alpamayo_shadow.jsonl` | `$SAVE_PATH/alpamayo_shadow.jsonl` | VLM 추론 트리거 사유, 요청 내용, 응답 CoT [추론 가능] |
| `enhance_compare/*.png` | `$SAVE_PATH/{route_tag}/enhance_compare/` | 원본/보정 비교 이미지 (4스텝마다 저장) |
| `{route_tag}/*.png` | `$SAVE_PATH/{route_tag}/` | CARLA BEV 시각화 (DEBUG_CHALLENGE=1, VIZ_ROUTE_LIMIT 이내) |

---

## 11. 핵심 파일 목록

```
src/garage_ext/
├── agents/ext_sensor_agent.py       ← 진입점, 결합 지점 집합소
├── pipeline.py                      ← VLM→Risk→Safety 오케스트레이션
├── registry.py                      ← 모듈 등록/팩토리
├── config/ext_config.py             ← ExtConfig, YAML 로더
├── modules/base.py                  ← Observation / Plan / Control / RiskReport 정의
├── modules/vlm/alpamayo_zmq.py      ← 비동기 ZMQ VLM 클라이언트
├── modules/risk/action_aware_risk.py ← 4-priority 위험 점수 산출
├── modules/safety/semantic_arbiter.py ← 5-gate 제어 개입
├── modules/image_enhancer/classic.py  ← 전처리 보정
├── buffers/frame_buffer.py          ← 카메라 4프레임 롤링 버퍼
└── buffers/pose_buffer.py           ← 에고 pose 16스텝 롤링 버퍼

tools/alpamayo_server.py             ← 별도 프로세스 VLM 서버 (ZMQ REP)
configs/experiments/alpamayo_structured.yaml  ← 제안 방법 설정
configs/experiments/alpamayo_bool_only.yaml   ← Ablation B 설정
```
