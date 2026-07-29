# TF++ + Alpamayo 1.5 결합 구조

## 1. 한 줄 요약

> **TF++가 기본 주행을 담당하고, Alpamayo는 위험한 장면에서만 호출되어 개입 여부를 결정한다.**

---

## 2. 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         매 스텝 (10Hz)                           │
│                                                                   │
│  카메라 / LiDAR / GPS ──→  TF++  ──→  base_control              │
│                              │          (throttle/brake/steer)    │
│                              │                                    │
│                         pred_wp (8개, 2초 미래 경로)              │
│                              │                                    │
│                              ▼                                    │
│                    ┌─── ExtPipeline ───┐                         │
│                    │                   │ ← Alpamayo 캐시된 hint  │
│                    │  위험도 계산       │   (1~3초마다 갱신)      │
│                    │  개입 여부 결정   │                          │
│                    └─────────┬─────────┘                         │
│                              │                                    │
│                              ▼                                    │
│                       final_control ──→ CARLA 시뮬레이터         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 각 모델의 역할과 입출력

### TF++ (TransFuser++)
**역할:** 기본 주행 담당. 매 스텝(10Hz) 실행.

| | 내용 |
|--|------|
| **입력** | 멀티뷰 RGB 카메라 + LiDAR BEV + GPS 목표점 |
| **출력** | `throttle`, `brake`, `steer` (즉시 실행 제어값) |
| **부산물** | `pred_wp`: 앞으로 2초간 8개의 예측 경로점 (x, y, ego 좌표계) |

```
pred_wp = [
  [1.2, 0.0],   # 0.25초 후 위치
  [2.4, 0.1],   # 0.50초 후 위치
  ...
  [9.6, 0.3],   # 2.00초 후 위치  (총 8개)
]
```

---

### Alpamayo 1.5
**역할:** 위험 장면에서만 호출되는 느린 추론기. 백그라운드 스레드에서 실행.

| | 내용 |
|--|------|
| **입력** | 멀티뷰 카메라 4프레임 + 과거 16스텝 자차 이동 이력 (xyz + 회전행렬) |
| **출력 ①** | `pred_xyz`: 앞으로 6.4초간 64개의 예측 경로점 (x, y, z, ego 좌표계) |
| **출력 ②** | `meta_action`: 행동 결정 (`stop` / `yield` / `slow_down` / `cautious_proceed` / `proceed`) |
| **출력 ③** | `confidence`: 확신도 (0~1) |
| **출력 ④** | `hazard_type`: 위험 유형 (`vehicle` / `pedestrian` / `construction` 등) |
| **출력 ⑤** | `reasoning`: 한 문장 설명 |

```
pred_xyz = [
  [0.15, 0.0, 0.0],   # 0.1초 후
  [0.30, 0.0, 0.0],   # 0.2초 후
  ...
  [0.50, 0.1, 0.0],   # 1.0초 후  (index 9)
  ...
  [1.20, 0.3, 0.0],   # 2.0초 후  (index 19)
  ...                  # 총 64개 (6.4초)
]

meta_action = "yield"
confidence  = 0.75
reasoning   = "Stopped vehicle blocking our lane"
```

---

## 4. Alpamayo는 언제 호출되는가?

매 스텝 호출하면 너무 느리기 때문에 4가지 조건 중 하나를 만족할 때만 비동기로 호출합니다.

| 트리거 | 조건 |
|--------|------|
| 명령 전환 | 교차로 접근 등 주행 명령이 바뀔 때 |
| 급커브 | TF++ 예측 경로 곡률 > 0.3 rad |
| 근접 물체 | 15m 이내 차량/보행자 감지 |
| 기본 주기 | 마지막 호출로부터 5초 경과 |

→ 평균 **1~3Hz**로 동작하고, 결과는 캐시되어 다음 호출 전까지 재사용됩니다.

---

## 5. 핵심: Trajectory Disagreement

Alpamayo의 출력(`pred_xyz`)과 TF++의 출력(`pred_wp`)을 **같은 시점에서 비교**하여
"두 모델이 얼마나 다른 미래를 예측하는가"를 수치화합니다.

```
비교 시점: 1초 후, 2초 후

TF++ pred_wp[3]  ← 1초 후 TF++ 예측 위치  (0.25s × 4 = 1.0s)
Alp pred_xyz[9]  ← 1초 후 Alp 예측 위치  (0.10s × 10 = 1.0s)

전방 진행 차이 (prog_gap):
  TF++ 1초 후 x=6.0m  vs  Alp 1초 후 x=1.5m
  → prog_gap_1s = 4.5m  (TF++는 빠르게 가려는데 Alp는 거의 안 가려 함 → 위험!)

횡방향 회피 차이 (lat_gap):
  TF++ 횡방향 y=0.1m  vs  Alp 횡방향 y=1.4m
  → lat_gap_1s = 1.3m  (Alp가 옆으로 피하려 함 → 위험!)
```

**Disagreement Score 계산:**
```
disagreement_score = (
  0.40 × max(0, prog_gap_1s - 1.0m) / 5.0   +   # 1초 전진 차이 (1m 이상부터)
  0.30 × max(0, prog_gap_2s - 1.0m) / 10.0  +   # 2초 전진 차이
  0.20 × max(0, lat_gap_1s) / 2.0            +   # 1초 횡방향 차이
  0.10 × max(0, lat_gap_2s) / 3.0                # 2초 횡방향 차이
) × 0.5  (disagreement_weight)
```

---

## 6. 위험도 계산 (4단계 우선순위)

| 우선순위 | 신호 | 조건 | 위험 점수 |
|----------|------|------|-----------|
| **P1** | meta_action | JSON 파싱 성공 or 확신도 ≥ 0.60 | `meta_score × confidence` |
| **P2** | 궤적 bool | Alp 1초 전진 < 0.5m → stop / < 2.0m → slow | 0.9 / 0.55 |
| **P3** | **Disagreement** | TF++–Alp 경로 차이가 임계값 초과 | `dis_score` (0~0.5) |
| **P4** | 텍스트 키워드 | reasoning에 위험 키워드 존재 | 최대 0.5 (cap) |

meta_action별 기본 위험 점수:

| meta_action | 기본 점수 |
|-------------|----------|
| `stop` | 1.0 |
| `yield` | 0.85 |
| `slow_down` | 0.60 |
| `cautious_proceed` | 0.35 |
| `proceed` | 0.0 |

---

## 7. 개입 결정 (5단계 게이트)

위험 점수가 계산되어도 아래 5개 조건을 모두 통과해야 실제로 개입합니다.

```
Gate 1: Alpamayo 결과(hint)가 존재하는가?
Gate 2: hint가 8초 이내로 신선한가?             ← 너무 오래된 결과는 무시
Gate 3: confidence ≥ 0.4 인가?                ← 불확실한 결과는 무시
Gate 4: meta_action과 trajectory가 모순 없는가?  ← "멈춰라"인데 경로는 앞으로 가면 무시
Gate 5: 위험 점수 ≥ 0.45 인가?                ← 임계값 이상일 때만 개입
         ↓ (모두 통과)
개입 실행
```

---

## 8. 개입 시 무엇을 바꾸는가?

**steer(조향)는 절대 수정하지 않습니다.** throttle과 brake만 조정합니다.

| meta_action | 제어 변화 |
|-------------|----------|
| `stop` | brake = 1.0, throttle = 0 |
| `yield` | brake = max(현재값, 0.5), throttle = 0 |
| `slow_down` | throttle = min(현재값, 0.5) |
| `cautious_proceed` | throttle × 0.5 |
| `proceed` | 변화 없음 (pass-through) |

---

## 9. 왜 이 구조인가?

| 선택 | 이유 |
|------|------|
| TF++를 기본 플래너로 유지 | 안정적인 closed-loop 주행 성능 보존 |
| Alpamayo를 선택적으로만 호출 | 3~5초 추론 지연을 허용 가능하게 만듦 |
| Disagreement를 위험 신호로 사용 | "Alpamayo가 trajectory를 생성한다"는 능력을 실제로 활용 |
| steer 미수정 | 잘못된 조향 개입이 정상 루트에서 충돌 유발할 위험 방지 |

---

## 10. 관련 코드 위치

| 역할 | 파일 |
|------|------|
| TF++ pred_wp 추출 및 obs 주입 | `src/garage_ext/agents/ext_sensor_agent.py` |
| Alpamayo 비동기 ZMQ 클라이언트 | `src/garage_ext/modules/vlm/alpamayo_zmq.py` |
| Alpamayo 추론 서버 | `tools/alpamayo_server.py` |
| 위험도 계산 (disagreement 포함) | `src/garage_ext/modules/risk/action_aware_risk.py` |
| 개입 결정 및 제어 수정 | `src/garage_ext/modules/safety/semantic_arbiter.py` |
| 전체 파이프라인 연결 | `src/garage_ext/pipeline.py` |
| 실험 설정 (YAML) | `configs/experiments/alpamayo_structured.yaml` |

---

## 11. 논문 기여 한 줄

> *"We retain TF++ as the primary closed-loop backbone and use trajectory disagreement between TF++ predicted waypoints and Alpamayo predicted future motion as a risk signal for selective longitudinal intervention in safety-critical scenarios."*
