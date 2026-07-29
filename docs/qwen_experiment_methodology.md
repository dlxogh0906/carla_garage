# Qwen VLM Intervention 실험 방법론 및 기여점

작성 기준: 2026-04-25  
대상 코드: `carla_garage/src/garage_ext/agents/qwen_sensor_agent.py`, `carla_garage/src/garage_ext/vlm_intervention/`

## 1. 실험 목표

이 실험의 목표는 TF++를 대체하는 것이 아니라, TF++ 위에 Qwen VLM을 보조 안전 판단기로 붙여서 기존 TF++가 놓치는 시각적 위험 상황을 줄이는 것이다.

핵심 가정은 다음과 같다.

- TF++는 기본 주행, 경로 추종, 제어를 담당한다.
- Qwen은 전방 이미지와 TF++ 내부 신호를 같이 보고 “속도를 줄여야 하는지”만 판단한다.
- Qwen은 조향이나 경로를 직접 바꾸지 않는다.
- 최종 개입은 `speed_scale`로만 적용한다.

즉 전체 구조는 다음과 같다.

```text
CARLA sensor
  -> TF++ perception/planning/control
  -> TF++ bbox/path/speed/TTC context 추출
  -> Qwen VLM speed critic 또는 traffic-rule critic
  -> confidence/gate/state-machine
  -> TF++ target speed x speed_scale
  -> 최종 control
```

## 2. 기존 TF++의 문제

dev10 기준으로 기존 TF++는 평균 driving score가 74.75였다.

주요 실패 유형은 다음과 같았다.

| Scenario | 기존 TF++ 결과 | 실패 유형 |
|---|---:|---|
| `RouteScenario_25381_rep0` | 60.00 | vehicle collision |
| `RouteScenario_25378_rep0` | 70.00 | emergency vehicle yield |
| `RouteScenario_25424_rep0` | 51.51 | vehicle collision, outside route lanes |
| `RouteScenario_2091_rep0` | 60.00 | vehicle collision |
| `RouteScenario_17569_rep0` | 36.00 | vehicle collision 2회 |
| `RouteScenario_28198_rep0` | 70.00 | red light violation |

여기서 Qwen을 붙인 이유는 “TF++의 숫자 출력만으로는 애매한 장면”을 전방 이미지 기반으로 다시 검토하기 위해서다. 특히 보행자, 자전거, 정지 차량, 신호등, stop sign처럼 카메라 의미 정보가 중요한 장면에서 VLM의 이점이 있다고 보았다.

## 3. 방법론

### 3.1 TF++를 primary driver로 유지

Qwen은 주행 에이전트가 아니다. 매 step마다 TF++가 먼저 정상 실행된다.

TF++에서 가져오는 값은 다음과 같다.

| 항목 | 설명 |
|---|---|
| `rgb_front` | Qwen에 들어가는 전방 카메라 이미지 |
| `bb_buffer[-1]` | TF++가 감지한 3D bbox 목록 |
| `pred_checkpoints` | TF++가 예측한 주행 경로 |
| `pred_target_speed` | Qwen 개입 전 TF++ 목표 속도 |
| ego speed | 현재 속도 |
| control | TF++가 계산한 원래 throttle/brake/steer |

이 설계의 장점은 TF++의 안정적인 주행 능력을 유지하면서, Qwen은 위험 판단만 덧붙일 수 있다는 점이다.

### 3.2 Qwen 입력 이미지 강화

초기에는 Qwen에 raw front image와 몇 개의 숫자만 넣었다. 이 방식은 Qwen이 “어떤 객체가 TF++에서 위험 후보로 잡혔는지”를 모르기 때문에 불안정했다.

현재는 Qwen 입력 이미지를 다음처럼 강화했다.

- 전방 RGB 이미지 위에 ego driving corridor 표시
- TF++ bbox 기반 위험 후보 표시
- 같은 차선 후보와 주요 front obstacle 구분
- BEV inset 추가
- 헤더에 ego speed, front distance, TTC, TF++ target speed, traffic light/stop sign count 표시

Qwen은 raw image만 보는 것이 아니라, TF++가 감지한 구조화된 힌트가 얹힌 이미지를 본다. 이게 가장 중요한 방법론적 변화다.

### 3.3 텍스트 context 추가

Qwen에는 이미지와 함께 다음 텍스트 context가 들어간다.

| 입력 | 역할 |
|---|---|
| ego speed | 현재 차량 속도 |
| TF++ target speed | TF++가 원래 가려고 한 속도 |
| front distance | 전방 가장 가까운 후보 거리 |
| TTC | 충돌까지 예상 시간 |
| TTC source | bbox 기반인지 planner proxy 기반인지 |
| object table | bbox 객체의 class, x/y 위치, same-lane 여부, score |
| bbox class count | traffic light, stop sign, vehicle, pedestrian 개수 |
| path summary | TF++ predicted checkpoints |
| TTC history | 최근 TTC 변화 추세 |

중요한 점은 Qwen에게 TF++의 출력을 “그대로 믿으라”고 준 것이 아니라, 이미지와 대조해서 검증하라고 준 것이다.

### 3.4 prompt를 두 개로 분리

Qwen prompt는 하나로 모든 일을 시키지 않고 두 개로 나눴다.

| Prompt | 목적 | 출력 |
|---|---|---|
| speed critic | 전방 장애물, 차량, 보행자, 자전거, 경로 차단 판단 | `intervene`, `risk_level`, `speed_scale`, `hazard_type` |
| traffic-rule critic | 빨간불, 노란불, stop sign이 ego lane에 relevant한지 판단 | `rule_intervene`, `rule_type`, `traffic_light_state`, `confidence`, `speed_scale` |

분리한 이유는 generic collision risk와 traffic rule 판단이 성격이 다르기 때문이다. 예를 들어 빨간불은 TTC가 작지 않아도 멈춰야 하고, 앞차는 빨간불이 아니라도 감속해야 한다.

### 3.5 비동기 Qwen 추론

Qwen 추론은 10Hz CARLA loop 안에서 동기로 돌리면 너무 느리다. 그래서 background daemon thread에서 비동기로 돌린다.

```text
main control loop
  - TF++는 매 step 실행
  - Qwen 요청 조건만 판단
  - 최신 cached Qwen 결과만 즉시 읽음

Qwen worker thread
  - queue에서 최신 요청 1개만 처리
  - generate 후 JSON parse
  - cached result 갱신
```

큐 크기를 1로 제한해서 오래된 요청이 쌓이지 않게 했다. 이 구조 덕분에 Qwen 추론이 느려도 제어 loop 자체는 멈추지 않는다.

### 3.6 speed intervention 방식

Qwen은 최종적으로 `speed_scale`만 제안한다.

```text
final_target_speed = tfpp_target_speed * speed_scale
```

예시는 다음과 같다.

| `speed_scale` | 의미 |
|---:|---|
| 1.0 | TF++ 속도 유지 |
| 0.8 | 약한 감속 |
| 0.5 | 강한 감속 |
| 0.0 | 정지 |

조향은 건드리지 않는다. 따라서 이 방법은 “속도를 줄이면 피할 수 있는 실패”에는 효과가 있지만, “차선을 바꾸거나 경로를 수정해야만 피할 수 있는 실패”에는 한계가 있다.

### 3.7 TTC guard와 planner proxy

초기 TTC는 단순히 다음처럼 계산했다.

```text
TTC = front_distance / ego_speed
```

하지만 bbox가 없거나 거리 추정이 흔들리면 TTC가 너무 둔감했다. 그래서 TF++가 목표 속도를 급격히 낮추는 상황을 planner-proxy 위험 신호로 추가했다.

다만 이 값을 너무 민감하게 만들면 계속 브레이크를 밟고 route timeout이 발생했다. 그래서 현재는 다음처럼 완화했다.

- bbox TTC가 valid하면 bbox TTC를 우선 사용
- bbox TTC가 없을 때만 planner-proxy 사용
- TF++ target speed가 실제로 ego speed보다 낮을 때만 proxy 위험으로 인정
- proxy TTC guard는 완전 정지보다 부드러운 speed scale로 제한

이 부분은 성능을 올리기도 했지만, 과민하게 설정하면 바로 timeout으로 이어졌다.

### 3.8 traffic rule critic과 rule hold

빨간불/stop sign은 단발성 Qwen 결과만으로 처리하면 문제가 생긴다. 빨간불에서 한 번 멈춘 뒤 다음 step에 Qwen 응답이 늦거나 캐시가 비면 차량이 다시 슬금슬금 움직일 수 있기 때문이다.

그래서 rule hold 상태를 추가했다.

```text
Qwen: red_light or stop_sign, relevant_to_ego, high confidence
  -> rule_hold_active = True
  -> speed_scale = 0.0

release 조건:
  - green light를 confidence 높게 연속 확인
  - stop sign 정지 후 release guard 통과
  - not_visible/unknown이 반복되어 no-stop vote 충족
  - safety fallback step 초과
```

이 구조로 `RouteScenario_28198_rep0`의 red light violation은 개선되었다. 다만 안개나 visibility가 나쁜 장면에서는 Qwen이 보이지 않는 신호를 red로 hallucination하면 route timeout이 발생할 수 있었다.

### 3.9 logging 및 dashboard

모든 Qwen 판단은 `qwen_intervention.jsonl`로 저장된다.

저장되는 핵심 정보는 다음과 같다.

- step
- Qwen trigger reason
- prompt mode
- raw Qwen response
- parsed speed/rule decision
- speed scale
- TTC, front distance
- bbox class count
- rule hold state
- final applied scale

또한 dashboard PNG와 video를 만들어서, 실제 장면과 Qwen 판단을 같이 확인할 수 있게 했다. 이 부분은 실험 해석에 매우 중요했다. 점수만 보면 좋아 보이거나 나빠 보여도, 영상으로 보면 “왜 멈췄는지”, “왜 timeout이 났는지”가 보인다.

## 4. 실험 결과 요약

dev10 10개 route 기준이다.

| Run | Avg DS | Route Completion | Penalty | 해석 |
|---|---:|---:|---:|---|
| TF++ baseline | 74.75 | 100.00 | 0.7475 | 기준점 |
| `qwen_dev10_1` | 76.48 | 100.00 | 0.7648 | 약한 개선, red light는 아직 실패 |
| `qwen_dev10_2` | 73.68 | 86.33 | 0.8450 | 일부 충돌 개선, timeout 증가 |
| `qwen_dev10_3` | 84.56 | 94.89 | 0.8750 | 최고 점수, red light와 일부 collision 개선 |
| `qwen_dev10_4` | 69.35 | 83.09 | 0.8145 | rule hold 과민화로 timeout 다수 |
| `qwen_dev10_5` | 76.47 | 91.82 | 0.8264 | dev10_4보다 회복, 하지만 timeout/보행자 회귀 존재 |

가장 좋은 실험은 `qwen_dev10_3`이다. baseline 대비 평균 DS가 `+9.81` 상승했다.

```text
74.75 -> 84.56  (+9.81)
```

다만 최신 안정화 시도인 `qwen_dev10_5`는 baseline 대비 평균 DS가 `+1.72`만 상승했다.

```text
74.75 -> 76.47  (+1.72)
```

따라서 “Qwen이 가능성을 보였다”는 말은 맞지만, “이미 안정적으로 baseline을 압도한다”라고 말하기는 아직 어렵다.

## 5. 개선된 사례

### 5.1 `RouteScenario_25381_rep0`

기존 TF++는 vehicle collision으로 DS 60.00이었다.

Qwen 실험에서는 `qwen_dev10_1`, `qwen_dev10_3`, `qwen_dev10_5`에서 DS 100.00을 기록했다.

해석:

- 전방 위험 후보를 Qwen이 이미지와 bbox context로 확인
- speed-only 감속이 충돌 회피에 충분한 유형
- VLM critic이 가장 잘 맞는 케이스

### 5.2 `RouteScenario_2091_rep0`

기존 TF++는 vehicle collision으로 DS 60.00이었다.

`qwen_dev10_3`, `qwen_dev10_5`에서는 DS 100.00을 기록했다.

해석:

- 속도 조절만으로 해결 가능한 collision risk였던 것으로 보임
- 단, `qwen_dev10_2`, `qwen_dev10_4`에서는 과민 설정으로 timeout/collision이 발생했기 때문에 threshold tuning이 중요함

### 5.3 `RouteScenario_28198_rep0`

기존 TF++는 red light violation으로 DS 70.00이었다.

`qwen_dev10_2` 이후에는 DS 100.00을 기록했다.

해석:

- generic speed critic이 아니라 traffic-rule critic이 효과적이었던 사례
- red/yellow/green 상태와 ego-lane relevance를 분리한 것이 기여
- rule hold로 빨간불에서 슬금슬금 움직이는 문제를 줄임

### 5.4 `RouteScenario_17569_rep0`

기존 TF++는 vehicle collision 2회로 DS 36.00이었다.

`qwen_dev10_3`, `qwen_dev10_4`, `qwen_dev10_5`에서는 DS 60.00으로 개선되었다.

해석:

- collision을 완전히 제거하지는 못했지만 일부 위험을 줄임
- 남은 collision은 speed-only 개입만으로는 부족할 가능성이 큼

## 6. 실패 및 회귀 사례

### 6.1 `RouteScenario_25424_rep0`

이 route는 Qwen으로도 안정적으로 해결되지 않았다.

관찰:

- 기존 TF++도 collision/outside route lane으로 실패
- Qwen은 속도를 줄일 수는 있지만 경로를 바꾸지 못함
- 앞 장애물을 돌아가야 하는 상황에서는 조향/route-level 판단이 필요함
- 과도한 감속은 오히려 route timeout으로 이어짐

결론:

`RouteScenario_25424_rep0`는 현재 Qwen speed-only 구조의 한계를 보여주는 대표 사례다. 이 문제를 풀려면 Qwen이 단순히 speed scale만 주는 것이 아니라, TF++ trajectory 후보의 위험도를 평가하거나 lane-change 안전 판단에 개입해야 한다.

### 6.2 `RouteScenario_27494_rep0`

`qwen_dev10_3`, `qwen_dev10_4`, `qwen_dev10_5`에서 timeout 문제가 발생했다.

관찰:

- 안개/저시정 조건에서 traffic light 색이 명확하지 않음
- Qwen이 보이지 않거나 불명확한 신호를 red로 확신하는 문제가 발생
- rule hold가 걸리면 차가 지나가지 못하고 timeout 발생

결론:

traffic light 판단은 VLM 단독으로 confidence를 믿기 어렵다. 특히 안개, 작은 신호등, 화면 밖 신호등에서는 `not_visible` 또는 `unknown`으로 빠지는 보수적 정책이 필요하다.

### 6.3 `RouteScenario_3255_rep0`

`qwen_dev10_5`에서 pedestrian collision이 발생했다.

관찰:

- 이전 run에서는 100점이었으나 최신 rule/hold/tuning 후 회귀
- 교통 규칙 hold와 speed intervention 사이의 우선순위가 다른 위험 대응을 늦췄을 가능성이 있음

결론:

rule critic을 강화할수록 generic safety critic 반응이 늦어질 수 있다. 두 critic 사이의 arbitration이 필요하다.

### 6.4 `RouteScenario_25378_rep0`

baseline과 Qwen 모두 emergency vehicle yield 감점을 받았다.

결론:

현재 prompt와 logic은 emergency vehicle yield를 직접 다루지 않는다. 이 실패는 Qwen이 못 본다기보다, 아직 문제 정의에 포함하지 않은 영역이다.

## 7. 기여점

### 7.1 TF++를 유지한 VLM safety layer 설계

Qwen을 end-to-end driver로 쓰지 않고, TF++ 위의 safety/rule critic으로 제한했다. 이 덕분에 기존 TF++의 주행 안정성을 최대한 유지하면서 VLM의 시각적 판단 능력을 실험할 수 있었다.

### 7.2 구조화된 multimodal input

단순 이미지 입력이 아니라 다음을 함께 제공했다.

- annotated front camera
- ego corridor
- BEV inset
- object table
- bbox class count
- traffic light / stop sign candidate summary
- TF++ path checkpoint
- TTC history

이것은 “VLM에게 그냥 사진을 보여주는 방식”보다 훨씬 실험적으로 의미가 있다. TF++ perception과 VLM reasoning을 결합한 hybrid input 설계이기 때문이다.

### 7.3 speed critic과 traffic-rule critic 분리

충돌 위험과 교통 규칙 위반은 서로 다른 문제다. 두 prompt를 분리해서 red light/stop sign 판단을 별도 critic으로 만든 것이 중요한 기여다.

### 7.4 rule hold state machine

빨간불에서 한 번 멈춘 뒤 green 확인 전까지 정지를 유지하는 rule hold를 추가했다. 이것은 VLM의 단발성 판단을 제어 가능한 temporal policy로 바꾼 것이다.

### 7.5 confidence 기반 gating

Qwen 출력은 바로 믿지 않고 다음 조건을 통과해야 실제 개입된다.

- relevant_to_ego
- confidence threshold
- result age
- rule type validation
- speed scale sanity check

이런 gate 없이는 VLM hallucination이 곧바로 제어 실패로 이어진다.

### 7.6 비동기 추론 구조

VLM이 느려도 10Hz 제어 loop를 막지 않도록 daemon thread와 cached result 구조를 만들었다. 실제 자율주행 control loop에 VLM을 붙이려면 필수적인 구조다.

### 7.7 실험 해석 가능성

`qwen_intervention.jsonl`, dashboard PNG, dashboard video를 통해 Qwen의 판단을 추적 가능하게 만들었다. 이 덕분에 단순 점수 비교가 아니라 “어떤 장면에서 왜 실패했는지”까지 분석할 수 있었다.

## 8. VLM 기여도 평가

엄밀한 “순수 VLM 기여도”를 말하려면 ablation이 더 필요하다. 현재 시스템에는 Qwen prompt, TTC guard, rule hold, confidence gate가 같이 들어가 있기 때문이다. 그래도 현재 실험 결과를 기준으로 추정하면 다음과 같다.

| 기준 | 점수 | 근거 |
|---|---:|---|
| 최고 성능 run 기준 기여도 | 70 / 100 | `qwen_dev10_3`에서 DS 74.75 -> 84.56, red light와 일부 collision 개선 |
| 최신 안정화 run 기준 기여도 | 55 / 100 | `qwen_dev10_5`에서 DS 74.75 -> 76.47, 개선은 있으나 timeout/보행자 회귀 존재 |
| traffic rule 판단 기여도 | 65 / 100 | `RouteScenario_28198_rep0` red light violation 제거, 하지만 저시정 신호등 hallucination 존재 |
| 전방 collision 감속 기여도 | 65 / 100 | `25381`, `2091`, `17569` 일부 개선, lane-change/route-level 문제는 미해결 |
| 연구/방법론 기여도 | 85 / 100 | TF++와 VLM을 non-blocking hybrid critic으로 결합하고, 입력/로그/상태머신까지 설계 |
| 현재 실전 안정성 | 45 / 100 | threshold와 prompt에 민감하고, full 220 route 일반화 검증 전 |

현재 한 줄 결론은 다음과 같다.

```text
Qwen VLM의 현재 기여도는 연구 방법론 기준 85/100,
dev10 실제 성능 기여 기준 55~70/100,
종합 판단으로는 65/100 정도다.
```

즉 Qwen은 “분명히 기여하는 장면이 있다.” 하지만 아직 모든 route에서 안정적으로 이득을 주는 모듈은 아니다.

## 9. TF++ 출력을 Qwen에 주는 것이 기여가 없는가?

아니다. 오히려 이것이 핵심 기여다.

단순히 TF++ 출력을 그대로 후처리했다면 기여가 약했을 수 있다. 하지만 현재 방식은 다음을 한다.

- TF++ bbox/path/speed를 VLM이 이해할 수 있는 multimodal context로 변환
- Qwen이 이미지와 TF++ 구조 정보를 cross-check
- Qwen 판단을 confidence gate와 state machine으로 제어
- TF++ controller를 유지하면서 target speed만 안전하게 조절

즉 기여는 “Qwen이 혼자 운전한다”가 아니라, “기존 planner의 약점을 VLM critic으로 보완하는 구조”에 있다.

## 10. 현재 한계

### 10.1 speed-only 개입의 한계

차선을 바꾸거나 장애물을 돌아가야 하는 상황에서는 속도만 줄여도 해결되지 않는다. `RouteScenario_25424_rep0`가 대표적이다.

### 10.2 traffic light 색상 인식 불안정

작은 신호등, 안개, 화면 밖 신호등에서는 Qwen이 red/yellow/green을 잘못 확신할 수 있다. 이 경우 rule hold가 timeout을 만든다.

### 10.3 emergency vehicle yield 미구현

현재 prompt와 gate는 emergency vehicle yield를 명시적으로 다루지 않는다.

### 10.4 full benchmark 일반화 미검증

dev10에서는 가능성이 보였지만, 220개 route 전체에서 동일하게 좋아질지는 아직 검증되지 않았다. 일부 scenario에 맞춘 tuning이 되면 full set에서 오히려 성능이 떨어질 수 있다.

## 11. 다음 개선 방향

### 11.1 VLM traffic light 판단을 crop 기반으로 분리

현재는 전체 front image에서 신호등을 본다. 앞으로는 traffic light bbox crop을 따로 잘라서 Qwen 또는 작은 classifier에 넣는 편이 더 안정적이다.

필요한 출력:

```json
{
  "visible": true,
  "state": "red|yellow|green|unknown",
  "confidence": 0.0
}
```

### 11.2 green release를 더 엄격하게, red hold는 더 보수적으로

red로 멈추는 조건은 더 엄격하게 해야 한다.

- red 확신이 낮으면 hold 금지
- not_visible이면 hold 금지
- unknown이면 hold 금지
- bbox class가 traffic light여도 색이 안 보이면 hold 금지

반대로 이미 멈춘 상태에서 출발하는 green release는 crop 기반 green 확인을 쓰는 것이 좋다.

### 11.3 lane-change/route-risk critic 추가

`RouteScenario_25424_rep0` 같은 문제는 speed critic이 아니라 trajectory critic이 필요하다.

Qwen 또는 별도 module이 다음을 판단해야 한다.

- TF++ planned checkpoints가 장애물을 통과하는지
- target lane에 차량이 있는지
- route lane 밖으로 나가는지
- 감속만으로 충분한지, trajectory 자체가 위험한지

다만 현재 controller가 조향을 바꾸지 않기 때문에, 이 판단을 어떻게 TF++에 반영할지는 별도 설계가 필요하다.

### 11.4 ablation 실험

VLM의 순수 기여를 명확히 보려면 다음 ablation이 필요하다.

| 실험 | 목적 |
|---|---|
| TF++ baseline | 기준점 |
| TTC guard only | heuristic 감속 기여 |
| Qwen speed critic only | VLM collision critic 기여 |
| Qwen rule critic only | VLM traffic rule 기여 |
| Qwen speed + rule | 통합 성능 |
| Qwen without annotated input | bbox/BEV annotation 기여 |
| Qwen with annotated input | 최종 proposed method |

이 ablation을 해야 “VLM이 몇 점 기여했는지”를 더 엄밀하게 말할 수 있다.

## 12. 최종 요약

이번 Qwen 실험의 가장 큰 성과는 평균 점수 하나가 아니라, TF++와 VLM을 결합하는 구조를 실제 CARLA evaluation loop 안에 넣었다는 점이다.

구체적 기여는 다음과 같다.

- TF++ primary + Qwen secondary critic 구조 구현
- annotated image + BEV + object table + path/TTC context 입력 설계
- speed critic / traffic-rule critic prompt 분리
- confidence gate와 rule hold state machine 구현
- 비동기 VLM 추론으로 10Hz control loop 유지
- JSONL/dashboard/video 기반 분석 체계 구축
- dev10 일부 route에서 collision/red-light failure 개선 확인

현재 결론:

```text
Qwen은 특정 시각적 위험과 traffic-rule 문제에서 확실히 도움이 된다.
하지만 speed-only 개입이기 때문에 lane-change, route deviation, 저시정 신호등 문제는 아직 불안정하다.
현 단계의 VLM 기여도는 종합 65/100 정도로 보는 것이 적절하다.
```

