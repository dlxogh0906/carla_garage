# Qwen3-VL MixedGPTQ / ActAwareMixedGPTQ 적용 설명

작성 기준: 2026-05-17  
대상 모델: Qwen3-VL-8B-Instruct  
대상 실험: CARLA Bench2Drive rear ClassicCV, front+rear image input

## 1. 한 줄 요약

`MixedGPTQ-W4W8A16`과 `ActAwareMixedGPTQ-W4W8A16`은 Qwen3-VL 전체를 무조건 4bit로 낮추지 않고, 자율주행 판단에 민감할 수 있는 부분은 8bit 또는 BF16으로 남기고 나머지 LLM 내부 Linear layer만 GPTQ 4bit로 압축한 방식이다.

가장 중요한 차이는 다음과 같다.

| 방법 | W8로 보호할 LLM layer 선택 방식 | 핵심 아이디어 |
|---|---|---|
| `MixedGPTQ-W4W8A16` | 규칙 기반 | 처음과 마지막 language layer는 중요하다고 보고 W8로 보호 |
| `ActAwareMixedGPTQ-W4W8A16` | activation 분석 기반 | 실제 자율주행 calibration data를 넣어 activation이 큰 layer를 찾아 W8로 보호 |

즉 두 방법 모두 기본 양자화 알고리즘은 `GPTQ`이고, ActAware는 AWQ처럼 activation을 보고 중요한 부분을 보호한다는 아이디어를 차용했다. 하지만 AWQ quantizer 자체를 사용한 것은 아니다.

## 2. 왜 이런 방법을 썼는가

일반 GPTQ-W4A16은 많은 LLM Linear weight를 4bit로 줄여서 모델 크기와 GPU memory를 줄인다.

문제는 자율주행 VLM에서는 모든 layer가 똑같이 중요하지 않다는 점이다.

- vision feature를 language model로 넘기는 projection 부분
- 첫 language layer
- 마지막 decision/output에 가까운 language layer
- calibration data에서 activation이 크게 튀는 layer

이런 부분까지 전부 4bit로 낮추면 작은 수치 오차가 최종 판단, 예를 들면 감속/정지 판단에 크게 영향을 줄 수 있다.

그래서 적용한 전략은 다음이다.

```text
중요한 부분: W8A16 또는 BF16으로 보호
덜 민감한 LLM 내부 layer: GPTQ W4A16으로 강하게 압축
```

## 3. W4A16, W8A16 뜻

| 표기 | 의미 | 설명 |
|---|---|---|
| W4A16 | weight 4bit, activation 16bit | GPTQ로 weight만 4bit 압축. activation은 BF16/FP16 계열로 유지 |
| W8A16 | weight 8bit, activation 16bit | 중요한 layer를 4bit보다 덜 공격적으로 압축 |
| BF16 | 원래 정밀도 유지 | 양자화하지 않고 보존 |

여기서 중요한 점은 `W4A16`이 activation까지 4bit로 낮춘다는 뜻이 아니라는 것이다. activation은 16bit 계열로 유지하고 weight만 4bit로 압축한다.

## 4. 전체 적용 구조

전체 흐름은 아래처럼 보면 된다.

```text
원본 Qwen3-VL-8B BF16 모델
  |
  | 1. 어떤 module을 W4/W8/BF16으로 둘지 결정
  v
Mixed-bit bit allocation plan
  |
  | 2. llmcompressor GPTQModifier에 W4A16/W8A16 target 전달
  v
GPTQ calibration
  |
  | 3. compressed-tensors checkpoint 저장
  v
MixedGPTQ 또는 ActAwareMixedGPTQ checkpoint
  |
  | 4. vLLM compressed-tensors backend로 CARLA 평가
  v
Bench2Drive rear ClassicCV evaluation
```

사용한 핵심 스크립트는 다음이다.

| 파일 | 역할 |
|---|---|
| `tools/quantize_qwen3vl_mixed_gptq.py` | mixed-bit GPTQ checkpoint 생성 |
| `tools/analyze_qwen_activation_importance.py` | ActAware용 activation 중요도 분석 |
| `run_rear_classiccv_8meta_quant_methods_suite.sh` | rear ClassicCV 조건에서 vLLM 평가 실행 |

## 5. 공통으로 보호한 부분

두 방법 모두 아래 부분은 동일하게 처리했다.

| 모델 부분 | 적용 정밀도 | 이유 |
|---|---|---|
| Visual encoder blocks | BF16 | 이미지 feature를 뽑는 부분이라 4bit로 내리지 않고 보존 |
| Visual merger / projection | W8A16 | vision feature를 language model로 넘기는 연결부라 W4보다 안전하게 보호 |
| `lm_head` | BF16 | 마지막 token 출력부라 양자화 대상에서 제외 |
| 선택된 language layer | W8A16 | 자율주행 판단에 민감할 수 있는 layer 보호 |
| 나머지 language layer | W4A16 GPTQ | 모델 크기와 GPU memory를 줄이는 주 압축 대상 |

여기서 `Visual encoder blocks`는 아예 양자화하지 않고 BF16으로 남겼다. 반면 `Visual merger / projection`은 BF16까지는 아니고 W8A16으로 낮춰서 어느 정도 압축하면서도 정보 손실을 줄였다.

## 6. MixedGPTQ-W4W8A16 적용 방식

`MixedGPTQ-W4W8A16`은 규칙 기반 mixed-bit GPTQ이다.

기준은 단순하다.

```text
처음 language layer 일부: W8A16
마지막 language layer 일부: W8A16
중간 language layer 대부분: W4A16 GPTQ
visual projection: W8A16
visual blocks, lm_head: BF16
```

실제 생성된 모델의 bit allocation은 다음과 같다.

| 구간 | 적용 |
|---|---|
| Language layer `0, 1` | W8A16 |
| Language layer `2` to `33` | W4A16 GPTQ |
| Language layer `34, 35` | W8A16 |
| Visual merger / projection | W8A16 |
| Visual encoder blocks | BF16 |
| `lm_head` | BF16 |

왜 처음과 마지막 layer를 보호했는가?

- 처음 layer는 vision/text 입력이 LLM 내부 표현으로 바뀌는 초반부라 입력 정보 손실에 민감할 수 있다.
- 마지막 layer는 최종 answer token, 즉 meta-action 판단에 가까워서 출력 안정성에 중요할 수 있다.
- 중간 layer는 수가 많기 때문에 4bit로 압축하면 memory saving 효과가 크다.

생성된 모델 경로:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-MixedGPTQ-W4W8A16
```

## 7. ActAwareMixedGPTQ-W4W8A16 적용 방식

`ActAwareMixedGPTQ-W4W8A16`은 MixedGPTQ를 더 발전시킨 방식이다.

차이는 W8 layer를 사람이 정한 규칙만으로 고르지 않고, 실제 자율주행 calibration data를 Qwen3-VL에 넣어서 activation 통계를 보고 고른다는 점이다.

적용 순서는 다음과 같다.

```text
1. 자율주행 calibration JSONL을 준비
2. Qwen3-VL BF16 모델에 calibration sample을 forward
3. 각 Linear module의 입력 activation 통계 수집
4. activation score가 큰 layer를 중요 layer 후보로 선택
5. boundary layer도 안전하게 포함
6. 선택된 layer는 W8A16으로 보호
7. 나머지 language layer는 GPTQ W4A16으로 압축
```

activation 분석에서 본 통계는 다음이다.

| 통계 | 의미 |
|---|---|
| `rms` | activation의 평균적인 크기 |
| `max_abs` | calibration 중 가장 크게 튄 activation |
| `outlier_ratio` | 평균 대비 outlier가 얼마나 큰지 |
| `param_count` | 해당 module의 파라미터 규모 |

중요도 점수는 아래 proxy score를 사용했다.

```text
importance_score =
  0.40 * normalized_rms
+ 0.30 * normalized_max_abs
+ 0.20 * normalized_outlier_ratio
+ 0.10 * normalized_param_count
```

이 점수는 "activation이 크고 outlier가 큰 layer일수록 4bit 양자화 오차에 민감할 수 있다"는 가정을 반영한다.

실제 activation 분석 결과로 선택된 W8 language layer는 다음이다.

```text
0, 32, 33, 34, 35
```

그래서 최종 bit allocation은 다음과 같다.

| 구간 | 적용 |
|---|---|
| Language layer `0` | W8A16 |
| Language layer `1` to `31` | W4A16 GPTQ |
| Language layer `32, 33, 34, 35` | W8A16 |
| Visual merger / projection | W8A16 |
| Visual encoder blocks | BF16 |
| `lm_head` | BF16 |

생성된 모델 경로:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-ActAwareMixedGPTQ-W4W8A16-n8
```

주의할 점:

- 경로의 `n8`은 activation importance 분석에 8개 calibration sample을 사용했다는 의미로 붙인 이름이다.
- 실제 GPTQ checkpoint 생성에는 metadata 기준 `num_calibration_samples=64`가 사용되었다.

## 8. MixedGPTQ와 ActAwareMixedGPTQ의 차이

두 방법의 차이는 "어떤 language layer를 W8로 보호할지"이다.

| 항목 | MixedGPTQ-W4W8A16 | ActAwareMixedGPTQ-W4W8A16 |
|---|---|---|
| W8 layer 선택 | 규칙 기반 | calibration activation 기반 |
| W8 language layer | `0, 1, 34, 35` | `0, 32, 33, 34, 35` |
| W4 language layer | `2` to `33` | `1` to `31` |
| Visual projection | W8A16 | W8A16 |
| Visual blocks | BF16 | BF16 |
| `lm_head` | BF16 | BF16 |
| 목적 | 단순하고 안정적인 mixed-bit baseline | 자율주행 data에 맞춘 중요 layer 보호 |

쉽게 말하면:

```text
MixedGPTQ:
  "처음과 마지막은 중요할 것 같으니 보호하자."

ActAwareMixedGPTQ:
  "실제 주행 calibration data를 넣어보니 activation상 중요한 layer가 여기니까 보호하자."
```

## 9. GPTQ baseline과의 차이

일반 `GPTQ-W4A16` baseline은 language Linear layer 대부분을 동일하게 W4A16으로 양자화한다.

반면 mixed 계열은 다음처럼 더 조심스럽다.

```text
GPTQ-W4A16:
  많은 LLM layer를 동일하게 4bit로 압축

MixedGPTQ-W4W8A16:
  boundary language layer + visual projection은 W8로 보호
  나머지만 W4 GPTQ

ActAwareMixedGPTQ-W4W8A16:
  activation이 중요한 language layer + visual projection은 W8로 보호
  나머지만 W4 GPTQ
```

따라서 mixed 계열은 모델 크기와 GPU memory를 조금 더 쓰지만, 자율주행 판단 정확도를 더 안정적으로 유지하는 것을 목표로 한다.

## 10. 실제 생성 명령어

### 10.1 MixedGPTQ 생성

아래 명령은 처음 2개 layer와 마지막 2개 layer를 W8A16으로 보호하고, 나머지 language layer를 W4A16 GPTQ로 양자화한다.

```bash
cd /mnt/2/carla_garage

CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python tools/quantize_qwen3vl_mixed_gptq.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-MixedGPTQ-W4W8A16 \
  --num-samples 64 \
  --max-seq-length 4096 \
  --device-map auto \
  --dtype bfloat16 \
  --w8-first-layers 2 \
  --w8-last-layers 2 \
  --offload-hessians
```

### 10.2 Activation importance 분석

아래 명령은 자율주행 calibration data를 Qwen3-VL에 넣고 activation 통계를 수집한다.

```bash
cd /mnt/2/carla_garage

CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python tools/analyze_qwen_activation_importance.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --num-samples 8 \
  --target-scope language_projection \
  --w8-top-layers 4 \
  --always-w8-boundary-layers 1
```

이 실행에서 나온 추천 W8 layer:

```text
0,32,33,34,35
```

분석 결과 파일:

```text
/mnt/2/carla_metric_result/qwen_activation_importance/qwen3vl_activation_importance.json
/mnt/2/carla_metric_result/qwen_activation_importance/qwen3vl_activation_importance.csv
```

### 10.3 ActAwareMixedGPTQ 생성

activation 분석에서 얻은 `0,32,33,34,35`를 `--w8-layers`로 넣어 생성한다.

```bash
cd /mnt/2/carla_garage

CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python tools/quantize_qwen3vl_mixed_gptq.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-ActAwareMixedGPTQ-W4W8A16-n8 \
  --num-samples 64 \
  --max-seq-length 4096 \
  --device-map auto \
  --dtype bfloat16 \
  --w8-layers 0,32,33,34,35 \
  --offload-hessians
```

## 11. CARLA 평가 적용 방식

생성된 checkpoint는 vLLM의 `compressed-tensors` backend로 실행했다.

공통 평가 조건:

| 항목 | 값 |
|---|---|
| 평가 | Bench2Drive dev10 |
| 방법 | TF++ + Meta-Action VLA + ClassicCV(front+rear) |
| 입력 이미지 | front + rear, request당 2 images |
| vLLM image limit | `{"image": 2}` |
| prompt mode | `team8_meta_action_digit` |
| backend | vLLM OpenAI-compatible server |
| dtype | BF16 |
| max model length | 4096 |
| quantization loader | `compressed-tensors` |

실행 예시:

```bash
cd /mnt/2/carla_garage

SUITE_ROOT=/mnt/2/carla_metric_result2/rear_classiccv_8meta_quant_methods_suite_YYYYMMDD_HHMMSS \
SUITE_FRESH=1 \
CARLA_MAX_GAME_TIME_SECONDS=60 \
RUNS=gptq_w4a16_vllm,mixed_gptq_w4w8a16_vllm,actaware_mixed_gptq_w4w8a16_vllm \
MODEL_GPTQ=/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16-n64 \
MODEL_MIXED=/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-MixedGPTQ-W4W8A16 \
MODEL_ACTAWARE_MIXED=/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-ActAwareMixedGPTQ-W4W8A16-n8 \
VLLM_CUDA_VISIBLE_DEVICES=1 \
bash run_rear_classiccv_8meta_quant_methods_suite.sh
```

## 12. 결과 해석 방법

결과를 볼 때는 반드시 같은 조건끼리 비교해야 한다.

비교 가능한 조건:

```text
front+rear 입력
image=2
input tokens 약 1241
same rear ClassicCV suite
same backend vLLM
```

비교하면 안 되는 조건:

```text
front-only 실험과 front+rear 실험
image=1 실험과 image=2 실험
input tokens 691 실험과 input tokens 1241 실험
transformers backend와 vLLM backend를 latency로 직접 비교
```

예를 들어 front-only 실험은 latency가 400ms대로 나올 수 있지만, rear 포함 실험은 request당 image가 2장이고 input token도 많아져 700ms대로 나오는 것이 자연스럽다.

## 13. 팀원에게 설명할 때 쓰는 쉬운 표현

짧게 설명하면:

```text
기존 GPTQ는 모델 내부 layer를 거의 똑같이 4bit로 낮춥니다.
우리는 자율주행 판단에 중요한 부분까지 무조건 4bit로 낮추면 위험하다고 보고,
vision encoder와 출력부는 보존하고,
vision-language projection과 중요한 LLM layer는 8bit로 보호했습니다.
나머지 LLM layer만 GPTQ 4bit로 강하게 압축했습니다.
```

MixedGPTQ 설명:

```text
MixedGPTQ는 처음과 마지막 LLM layer가 중요하다고 보는 규칙 기반 방법입니다.
처음 2개 layer와 마지막 2개 layer를 W8로 남기고,
중간 layer만 W4 GPTQ로 압축했습니다.
```

ActAwareMixedGPTQ 설명:

```text
ActAwareMixedGPTQ는 실제 자율주행 calibration data를 넣어서 activation이 크게 반응하는 layer를 찾고,
그 layer를 W8로 보호한 방법입니다.
이번 모델에서는 layer 0, 32, 33, 34, 35를 W8로 남겼고,
나머지 layer 1-31을 W4 GPTQ로 압축했습니다.
```

논문/발표용 표현:

```text
We apply an activation-aware mixed-bit GPTQ scheme for Qwen3-VL.
Safety-sensitive layers identified from autonomous driving calibration activations are preserved in W8A16/BF16,
while less sensitive language layers are compressed using GPTQ W4A16.
```

## 14. 자주 생기는 오해

### 오해 1. ActAwareMixedGPTQ는 AWQ인가?

아니다. 기본 양자화 알고리즘은 GPTQ이다.

다만 AWQ처럼 calibration activation을 보고 중요한 부분을 보호한다는 아이디어를 차용했다.

정확한 표현:

```text
AWQ-inspired activation-aware mixed-bit GPTQ
```

부정확한 표현:

```text
AWQ를 적용했다
```

### 오해 2. 모든 layer를 4bit로 낮춘 것인가?

아니다. 중요한 부분은 W8A16 또는 BF16으로 남겼다.

```text
W4A16: 중간 LLM layer 대부분
W8A16: 선택된 LLM layer + visual projection
BF16: visual encoder blocks + lm_head
```

### 오해 3. ActAware의 `n8`은 GPTQ calibration sample 8개라는 뜻인가?

아니다. `n8`은 activation importance 분석에 사용한 sample 수를 의미하는 이름이다.

실제 GPTQ checkpoint 생성 metadata에는 `num_calibration_samples=64`로 기록되어 있다.

### 오해 4. front-only 결과와 비교해도 되는가?

안 된다. 현재 rear ClassicCV 실험은 front+rear 두 장의 이미지를 넣는다.

front-only는 평균 input token이 약 691이고, rear 포함 실험은 약 1241이다. latency와 VLM FPS를 직접 비교하면 안 된다.

## 15. 확인용 명령어

모델에 실제로 어떤 layer가 W8/W4로 들어갔는지 확인하려면 metadata를 보면 된다.

```bash
sed -n '1,220p' /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-MixedGPTQ-W4W8A16/carla_quantization_metadata.json

sed -n '1,260p' /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-ActAwareMixedGPTQ-W4W8A16-n8/carla_quantization_metadata.json
```

평가 결과는 각 suite root의 아래 파일을 보면 된다.

```text
rear_classiccv_8meta_quant_methods_metrics.md
rear_classiccv_8meta_quant_methods_metrics.csv
```

예시 결과 경로:

```text
/mnt/2/carla_metric_result2/rear_classiccv_8meta_quant_methods_suite_20260515_223048
/mnt/2/carla_metric_result2/rear_classiccv_8meta_quant_methods_rep1_20260516_233318
```

