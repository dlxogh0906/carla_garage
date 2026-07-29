# Qwen3-VL Quantization Worklog

작성 시각: 2026-05-03 20:00 KST  
작성 기준 경로: `/mnt/2/carla_garage`  
대상 모델: `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct`

이 문서는 현재 프로젝트에서 실제로 수행한 Qwen3-VL-8B calibration data 생성, 양자화 checkpoint 생성, benchmark, 그리고 RTX A6000에서 FP8을 메인 실험으로 쓰지 않은 이유를 정리한 작업 기록이다.

## 1. Calibration Data를 어떻게 만들었나

calibration data는 외부 일반 VLM 데이터셋이 아니라, **Bench2Drive/CARLA 주행 중 Qwen VLM client가 실제로 받는 image/prompt 분포**를 저장해서 만들었다.

핵심 파일:

```text
/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl
```

생성 방식:

1. BF16 Qwen3-VL-8B를 연결한 Bench2Drive route 실행에서 `QWEN_SAVE_CALIB=1`을 켰다.
2. `src/garage_ext/vlm_intervention/qwen_client.py`의 `_dump_calibration_sample()`이 Qwen 호출마다 calibration sample을 저장했다.
3. 각 sample은 이미지 파일과 Qwen chat template 적용 후의 최종 text를 함께 저장한다.
4. route `idx0`부터 `idx9`까지의 calibration JSONL을 합쳐 `calibration_merged.jsonl`로 사용했다.

저장되는 주요 필드:

| Field | 의미 |
|---|---|
| `sample_id` | request id, step, prompt mode를 포함한 sample 이름 |
| `image_path` | 해당 Qwen 호출에 들어간 front image PNG 경로 |
| `prompt_text` | 원본 prompt text |
| `chat_template_text` | tokenizer/chat template 적용 후 calibration에 사용한 최종 text |
| `input_tokens` | 기록된 input token 수 |
| `prompt_mode` | `speed` 또는 `traffic_rule` |
| `trigger` | `object`, `ttc`, `traffic_rule` 등 호출 trigger |
| `context` | object table, rule context, path summary, TTC history 등 주행 context |

확인된 calibration data 통계:

| 항목 | 값 |
|---|---:|
| total samples | 458 |
| image files | 458 |
| missing images | 0 |
| route range | `idx0`-`idx9` |
| prompt mode: speed | 419 |
| prompt mode: traffic_rule | 39 |
| trigger: object | 398 |
| trigger: ttc | 21 |
| trigger: traffic_rule | 39 |
| input tokens min / max / mean | 1300 / 1911 / 1716.93 |

양자화 스크립트에서 calibration data를 쓰는 방식:

- `tools/quantize_qwen3vl_w8a8.py`
- `tools/quantize_qwen3vl_w4a16.py`

두 스크립트 모두 `calibration_merged.jsonl`을 읽어서 아래 형태의 HuggingFace `Dataset`으로 바꾼다.

```text
text   <- chat_template_text
images <- image_path
```

그 후 `images` column을 `datasets.Image(decode=True)`로 cast하고, batch size 1 전용 collator를 사용해서 Qwen3-VL processor/model에 넣었다. `max_seq_length`는 4096으로 맞췄다.

## 2. 수행한 양자화 작업

### 2.1 BF16 / FP16 baseline

원본 Qwen3-VL-8B checkpoint를 baseline으로 사용했다.

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct
```

역할:

- 모든 양자화 결과의 기준
- calibration data 생성용 BF16 주행 run의 source model
- Bench2Drive 단일 시나리오 비교 기준

Bench2Drive single scenario active 결과:

| Quant | Params(B) | Model Size(GB) | Load Mem(GB) | Peak Mem(GB) | Avg Latency(ms) | P95(ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 8.767 | 16.33 | 16.359 | 16.826 | 6032.01 | 6712.67 | 15.205 |

주의: 이전 backup run에는 avg 5985.27 ms, P95 6642.10 ms, peak 16.895 GB 값도 있다. 표에 쓸 때는 active run 또는 backup run 중 하나로 통일해야 한다.

### 2.2 BitsAndBytes runtime INT8 / 4bit

BitsAndBytes는 별도 checkpoint를 저장하는 방식이 아니라, 원본 checkpoint를 로드할 때 runtime quantization을 적용했다.

관련 코드:

```text
src/garage_ext/vlm_intervention/qwen_client.py
tools/benchmark_qwen_vl.py
```

사용 방식:

| Runtime quant | 설정 |
|---|---|
| BNB INT8 | `QWEN_RUNTIME_QUANT=bnb8`, `BitsAndBytesConfig(load_in_8bit=True)` |
| BNB 4bit | `QWEN_RUNTIME_QUANT=bnb4`, NF4, double quant 사용 |

단일 시나리오에서 INT8-bnb는 메모리는 줄었지만 latency가 크게 나빠졌다.

| Quant | Params(B) | Model Size(GB) | Load Mem(GB) | Peak Mem(GB) | Avg Latency(ms) | P95(ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT8-bnb | 8.767 | 16.33 | 9.387 | 10.061 | 22402.73 | 25234.15 | 3.975 |

해석:

- A6000에서 실행은 가능하다.
- VRAM 감소 효과는 있다.
- 하지만 이 프로젝트의 주행 loop 기준으로는 너무 느려서 메인 후보로 보기 어렵다.

BNB 4bit는 standalone benchmark 결과는 있으나, 이 문서 작성 시점 기준 Bench2Drive 단일 시나리오 결과는 아직 정리 대상에 없다.

### 2.3 LLMCompressor W8A8-INT8

W8A8-INT8 checkpoint는 `llmcompressor`로 생성했다.

관련 스크립트:

```text
tools/quantize_qwen3vl_w8a8.py
```

출력 checkpoint:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8
```

vLLM 0.12용 실행 경로:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012
```

주요 설정:

| 항목 | 값 |
|---|---|
| calibration samples | 458 |
| max seq length | 4096 |
| tool | `llmcompressor` |
| recipe | `SmoothQuantModifier` + `GPTQModifier` |
| scheme | `W8A8` |
| targets | `Linear` |
| ignored layers | `lm_head`, `re:.*visual.*` |
| SmoothQuant smoothing strength | 0.8 |
| visual tower quantization | false |

metadata:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8/carla_quantization_metadata.json
```

단일 시나리오 결과:

| Quant | Params(B) | Model Size(GB) | Load Mem(GB) | Peak Mem(GB) | Avg Latency(ms) | P95(ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| W8A8-INT8-vLLM | 8.767 | 9.864 | 13.940 | 23.285 | 2031.99 | 2229.15 | 48.393 |

해석:

- Transformers 경로에서는 압축 runtime 이득이 잘 나오지 않았다.
- vLLM compressed-tensors 경로에서는 latency가 크게 개선됐다.
- 다만 vLLM engine/KV/encoder cache 때문에 peak memory가 BF16보다 높게 찍혔다.

### 2.4 LLMCompressor AWQ W4A16

AWQ W4A16 checkpoint는 `tools/quantize_qwen3vl_w4a16.py`로 생성했다.

관련 스크립트:

```text
tools/quantize_qwen3vl_w4a16.py
```

공통 설정:

| 항목 | 값 |
|---|---|
| method | `awq` |
| scheme | `W4A16_ASYM` |
| targets | `Linear` |
| ignored layers | `lm_head`, `re:.*visual.*` |
| visual tower quantization | false |
| max seq length | 4096 |
| duo scaling | true |
| n_grid | 20 |

수행한 AWQ 작업:

| 작업 | 상태 | 경로 / 결과 |
|---|---|---|
| AWQ W4A16 no-save smoke | 성공 | `--num-samples 1 --no-save` |
| AWQ W4A16 n4 checkpoint | 성공 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n4` |
| AWQ W4A16 n4 vLLM benchmark | 성공 | 약 914 ms, 약 99.5 tokens/sec |
| AWQ W4A16 n458 | 실패 | `(1/37): Calibrating` 425/458에서 CUDA OOM |
| AWQ W4A16 n458 부산물 | 삭제 완료 | n458 output dir/log 삭제 |
| AWQ W4A16 n64 checkpoint | 성공 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n64` |
| AWQ W4A16 n64 vLLM benchmark | 성공 | 약 923 ms, 약 100.8 tokens/sec |
| AWQ W4A16 n64 Bench2Drive single scenario | 성공 | 아래 표 참조 |

AWQ n64 metadata:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n64/carla_quantization_metadata.json
```

AWQ n64 standalone vLLM benchmark:

| Quant | Model Size(GB) | Load Mem(GB) | Peak Mem(GB) | Avg Latency(ms) | P95(ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| AWQ-W4A16-n64 | 6.769 | 10.997 | 10.997 | 922.682 | 924.284 | 100.793 |

AWQ n64 Bench2Drive single scenario:

```text
/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_AWQ-W4A16-n64-vLLM_idx0/
```

| Quant | Params(B) | Model Size(GB) | Load Mem(GB) | Peak Mem(GB) | Avg Latency(ms) | P95(ms) | Tokens/sec | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AWQ-W4A16-n64-vLLM | 8.767 | 6.753 | 10.809 | 20.040 | 1589.54 | 1724.44 | 65.821 | 123 |

주의:

- vLLM/summary CSV에는 AWQ params가 `1.821B`로 찍히는 경우가 있다.
- 이는 compressed tensor accounting이고, logical model parameter count가 1.821B라는 뜻이 아니다.
- Qwen3-VL-8B의 logical params는 BF16, INT8, W8A8, AWQ 모두 8.767B로 쓰는 것이 맞다.
- 양자화의 이득은 params 감소가 아니라 model size, memory, latency로 보고해야 한다.

AWQ n64 단일 시나리오 주행 결과:

| 항목 | 결과 |
|---|---:|
| route | `RouteScenario_1711_rep0` / `ParkingCutIn_1` |
| status | Completed |
| driving score | 100.0 |
| route completion | 100.0 |
| collision / red light / stop sign | 0 / 0 / 0 |
| VLM calls | 123 benchmarked calls |
| MinSpeedTest | failure |

해석:

- route completion과 composed score는 100이다.
- 다만 기존 BF16/W8A8 run과 마찬가지로 MinSpeedTest failure가 있다.
- 따라서 “주행 route는 완료했지만 MinSpeedTest 기준에서는 partial/failure”로 적는 것이 안전하다.

### 2.5 LLMCompressor GPTQ W4A16

GPTQ도 `tools/quantize_qwen3vl_w4a16.py`에서 같은 calibration data로 실행하도록 구성했다.

공통 설정:

| 항목 | 값 |
|---|---|
| method | `gptq` |
| scheme | `W4A16` |
| targets | `Linear` |
| ignored layers | `lm_head`, `re:.*visual.*` |
| max seq length | 4096 |
| block size | 128 |
| dampening frac | 0.01 |
| actorder | static |
| offload hessians | true for n64 run |

수행 상태:

| 작업 | 상태 |
|---|---|
| GPTQ W4A16 no-save smoke | 성공 |
| GPTQ W4A16 458 full probe | ETA 과다로 중단 |
| GPTQ W4A16 n64 checkpoint | 2026-05-03 20:00 KST 기준 진행 중 |
| GPTQ W4A16 n64 vLLM benchmark | 결과 대기 |

진행 중인 output path:

```text
/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16-n64
```

진행 로그:

```text
/mnt/2/carla_metric_result/qwen_gptq_w4a16_n64.log
```

문서 작성 시점에는 아직 `carla_quantization_metadata.json`이 없으므로 완료 checkpoint로 간주하지 않았다.

## 3. 현재까지의 주요 비교표

아래 표는 Bench2Drive single scenario 기준이다. `Params(B)`는 logical parameter count로 통일했다.

| Model | Quant | A6000 가능 여부 | Params(B) | Model Size(GB) | Load GPU Mem(GB) | Peak GPU Mem(GB) | Avg Latency(ms) | P95 Latency(ms) | Tokens/sec |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | BF16 | 가능 | 8.767 | 16.33 | 16.359 | 16.826 | 6032.01 | 6712.67 | 15.205 |
| Qwen3-VL-8B | INT8-bnb | 가능, 느림 | 8.767 | 16.33 | 9.387 | 10.061 | 22402.73 | 25234.15 | 3.975 |
| Qwen3-VL-8B | W8A8-INT8-vLLM | 가능 | 8.767 | 9.864 | 13.940 | 23.285 | 2031.99 | 2229.15 | 48.393 |
| Qwen3-VL-8B | AWQ-W4A16-n64-vLLM | 가능, 현재 최선 | 8.767 | 6.753 | 10.809 | 20.040 | 1589.54 | 1724.44 | 65.821 |

현재 해석:

- INT8-bnb는 메모리는 작지만 latency가 너무 나쁘다.
- W8A8-vLLM은 BF16보다 훨씬 빠르다.
- AWQ-W4A16-n64-vLLM은 현재까지 가장 좋은 8B quant 후보다.
- AWQ는 standalone vLLM에서는 약 0.92초였고, Bench2Drive에서는 remote vLLM 호출, image preprocessing, 첫 요청 warmup 등이 포함되어 평균 약 1.59초로 측정됐다.

## 4. RTX A6000에서 FP8이 왜 안 됐나

현재 서버 GPU는 NVIDIA RTX A6000이고 compute capability는 8.6이다.

FP8 W8A8 Transformer inference는 일반적으로 compute capability 8.9 이상, 예를 들어 Ada 계열 RTX 4090/RTX 6000 Ada 또는 Hopper H100 계열에서 제대로 지원된다. RTX A6000은 Ampere 계열이므로 native FP8 Transformer inference 대상으로 보기 어렵다.

실제 FP8 reference run의 로그에도 다음 경고가 남아 있다.

```text
[transformers] FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100), actual = `8.6`. We will default to dequantizing the model to bf16. Feel free to use a different quantization method like bitsandbytes or torchao
```

로그 경로:

```text
/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_FP8_idx0/eval.log
```

따라서 A6000에서 FP8 checkpoint가 로드되더라도, 실제로는 FP8 kernel로 빠르게 추론하는 것이 아니라 BF16으로 dequantize되어 돌 수 있다. 이 경우 memory/latency 이득을 native FP8의 효과로 해석하면 안 된다.

정리:

| GPU | Compute Capability | FP8 native inference 해석 |
|---|---:|---|
| RTX A6000 | 8.6 | 부적합. BF16 dequantize 가능성 큼 |
| RTX 4090 | 8.9 | 가능 후보 |
| RTX 6000 Ada | 8.9 | 가능 후보 |
| H100 | 9.0 | 가능 후보 |

이 프로젝트에서 A6000 기준 메인 quant 후보를 FP8이 아니라 W8A8-INT8, AWQ W4A16, GPTQ W4A16, BNB 4bit/8bit로 둔 이유가 여기에 있다.

## 5. 남은 작업

1. GPTQ W4A16 n64 checkpoint 완료 여부 확인
2. GPTQ W4A16 n64 vLLM load/benchmark
3. GPTQ n64와 AWQ n64의 standalone 및 single-scenario latency 비교
4. BNB 4bit Bench2Drive single scenario 실행
5. TF++ only baseline을 같은 route에서 실행
6. AWQ n64를 dev10 또는 idx0-idx9로 확장
7. `qwen_intervention.jsonl` 기반 JSON parse success, fallback count, error count 자동 집계

