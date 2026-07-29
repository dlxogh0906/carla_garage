# Qwen3-VL Quantization & Efficiency Experiment Summary

작성 기준일: 2026-05-03  
작성 기준 경로: `/mnt/2/carla_garage`

이 문서는 현재 코드베이스와 로컬 결과 파일을 기준으로 작성했다. 확인되지 않은 내용은 `확인 필요`로 표시했으며, 결과 파일이 없는 항목은 없는 상태 그대로 기록했다.

## 1. Overview

이 문서의 목적은 현재 프로젝트에서 수행된 Qwen3-VL 양자화/경량화 실험의 코드, 실행 스크립트, calibration data, 결과 파일, 하드웨어 제약을 팀원이 한 파일만 보고 이해할 수 있도록 정리하는 것이다.

현재 프로젝트는 Bench2Drive/CARLA 주행 루프에서 TransFuser++ 계열 주행 agent에 Qwen3-VL을 연결해 위험 상황, 교통 규칙, 감속/정지 판단을 보조하는 구조를 사용한다. Qwen3-VL을 주행 개입에 쓰려면 모델 호출 latency와 GPU memory가 실제 주행 루프를 막지 않아야 하므로, BF16/FP16 baseline 대비 INT8, W8A8, 4bit 계열의 경량화 실험이 필요하다.

핵심 메시지는 다음과 같다.

- Qwen3-VL-8B는 실시간 주행 개입 후보로 사용되고 있다.
- Qwen3-VL-30B-A3B는 로컬에 다운로드되어 있으며, 대형 비교군 또는 teacher 후보로 볼 수 있다.
- 현재 서버 GPU는 RTX A6000이고 compute capability는 8.6이다.
- RTX A6000 환경에서는 native FP8 W8A8 Transformer inference가 제한된다.
- 따라서 본 프로젝트의 주 실험 후보는 INT8, AWQ, GPTQ, BNB 4bit, llmcompressor W8A8-INT8 + vLLM 계열이다.
- FP8 checkpoint가 로드되더라도 A6000에서는 BF16으로 dequantize될 수 있으므로, FP8의 latency/VRAM 이득으로 해석하면 안 된다.

## 2. Hardware / Software Environment

| Item | Value | Source |
|---|---|---|
| GPU | 2 x NVIDIA RTX A6000, 각 49140 MiB | `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version` |
| Compute Capability | 8.6 | `nvidia-smi`, `torch.cuda.get_device_capability()` |
| CUDA Version | Driver CUDA 12.8, qwen_quant torch CUDA 12.8, garage_2 torch CUDA 12.4 | `nvidia-smi`, Python version probe |
| PyTorch Version | qwen_quant: 2.9.0, garage_2: 2.5.0 | `/home/kwy00/anaconda3/envs/qwen_quant/bin/python`, `/home/kwy00/anaconda3/envs/garage_2/bin/python` |
| Transformers Version | qwen_quant: 4.57.6, garage_2: 5.6.2 | Python package metadata probe |
| vLLM Version | qwen_quant: 0.12.0, garage_2: 설치 안 됨 | Python package metadata probe |
| LLMCompressor Version | qwen_quant: 0.10.0.2, garage_2: 설치 안 됨 | Python package metadata probe |
| Python Version | qwen_quant: 3.10.20, garage_2: 3.10.15 | Python version probe |

추가 메모:

- `nvcc --version`은 현재 shell에서 확인되지 않았다. CUDA toolkit 설치 여부는 확인 필요.
- 양자화와 vLLM 관련 스크립트는 주로 `/home/kwy00/anaconda3/envs/qwen_quant/bin/python`을 사용한다.
- CARLA/Bench2Drive 실행 스크립트는 주로 `/home/kwy00/anaconda3/envs/garage_2/bin/python`을 사용한다.
- 2026-05-03 기준 W4A16 AWQ/GPTQ script smoke test를 위해 qwen_quant 환경의 `compressed-tensors`를 0.12.2에서 0.14.0.1로 업데이트했다. 이 버전은 `llmcompressor==0.10.0.2` import 문제를 해결하지만, `vllm==0.12.0`의 metadata 요구사항(`compressed-tensors==0.12.2`)과는 충돌한다. `import vllm`은 확인됐지만, vLLM 실제 checkpoint load는 별도 확인 필요.

## 3. Models Considered

| Model | Path / ID | Role | Quantization Status | Notes |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct` | 실시간 주행 개입 후보 | BF16/FP16 baseline, BNB INT8, BNB 4bit, W8A8-INT8 실험 확인 | 로컬 경로 존재. 원본 디렉터리 `du -sh` 기준 33G. benchmark param storage는 약 16.33 GiB. |
| Qwen3-VL-30B-A3B-Instruct | `/mnt/2/pretrained_models/Qwen3-VL-30B-A3B-Instruct` | 대형 비교군 / teacher 후보 | 로컬 다운로드 확인, profiling/quant 결과는 발견 안 됨 | 로컬 경로 존재. `du -sh` 기준 58G. config는 `qwen3_vl_moe`. |
| Qwen3-VL-8B-FP8, if exists | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-FP8` | 참고용 | FP8 단일 시나리오 결과 로그는 존재하지만 현재 로컬 모델 디렉터리는 발견 안 됨 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_FP8_idx0/eval.log`에서 FP8 dequantize 경고 확인. 실제 checkpoint 경로는 확인 필요. |
| Qwen3-VL-30B-A3B-FP8, if exists | 확인 필요 | 참고용 | 발견 안 됨 | 로컬 FP8 경로 및 결과 파일 발견 안 됨. |

추가로 발견된 quantized checkpoint:

| Model | Path / ID | Role | Quantization Status | Notes |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct-W8A8-INT8 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8` | INT8 압축 checkpoint | llmcompressor W8A8-INT8 생성 완료 | `carla_quantization_metadata.json`, `recipe.yaml` 존재. safetensors 합계 약 9.9G. |
| Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012` | vLLM 0.12용 W8A8-INT8 실행 경로 | compressed-tensors + vLLM benchmark와 Bench2Drive 결과 확인 | safetensors는 W8A8-INT8 경로로 symlink. 디렉터리 자체 `du -sh`는 16M이나 resolved model size는 약 9.88 GiB. |
| Qwen3-VL-8B-Instruct-AWQ-W4A16-n4 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n4` | AWQ 4bit 실행 가능성 확인용 checkpoint | llmcompressor AWQ W4A16 생성 완료, vLLM load/benchmark 확인 | calibration sample 4개 smoke checkpoint. 디렉터리 `du -sh` 기준 6.8G. Transformers 경로는 매우 느리지만 vLLM Marlin kernel 경로는 정상 수치 확인. |

## 4. Quantization Methods

| Method | Scheme | A6000 Feasibility | Used Already? | Purpose | Notes |
|---|---|---|---|---|---|
| BF16/FP16 | baseline | 가능 | 예 | 원본 기준 | standalone benchmark와 Bench2Drive 단일 시나리오 결과 확인. |
| INT8 | W8A8 or BNB INT8 | 가능 | 예 | 8B 경량화 | BNB INT8 runtime, llmcompressor W8A8-INT8 checkpoint, vLLM compressed-tensors 실행 확인. |
| AWQ | W4A16 | 가능성 높음 | 예, n4 checkpoint/vLLM 확인 | 4bit 압축 | `tools/quantize_qwen3vl_w4a16.py` 추가. 458-sample full run은 ETA 과다로 중단. n4 checkpoint는 vLLM에서 Marlin kernel 사용 확인. |
| GPTQ | W4A16 또는 W8A8 recipe 내부 modifier | 가능성 높음 | 코드/smoke만 확인, W8A8에는 사용 | 4bit 압축 또는 W8A8 calibration quant | W4A16 script 추가. 458-sample full run은 ETA 과다로 중단, checkpoint 없음. |
| BNB 4bit | NF4/INT4 | 가능성 높음 | 예, standalone benchmark | 빠른 4bit 실험 | `benchmark_qwen_vl.py`와 `qwen_client.py`에서 NF4 runtime loading 지원. Bench2Drive 단일 시나리오 결과는 발견 안 됨. |
| FP8 | W8A8 | A6000 native 불가 | 참고 실험만 확인 | 참고용 | compute capability 8.9+ 필요. A6000에서는 BF16 dequantize 경고 확인. |
| FP8 checkpoint dequantized | FP8 -> BF16 | 실행 가능할 수 있음 | 예, 로그 확인 | 참고용 | FP8 성능 이득으로 해석하면 안 됨. 현재 FP8 checkpoint 디렉터리는 발견 안 됨. |

RTX A6000에서 FP8 checkpoint가 로드되더라도 native FP8 inference가 아닐 수 있다. 실제 로그에 다음 경고가 남아 있다.

```text
[transformers] FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100), actual = `8.6`. We will default to dequantizing the model to bf16.
```

로그 경로: `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_FP8_idx0/eval.log`

## 5. Dataset / Calibration Data

| Dataset | Split | Purpose | Num Samples | Max Seq Length | Preprocessing | File / Code Location |
|---|---|---|---:|---:|---|---|
| LGAI-EXAONE/KoMT-Bench | train | calibration 후보 | 확인 필요 | 확인 필요 | 현재 코드/결과에서 사용 흔적 발견 안 됨 | 발견 안 됨 |
| LGAI-EXAONE/MANTA-1M, if used | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 현재 코드/결과에서 사용 흔적 발견 안 됨 | 발견 안 됨 |
| Bench2Drive frames / QwenVLMClient calibration dump | route idx0-9 | W8A8-INT8 calibration | 458 | 4096 | Qwen client가 image와 prompt/chat template를 JSONL로 저장, quantizer가 `chat_template_text`와 image를 로드 | `/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl`, `src/garage_ext/vlm_intervention/qwen_client.py`, `tools/quantize_qwen3vl_w8a8.py` |
| Bench2Drive frames, if used | single route idx0 | VLM profiling | 모델별 호출 수 상이 | runtime별 상이 | `qwen_sensor_agent.py`가 주행 중 Qwen 호출, `qwen_client.py`가 benchmark fields 기록 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/*/qwen_intervention.jsonl` |

확인된 calibration data 상세:

- summary file: `/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_summary.txt`
- merged JSONL: `/mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl`
- sample 수: 458 rows
- image 수: 458 files, missing image 0건으로 확인
- route 범위: idx0-idx9, 총 10개 route
- prompt mode 분포: speed 419, traffic_rule 39
- trigger 분포: object 398, ttc 21, traffic_rule 39
- recorded input token 범위: min 1300, max 1911, mean 약 1716.93
- W8A8 quantization metadata의 실제 `num_calibration_samples`: 458
- W8A8 quantization metadata의 `max_seq_length`: 4096

`calibration_texts`라는 별도 변수명은 현재 코드에서 발견되지 않았다. 현재 구현에서는 `qwen_client.py`가 `QWEN_SAVE_CALIB=1`일 때 각 sample에 `chat_template_text`를 저장하고, `tools/quantize_qwen3vl_w8a8.py`가 이를 `text` column으로 매핑해서 llmcompressor calibration dataset으로 사용한다.

## 6. Existing Quantization Code

| File | Purpose | Key Functions / Classes | Quantization Method | Output |
|---|---|---|---|---|
| `tools/quantize_qwen3vl_w8a8.py` | Qwen3-VL-8B W8A8-INT8 checkpoint 생성 | `parse_args`, `load_calibration_dataset`, `main`, `SmoothQuantModifier`, `GPTQModifier`, `oneshot()` | llmcompressor SmoothQuant + GPTQModifier, scheme W8A8 | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8`, metadata JSON, recipe, logs |
| `tools/quantize_qwen3vl_w4a16.py` | Qwen3-VL-8B AWQ/GPTQ W4A16 checkpoint 생성용 공용 스크립트 | `parse_args`, `load_calibration_dataset`, `build_modifier`, `AWQModifier`, `GPTQModifier`, `oneshot()` | llmcompressor AWQ W4A16_ASYM 또는 GPTQ W4A16 | 기본 output은 `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16`, `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16`. 현재 full checkpoint 없음 |
| `src/garage_ext/vlm_intervention/qwen_client.py` | Qwen model loading, runtime quant, VLM 호출, calibration dump, benchmark logging | `QwenVLMClient`, `_load_model`, `_dump_calibration_sample`, `_detect_quant_method` | BF16/FP16, BNB INT8, BNB 4bit, remote/vLLM backend, FP8 runtime warning detection | `qwen_intervention.jsonl`, calibration JSONL/images |
| `tools/benchmark_qwen_vl.py` | standalone Transformers benchmark | `BitsAndBytesConfig`, `AutoModelForImageTextToText`, `Qwen3VLForConditionalGeneration`, latency/memory measurement | none, bnb8, bnb4, auto-detected AWQ/GPTQ/FP8/compressed-tensors | JSON/CSV benchmark files |
| `tools/benchmark_qwen_vllm.py` | standalone vLLM benchmark | `LLM`, `SamplingParams`, NVML memory measurement | vLLM `compressed-tensors`, W8A8-INT8 | JSON/CSV benchmark files |
| `tools/run_qwen_quant_benchmarks.py` | 여러 Qwen checkpoint benchmark 실행 | `run_one`, `parse_args` | BF16/FP16, BNB runtime quant, auto quantized checkpoint | `summary.csv` |
| `tools/summarize_qwen_runtime.py` | Bench2Drive Qwen runtime 결과 요약 | `summarize`, `build_rows`, `write_csv` | quant method label 정리, FP8 dequantized label 처리 | `qwen_runtime_calls.csv`, `qwen_runtime_summary.csv`, `qwen_runtime_paper_table.csv` |
| `run_qwen_single_scenario_benchmark.sh` | Bench2Drive 단일 route Qwen runtime benchmark | env 기반 실행, single route XML 생성, summary script 호출 | BF16, BNB INT8, BNB 4bit, FP8 reference label | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/<label>/` |
| `run_qwen_single_scenario_benchmark_w8a8_vllm.sh` | vLLM OpenAI-compatible server + Bench2Drive 단일 route 실행 | `python -m vllm.entrypoints.openai.api_server`, `QWEN_VLM_BACKEND=vllm_openai` | W8A8-INT8 compressed-tensors via vLLM | `Qwen3-VL-8B_W8A8-INT8-vLLM_idx0` 결과 |
| `run_qwen_dev10.sh` | Bench2Drive dev10 Qwen sensor agent 실행 | `leaderboard_evaluator_local.py`, `qwen_sensor_agent.py` | runtime Qwen backend 설정에 따름 | eval log/json, viz, qwen_intervention |

핵심 설정값:

```text
MODEL_ID default: /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct
OUT_DIR default: /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8
CALIB_JSONL default: /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl
NUM_CALIBRATION_SAMPLES default: 512
실제 W8A8 metadata num_calibration_samples: 458
MAX_SEQUENCE_LENGTH: 4096
SCHEME: W8A8
TARGETS: Linear
IGNORE: lm_head, re:.*visual.*
SmoothQuant smoothing_strength: 0.8
quantize_visual: false

W4A16 script defaults:
METHOD: awq or gptq
AWQ OUT_DIR default: /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16
GPTQ OUT_DIR default: /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16
W4A16 NUM_CALIBRATION_SAMPLES default: 458
W4A16 MAX_SEQUENCE_LENGTH: 4096
AWQ SCHEME default: W4A16_ASYM
GPTQ SCHEME default: W4A16
TARGETS: Linear
IGNORE: lm_head, re:.*visual.*
GPTQ block_size: 128
GPTQ dampening_frac: 0.01
GPTQ actorder: static
```

추가 확인 사항:

- `QuantizationModifier` 사용 코드는 발견 안 됨.
- `AWQModifier`는 `tools/quantize_qwen3vl_w4a16.py`에서 사용됨.
- `GPTQModifier`는 W8A8-INT8 생성 recipe와 W4A16 GPTQ script에서 사용됨.
- zip 생성 코드는 발견 안 됨.
- KV cache 관련 설정은 `tools/benchmark_qwen_vllm.py`의 `--kv-cache-memory-gib`와 `run_qwen_single_scenario_benchmark_w8a8_vllm.sh`의 `VLLM_KV_CACHE_MEMORY_BYTES=1073741824`에서 확인됨.

## 7. Experiment Results Found

| Experiment | Model | Quant | Dataset | Output Dir | Result File | Status | Main Findings |
|---|---|---|---|---|---|---|---|
| standalone Transformers benchmark | Qwen3-VL-8B | FP16/BF16 baseline | sample image/prompt | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_fp16.json` | success | params 8.767B, param storage 16.33 GiB, max allocated 16.77 GiB, avg latency 1.17s, tokens/sec 16.23 |
| standalone Transformers benchmark | Qwen3-VL-8B | BNB INT8 | sample image/prompt | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_bnb8.json` | success | param storage 9.33 GiB, max allocated 9.92 GiB, avg latency 3.51s, tokens/sec 5.42 |
| standalone Transformers benchmark | Qwen3-VL-8B | BNB 4bit NF4 | sample image/prompt | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_bnb4.json` | success | max allocated 6.56 GiB, avg latency 2.55s, tokens/sec 10.19. BNB packed representation 때문에 reported params 해석 주의 |
| standalone Transformers benchmark | Qwen3-VL-8B | W8A8-INT8 checkpoint via Transformers | sample image/prompt | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_w8a8_int8.csv` | partial | avg latency 8.02s, peak allocated 17.10 GiB. Transformers 경로에서는 압축 runtime 이득이 확인되지 않음 |
| standalone vLLM benchmark | Qwen3-VL-8B | W8A8-INT8 vLLM compressed-tensors | calibration sample index 0 | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_w8a8_int8_vllm.json`, `.csv` | success | resolved model size 9.879 GiB, load GPU mem 14.13 GiB, avg latency 1402.63 ms, P95 1406.52 ms, tokens/sec 62.03 |
| Bench2Drive single scenario | Qwen3-VL-8B | BF16 | route idx0, ParkingCutIn_1 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_BF16_idx0` | `qwen_runtime_paper_table.csv`, `eval.json`, `eval.log` | partial | route completion 100, composed score 100, Qwen calls 52, avg latency 6032.01 ms, P95 6712.67 ms, peak memory 16.826 GB. eval.log는 MinSpeedTest failure 표시 |
| Bench2Drive single scenario | Qwen3-VL-8B | FP8 reference, dequantized | route idx0, ParkingCutIn_1 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_FP8_idx0` | `qwen_runtime_paper_table.csv`, `eval.log` | partial | route completion 100, Qwen calls 53, avg latency 5978.92 ms, peak memory 16.874 GB. A6000에서 BF16 dequantize 경고 확인 |
| Bench2Drive single scenario | Qwen3-VL-8B | INT8-bnb | route idx0, ParkingCutIn_1 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_INT8-bnb_idx0` | `qwen_runtime_paper_table.csv`, `eval.json` | partial | route completion 100, Qwen calls 14, load memory 9.387 GB, peak memory 10.061 GB, avg latency 22402.73 ms. eval.log는 MinSpeedTest failure 표시 |
| Bench2Drive single scenario | Qwen3-VL-8B | W8A8-INT8-vLLM | route idx0, ParkingCutIn_1 | `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_W8A8-INT8-vLLM_idx0` | `qwen_runtime_paper_table.csv`, `eval.json`, `eval.log` | partial | route completion 100, Qwen calls 120, model size 9.864 GB, load memory 13.94 GB, peak memory 23.285 GB, avg latency 2031.99 ms, P95 2229.15 ms, tokens/sec 48.393. eval.log는 MinSpeedTest failure 표시 |
| quantization smoke | Qwen3-VL-8B | AWQ W4A16 | calibration JSONL sample 1 | 없음 | no-save smoke command | success | `tools/quantize_qwen3vl_w4a16.py --method awq --num-samples 1 --no-save` 통과. checkpoint 저장 안 함 |
| quantization smoke | Qwen3-VL-8B | GPTQ W4A16 | calibration JSONL sample 1 | 없음 | no-save smoke command | success | `tools/quantize_qwen3vl_w4a16.py --method gptq --num-samples 1 --no-save --offload-hessians` 통과. checkpoint 저장 안 함 |
| standalone Transformers benchmark | Qwen3-VL-8B | AWQ W4A16 n4 | calibration JSONL sample 0 | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_awq_w4a16_n4.json`, `.csv` | fail for runtime | checkpoint load는 성공했지만 peak memory 20.56 GiB, avg latency 53859.84 ms, tokens/sec 0.353. Transformers compressed-tensors 경로는 실사용 부적합 |
| standalone vLLM benchmark | Qwen3-VL-8B | AWQ W4A16 n4 | calibration JSONL sample 0 | `/mnt/2/carla_metric_result/qwen_benchmarks` | `qwen3vl_8b_awq_w4a16_n4_vllm.json`, `.csv` | success | vLLM에서 `MarlinLinearKernel for CompressedTensorsWNA16` 사용 확인. model size 6.769 GiB, load/peak GPU mem 10.997 GiB, avg latency 902.19 ms, P95 903.44 ms, tokens/sec 100.87 |
| quantization full probe | Qwen3-VL-8B | AWQ W4A16 | calibration JSONL 458 samples | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16` | `/mnt/2/carla_metric_result/qwen_awq_w4a16_quantize_8b.log` | aborted | cache 준비 458/458 완료 후 `(1/37): Calibrating`에서 sample 5까지 약 1분 38초, 첫 subgraph ETA 약 2시간대라 중단. checkpoint 없음 |
| quantization full probe | Qwen3-VL-8B | GPTQ W4A16 | calibration JSONL 458 samples | `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16` | `/mnt/2/carla_metric_result/qwen_gptq_w4a16_quantize_8b.log` | aborted | cache 준비 458/458 완료 후 `(1/37): Calibrating`에서 sample 5까지 약 1분 28초, 첫 subgraph ETA 약 2시간대라 중단. checkpoint 없음 |
| local checkpoint only | Qwen3-VL-30B-A3B | BF16 original | 없음 | `/mnt/2/pretrained_models/Qwen3-VL-30B-A3B-Instruct` | config files only | no profiling result | 로컬 모델은 있으나 benchmark/result 파일 발견 안 됨 |

찾은 지표 요약:

- model size: BF16 8B는 benchmark 기준 약 16.33 GiB, W8A8-vLLM resolved size 약 9.88 GiB, AWQ n4 vLLM resolved size 약 6.77 GiB.
- parameter count: Qwen3-VL-8B benchmark 기준 약 8.767B.
- GPU memory: BF16 single scenario peak 약 16.826 GB, BNB INT8 peak 약 10.061 GB, W8A8-vLLM peak 약 23.285 GB, AWQ n4 vLLM standalone peak 약 10.997 GiB.
- inference time: BF16 single scenario avg 약 6032 ms, BNB INT8 avg 약 22403 ms, W8A8-vLLM avg 약 2032 ms, AWQ n4 vLLM standalone avg 약 902 ms.
- tokens/sec: BF16 single scenario 약 15.205, BNB INT8 약 3.975, W8A8-vLLM 약 48.393, AWQ n4 vLLM standalone 약 100.87.
- JSON success rate: summary 파일에서 별도 집계값 발견 안 됨. 확인 필요.
- error count: Qwen JSON parse error count 집계 파일 발견 안 됨. 확인 필요.
- OOM 여부: 검토한 결과 로그에서 OOM 증거는 발견 안 됨.
- FP8 dequantization 경고: `/mnt/2/carla_metric_result/qwen_runtime_single_scenario/Qwen3-VL-8B_FP8_idx0/eval.log`에서 확인.
- AWQ/GPTQ W4A16 checkpoint: AWQ n4 checkpoint는 존재하고 vLLM benchmark 성공. AWQ/GPTQ 458-sample full checkpoint는 아직 없음. llmcompressor full probe는 ETA 과다로 중단했으며 partial output directory는 제거함.

Bench2Drive 결과 해석 주의:

- 위 단일 시나리오 결과들은 `eval.json`에서 route completion 100%, composed score 100으로 기록되어 있다.
- 동시에 `eval.log`에는 `MinSpeedTest FAILURE`로 인해 route headline이 `FAILURE`로 표시된다.
- 따라서 이 결과는 “주행 route는 완료했지만 MinSpeedTest 기준은 실패한 partial result”로 해석하는 것이 안전하다.

## 8. RTX A6000 and FP8 Issue

RTX A6000은 Ampere 계열 GPU이며 compute capability는 8.6이다. native FP8 W8A8 Transformer inference는 일반적으로 compute capability 8.9 이상, 예를 들어 Ada RTX 4090/RTX 6000 Ada 또는 Hopper H100 계열에서 지원된다.

따라서 RTX A6000에서 FP8 checkpoint가 로드되더라도 실제 runtime은 BF16으로 dequantize되거나 weight-only/호환 경로로 동작할 수 있다. 이 경우 latency나 VRAM 감소가 FP8 native inference의 이득이라고 해석하면 안 된다. 본 프로젝트에서 FP8은 메인 경량화 결과가 아니라 참고 실험 또는 추가 하드웨어 환경 실험으로 두는 것이 적절하다.

| GPU | Architecture | Compute Capability | Native FP8 W8A8 Support | Notes |
|---|---|---:|---|---|
| RTX A6000 | Ampere | 8.6 | No | 현재 서버 |
| RTX 6000 Ada | Ada | 8.9 | Yes | 이름이 비슷하지만 다른 GPU |
| RTX 4090 | Ada | 8.9 | Yes | FP8 가능 |
| H100 | Hopper | 9.0 | Yes | FP8 가능 |

## 9. Metrics to Report

| Category | Metric | Unit | Priority | Why It Matters |
|---|---|---:|---|---|
| Model Size | Model Disk Size | GB | High | 양자화 저장 용량 감소 |
| Model Size | Parameter Count | B | Medium | 모델 규모 비교 |
| Memory | Load GPU Memory | GB | High | 모델 적재 비용 |
| Memory | Peak GPU Memory | GB | Highest | 실제 추론/주행 가능성 |
| Latency | Avg Inference Time | ms | Highest | 실시간성 핵심 |
| Latency | P95 Inference Time | ms | High | worst-case 지연 |
| Throughput | Tokens/sec | tokens/s | Medium | 생성 속도 |
| Runtime | FPS / Step/sec | frame/s | High | 실제 주행 루프 속도 |
| Runtime | VLM Call Count | count | High | 호출 비용 |
| Runtime | VLM Call Rate | % | High | 선택 호출 효과 |
| Output Stability | JSON Success Rate | % | High | 제어 연결 안정성 |
| Driving | Driving Score | score | High | 주행 성능 |
| Driving | Route Completion | % | High | 완주성 |
| Safety | Collision Count | count | High | 안전성 |
| Compute | FLOPs | FLOPs | Low | 보조 지표 |

FLOPs는 보조 지표로만 사용하는 것이 좋다. 현재 `tools/benchmark_qwen_vl.py`도 FLOPs를 `2 * params * generated_tokens` 형태의 rough proxy로 계산하며, 양자화는 이론 FLOPs보다 weight/activation bit-width, kernel, memory bandwidth, KV cache, serving backend 영향을 크게 받는다. 실제 주행에서는 latency, peak GPU memory, FPS, VLM call rate, JSON success rate가 더 중요한 핵심 지표다.

## 10. Recommended Experiment Table to Fill

### 10.1 Model Quantization Profiling Table

| Model | Quant | Params(B) | Model Size(GB) | Load GPU Mem(GB) | Peak GPU Mem(GB) | Avg Latency(ms) | P95 Latency(ms) | Tokens/sec | JSON Success(%) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-VL-8B | BF16/FP16 |  |  |  |  |  |  |  |  |  |
| Qwen3-VL-8B | INT8 |  |  |  |  |  |  |  |  |  |
| Qwen3-VL-8B | AWQ/GPTQ 4bit |  |  |  |  |  |  |  |  |  |
| Qwen3-VL-30B-A3B | BF16/FP16 |  |  |  |  |  |  |  |  |  |
| Qwen3-VL-30B-A3B | INT8/4bit |  |  |  |  |  |  |  |  |  |
| Qwen3-VL-30B-A3B | FP8 checkpoint |  |  |  |  |  |  |  |  | 참고용 |

### 10.2 Bench2Drive One-Scenario Table

| Method | Model | Quant | Scenario / Route | DS | RC | Collision | Steps | FPS | VLM Calls | Call Rate(%) | Avg VLM Latency(ms) | Peak GPU Mem(GB) | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TF++ only | None | - |  |  |  |  |  | 0 | 0 | - |  |  |  |
| TF++ + VLM | Qwen3-VL-8B | BF16/FP16 |  |  |  |  |  |  |  |  |  |  |  |
| TF++ + VLM | Qwen3-VL-8B | INT8 |  |  |  |  |  |  |  |  |  |  |  |
| TF++ + VLM | Qwen3-VL-30B-A3B | BF16/FP16 |  |  |  |  |  |  |  |  |  |  |  |

## 11. What Has Been Done So Far

- 완료:
  - [x] Qwen3-VL-8B 원본 checkpoint 로컬 다운로드 확인
  - [x] Qwen3-VL-30B-A3B 원본 checkpoint 로컬 다운로드 확인
  - [x] Qwen3-VL-8B BF16/FP16 standalone profiling 결과 확인
  - [x] Qwen3-VL-8B BNB INT8 standalone profiling 결과 확인
  - [x] Qwen3-VL-8B BNB 4bit standalone profiling 결과 확인
  - [x] Bench2Drive 기반 W8A8 calibration dataset 준비 확인
  - [x] Qwen3-VL-8B W8A8-INT8 llmcompressor 양자화 checkpoint 생성 확인
  - [x] Qwen3-VL-8B W8A8-INT8 vLLM standalone profiling 결과 확인
  - [x] Qwen3-VL-8B BF16 Bench2Drive one-scenario profiling 결과 확인
  - [x] Qwen3-VL-8B BNB INT8 Bench2Drive one-scenario profiling 결과 확인
  - [x] Qwen3-VL-8B W8A8-INT8-vLLM Bench2Drive one-scenario profiling 결과 확인
  - [x] RTX A6000에서 FP8 dequantize 경고 확인
  - [x] Qwen3-VL-8B AWQ/GPTQ W4A16 공용 스크립트 작성
  - [x] Qwen3-VL-8B AWQ W4A16 no-save smoke test
  - [x] Qwen3-VL-8B GPTQ W4A16 no-save smoke test
  - [x] Qwen3-VL-8B AWQ/GPTQ W4A16 458-sample full probe 실행 및 ETA 과다 확인
  - [x] Qwen3-VL-8B AWQ W4A16 n4 checkpoint 생성
  - [x] Qwen3-VL-8B AWQ W4A16 n4 vLLM load/standalone profiling
  - [ ] Qwen3-VL-30B-A3B BF16 짧은 profiling 결과
  - [ ] Qwen3-VL-30B-A3B INT8/4bit 양자화 결과
  - [ ] AWQ 4bit 32/64+ sample checkpoint 생성
  - [ ] GPTQ 4bit checkpoint 생성
  - [ ] GPTQ W4A16 checkpoint vLLM load 확인
  - [ ] native FP8 지원 GPU에서 FP8 latency/VRAM 실험
  - [ ] JSON success rate 자동 집계
  - [ ] 전체 dev10 또는 다중 route quant 비교표

## 12. Remaining Tasks

1. Qwen3-VL-8B BF16, BNB INT8, BNB 4bit, W8A8-vLLM 지표를 같은 prompt/image 조건과 같은 output schema로 재정렬한다.
2. JSON success rate, parse error count, fallback count를 `qwen_intervention.jsonl`에서 자동 집계한다.
3. Qwen3-VL-8B BNB 4bit를 Bench2Drive one-scenario에서 실행해 BF16/INT8/W8A8-vLLM과 비교한다.
4. Qwen3-VL-8B AWQ W4A16은 n4 vLLM 결과가 좋으므로 n32 또는 n64 checkpoint를 생성해 같은 vLLM benchmark를 반복한다.
5. 4bit checkpoint를 만들 경우 output path에 sample 수를 명시하거나 metadata에 `num_calibration_samples`를 반드시 남긴다.
6. AWQ/GPTQ 458-sample full checkpoint는 ETA가 과도하므로 필요성이 생기기 전까지 보류한다.
7. GPTQ W4A16도 작은 sample checkpoint를 생성한 뒤 vLLM load 가능성을 확인한다.
8. llmcompressor W4A16이 n32/n64에서도 과도하게 느리면 AutoAWQ/GPTQModel/optimum 등 별도 backend를 qwen_quant와 분리된 환경에서 검토한다.
9. Qwen3-VL-8B W8A8-vLLM의 peak GPU memory가 높은 원인을 vLLM KV cache, encoder cache, `gpu_memory_utilization`, `kv_cache_memory_bytes` 설정별로 분리 측정한다.
10. Qwen3-VL-30B-A3B BF16 로딩과 짧은 standalone profiling을 먼저 수행한다.
11. Qwen3-VL-30B-A3B INT8/4bit 가능성을 확인한다.
12. Bench2Drive 한 시나리오에서 TF++ only baseline을 같은 route로 실행한다.
13. Bench2Drive 한 시나리오에서 TF++ + VLM 결과를 route completion, collision, min-speed failure까지 함께 비교한다.
14. `summary.csv`와 `summary.md`를 자동 생성하는 reporting script를 정리한다.
15. FP8 checkpoint 경로 `/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-FP8`의 현재 존재 여부와 생성 절차를 확인한다.
16. native FP8은 RTX 4090, RTX 6000 Ada, H100 등 compute capability 8.9+ 환경에서만 별도 실험으로 분리한다.

## 13. How to Reproduce

아래 명령어 중 결과 파일이 이미 존재하는 항목은 실제 코드베이스에 있는 스크립트 기준이다. 30B profiling 명령은 현재 결과 파일이 발견되지 않았으므로 제안안이며 실행 검증 필요하다.

```bash
cd /mnt/2/carla_garage

# 8B W8A8-INT8 quantization with llmcompressor
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w8a8.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8 \
  --num-samples 458 \
  --max-seq-length 4096

# 8B AWQ W4A16 smoke test, no checkpoint save
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w4a16.py \
  --method awq \
  --num-samples 1 \
  --no-save

# 8B AWQ W4A16 n4 checkpoint creation
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w4a16.py \
  --method awq \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n4 \
  --num-samples 4 \
  --max-seq-length 4096

# 8B AWQ W4A16 n4 vLLM profiling
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vllm.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n4 \
  --label Qwen3-VL-8B \
  --quant-label AWQ-W4A16-n4 \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --sample-index 0 \
  --gpu-index 1 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --kv-cache-memory-gib 1.0 \
  --gpu-memory-utilization 0.82 \
  --max-new-tokens 128 \
  --warmup 2 \
  --repeat 10 \
  --output /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_awq_w4a16_n4_vllm.json \
  --output-csv /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_awq_w4a16_n4_vllm.csv

# 8B GPTQ W4A16 smoke test, no checkpoint save
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w4a16.py \
  --method gptq \
  --num-samples 1 \
  --offload-hessians \
  --no-save

# 8B AWQ W4A16 full command.
# 주의: 458-sample probe는 ETA 과다로 중단됨. 실행 전 sample 수 축소 또는 별도 backend 검토 필요.
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w4a16.py \
  --method awq \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16 \
  --num-samples 458 \
  --max-seq-length 4096

# 8B GPTQ W4A16 full command.
# 주의: 458-sample probe는 ETA 과다로 중단됨. 실행 전 sample 수 축소 또는 별도 backend 검토 필요.
CUDA_VISIBLE_DEVICES=1 /home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/quantize_qwen3vl_w4a16.py \
  --method gptq \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --output-dir /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16 \
  --num-samples 458 \
  --max-seq-length 4096 \
  --offload-hessians

# 8B BF16/FP16 standalone profiling
/home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vl.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --label qwen3vl_8b_fp16 \
  --quant none \
  --dtype float16 \
  --device cuda:0 \
  --warmup 2 \
  --repeat 10 \
  --max-new-tokens 128 \
  --output /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_fp16.json

# 8B BNB INT8 standalone profiling
/home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vl.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --label qwen3vl_8b_bnb8 \
  --quant bnb8 \
  --dtype float16 \
  --device cuda:0 \
  --warmup 2 \
  --repeat 10 \
  --max-new-tokens 128 \
  --output /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_bnb8.json

# 8B BNB 4bit standalone profiling
/home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vl.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
  --label qwen3vl_8b_bnb4 \
  --quant bnb4 \
  --dtype float16 \
  --device cuda:0 \
  --warmup 2 \
  --repeat 10 \
  --max-new-tokens 128 \
  --output /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_bnb4.json

# 8B W8A8-INT8 vLLM standalone profiling
/home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vllm.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012 \
  --label Qwen3-VL-8B \
  --calib-jsonl /mnt/2/carla_metric_result/qwen_w8a8_calib/Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl \
  --sample-index 0 \
  --gpu-index 1 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --kv-cache-memory-gib 1.0 \
  --gpu-memory-utilization 0.82 \
  --output-json /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_w8a8_int8_vllm.json \
  --output-csv /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_8b_w8a8_int8_vllm.csv

# Bench2Drive one-scenario BF16 run
QWEN_MODEL=/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
QWEN_MODEL_LABEL=Qwen3-VL-8B \
QWEN_QUANT=BF16 \
ROUTE_INDEX=0 \
bash run_qwen_single_scenario_benchmark.sh

# Bench2Drive one-scenario BNB INT8 run
QWEN_MODEL=/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct \
QWEN_MODEL_LABEL=Qwen3-VL-8B \
QWEN_QUANT=INT8-bnb \
QWEN_RUNTIME_QUANT=bnb8 \
ROUTE_INDEX=0 \
bash run_qwen_single_scenario_benchmark.sh

# Bench2Drive one-scenario W8A8-INT8 vLLM run
ROUTE_INDEX=0 \
bash run_qwen_single_scenario_benchmark_w8a8_vllm.sh
```

아래 명령어는 제안안이며 실행 검증 필요:

```bash
cd /mnt/2/carla_garage

# 30B-A3B short profiling proposal
/home/kwy00/anaconda3/envs/qwen_quant/bin/python \
  tools/benchmark_qwen_vl.py \
  --model /mnt/2/pretrained_models/Qwen3-VL-30B-A3B-Instruct \
  --label qwen3vl_30b_a3b_bf16_short \
  --quant none \
  --dtype bfloat16 \
  --device-map auto \
  --warmup 0 \
  --repeat 1 \
  --max-new-tokens 32 \
  --output /mnt/2/carla_metric_result/qwen_benchmarks/qwen3vl_30b_a3b_bf16_short.json
```
