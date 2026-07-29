# RAViD — VLA 기반 설명 가능한 위험 대응 자율주행 시스템

> **조선대학교 AI·SW학부 산학프로젝트 2팀 자비스(JARVIS)** · 2026.06 최종 발표
> [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage) (TransFuser++, CARLA Leaderboard 2.0 스타터킷) 포크 위에 팀 연구 레이어를 얹은 저장소입니다.

TransFuser++ 계열 E2E 주행 모델은 빠르지만 **왜 그렇게 판단했는지 설명하지 못하고**, 신호등·보행자 같은 규칙/위험 상황에서 취약합니다. RAViD는 주행 백본을 개량하고(**DTF**), 위험 상황에서만 **VLA(Qwen3-VL-8B)** 가 개입해 행동을 보정하며, 모든 판단 근거를 자연어로 남기는 **설명 가능한 위험 대응** 구조를 제안합니다.

<p align="center"><img src="assets/ravid/architecture.png" alt="RAViD architecture" width="850"/></p>

## 시스템 구성

| 구성요소 | 내용 |
|---|---|
| **DTF 백본** | TF++의 transformer fusion을 **Deformable Cross-Attention** 기반으로 개량 — 주행 판단 핵심 영역 중심의 image-BEV 특징 융합 |
| **Condition-Aware Enhancement** | 저조도·과노출·안개·흐림을 자동 판별·보정하는 카메라 전처리 모듈 (`classic_cv`) |
| **TTC Gate + VLA 개입** | TTC > 3s면 백본 계획 유지, TTC ≤ 3s면 Qwen3-VL-8B가 meta-action(감속·정지·차선변경 등)과 판단 근거를 생성해 행동 보정. 후방 카메라 입력 포함 |
| **VQA-LoRA** | SimLingo VQA 서브셋(Town12 100만 프레임, 2,800만 QA)으로 주행 도메인 LoRA 파인튜닝 |
| **GAWQ (제안)** | GPTQ 위에 **중요 레이어만 W8로 보호**하고 나머지를 W4로 양자화하는 layer-wise mixed-precision 기법 |
| **Layer-wise Merge (제안)** | LoRA 병합 시 레이어별 α 가중으로 병합하는 기법 |
| **설명가능성 UI** | 주행 근거 대시보드(JARVIS 웹) + 차량번호 로그인 기반 주행 리포트 앱(RAViD) |

## 주요 결과

평가: **Bench2Drive** (CARLA 0.9.15) · 지표: DS(Driving Score) / RC(Route Completion) / SR(Success Rate) · 환경: RTX 4090 24GB (Vast.ai)

**① 백본 개량 — Bench2Drive 220 (SOTA 비교, 상위권 발췌)**

| Method | DS ↑ | RC ↑ | SR ↑ |
|---|---|---|---|
| UniAD | 45.81 | 71.68 | 16.36 |
| ORION | 77.74 | 90.84 | 54.62 |
| AutoVLA | 78.84 | 92.31 | 57.73 |
| TF++ | 84.21 | 95.25 | 67.27 |
| **DTF (Ours)** | **86.73** | **96.82** | **70.45** |

**② 구성요소 누적 기여 (dev10 ablation)** — Baseline DTF 76.84 → **최종 93.53**

| 단계 | 추가 요소 | DS ↑ | RC ↑ | SR ↑ |
|---|---|---|---|---|
| Baseline | DTF | 76.84 | 92.37 | 50 |
| v1 | + 이미지 보정 | 81.43 | 95.02 | 60 |
| v2 | + VLA 위험 추론 | 87.12 | 100 | 70 |
| v3 | + 후방 카메라 | 90.64 | 100 | 70 |
| v4 | + VQA-LoRA | 92.82 | 100 | 70 |
| v5 | + GAWQ 양자화 | 91.38 | 100 | 60 |
| **v6** | **+ Layer-wise Merge** | **93.53** | **100** | **80** |

**③ GAWQ 양자화 — 성능 유지하며 실시간성 확보 (dev10)**

| 방식 | DS ↑ | RC ↑ | Avg Latency(ms) ↓ | VLM FPS ↑ | GPU Mem(GB) ↓ | 크기(GB) ↓ |
|---|---|---|---|---|---|---|
| Raw BF16 | 92.82 | 100 | 742.35 | 1.347 | 21.49 | 18.21 |
| W8A8-INT8 | 82.36 | 94.18 | **378.92** | **2.639** | 14.13 | 10.24 |
| AWQ-W4A16 | 85.14 | 95.07 | 512.68 | 1.951 | 11.36 | 7.18 |
| GPTQ-W4A16 | 86.28 | 95.84 | 421.35 | 2.373 | **10.94** | **6.91** |
| **GAWQ (Ours)** | **91.38** | **100** | 399.06 | 2.506 | 11.28 | 7.01 |

- DTF 백본: 기존 TF++ 대비 주행 성능 **+6.41%** 개선
- GAWQ: BF16 대비 DS를 거의 유지(−1.44)하면서 **추론 지연 42.2% 감소·GPU 메모리 47.5% 감소** — 4bit 균일 양자화(GPTQ/AWQ)의 DS 하락(−6.5~−7.7)을 중요 레이어 보호로 회복
- VLM 선택 실험(dev10): Qwen3-VL-8B 90.64 > Qwen3.5-VL-4B 82.87 ≈ InternVL3-8B 82.60 > Gemma-4-E4B 71.42
- LoRA 실험: VQA-LoRA 92.82 > LoRA+ 90.15 > Dreamer-LoRA 87.42 / Delta-LoRA 85.09

## 설명 가능성

위험 상황마다 **적용된 제어·게이트 판정·주행 상황 설명**을 함께 기록합니다. 아래는 안개 속 STOP 표지판 시나리오의 대시보드입니다.

<p align="center"><img src="assets/ravid/dashboard_stop.png" alt="JARVIS dashboard" width="850"/></p>

## 저장소 구조 (팀 코드)

```
src/garage_ext/               # 팀 확장 레이어 (upstream 무수정 플러그인 구조)
├── agents/                   #   ExtSensorAgent, Qwen 메타액션 에이전트
├── modules/                  #   image_enhancer(classic_cv) · risk(TTC/action-aware) · safety(brake_guard, semantic_arbiter) · vlm
├── vlm_intervention/         #   Qwen 클라이언트·입력 구성·TTC·로거
└── visualization/            #   판단 근거 대시보드 렌더러
team_code/sensor_agent_meta_action*.py   # 최종 메타액션 VLA 에이전트 (전·후방, rule-hold 변형)
tools/                        # GAWQ/GPTQ/AWQ/W8A8 양자화, LoRA layer-wise merge, 오프라인 평가·집계
run_*.sh                      # dev10/220 평가·양자화 스위트 실행 스크립트
docs/                         # 실험 방법론·양자화 워크로그·아키텍처 문서
```

- 실행·재현 절차: [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md) (인수인계 문서)
- VLA 개입 설계: [QWEN_INTERVENTION.md](QWEN_INTERVENTION.md) · [QWEN_WORK_SUMMARY.md](QWEN_WORK_SUMMARY.md)
- 양자화 실험 기록: [docs/qwen_quantization_summary.md](docs/qwen_quantization_summary.md)

## 팀 자비스 (JARVIS)

| 이름 | 역할 |
|---|---|
| 오은수 (팀장) | 이미지 보정 모듈 설계·구현, 학술대회 논문 작성·발표, 프로젝트 방향 관리, 시스템 구조도 |
| 이태호 | VLM 후방 입력 추가 실험, 통합 파이프라인 실험, **Layer-wise Merge 방법론 제안** 및 양자화 비교 실험, 논문 분석 |
| 강설아 | 교통사고 통계·위험요인 조사, VLM 후보 모델 비교, LoRA 계열 성능 비교, 연구·논문 분석 |
| 이다영 | 대시보드·웹 인터페이스 구현, VLA 방법론 구현, 데이터셋 구조 분석, LoRA 파인튜닝·결과 시각화 |
| 박한슬 | VLM 후방 입력 추가 실험, 통합 파이프라인 실험, LoRA 파인튜닝, **GAWQ 양자화 방법론 제안**·실험, 앱 구현 |

## Upstream (TransFuser++ / carla_garage)

이 저장소는 [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage)의 포크입니다. TransFuser++ 학습·데이터셋 생성·리더보드 제출 등 원본 사용법은 업스트림 README를 참고하세요. 아래 팀 확장 레이어 섹션은 upstream 코드를 수정하지 않고 우리 모듈을 얹는 방법을 설명합니다.

---


## 팀 확장 레이어 (Team Extension Layer)

> **이 섹션은 팀 내부용입니다.** upstream(autonomousvision) 코드를 건드리지 않고 연구 모듈을 추가·교체하는 방법을 설명합니다.

### 왜 이렇게 구성했는가

`team_code/`는 upstream에서 지속적으로 업데이트되는 코드입니다. 이 파일을 직접 수정하면 매번 머지 충돌이 발생합니다. 그래서 **우리 코드는 `src/garage_ext/` 안에만** 존재하며, upstream은 **import + 서브클래싱**으로만 접근합니다.

```
carla_garage/
├── team_code/                    # upstream — 절대 직접 수정 금지
├── src/garage_ext/               # 우리 팀 코드 전부 여기
│   ├── agents/
│   │   └── ext_sensor_agent.py   # SensorAgent 서브클래스 (진입점)
│   ├── config/
│   │   └── ext_config.py         # ExtConfig + YAML 오버레이 로더
│   ├── modules/
│   │   ├── base.py               # Observation/Plan/Control/RiskReport + Protocol 정의
│   │   ├── image_enhancer/       # 카메라 이미지 전처리 모듈
│   │   ├── vlm/                  # Vision-Language 모듈
│   │   ├── risk/                 # 위험도 추정 모듈
│   │   └── safety/               # 안전 필터 모듈
│   ├── overrides/                # 불가피한 upstream 수정사항 격리 (최후 수단)
│   ├── pipeline.py               # vlm → risk → safety 연결 파이프라인
│   └── registry.py               # (kind, name) → class 문자열 레지스트리
├── configs/
│   ├── base.yaml                 # 기본값 (모든 모듈 비활성)
│   └── experiments/              # 실험별 YAML (git 추적)
│       ├── classic_enhance.yaml  # 이미지 보정 실험
│       └── example_vlm_risk.yaml # VLM + 위험도 실험 템플릿
└── docs/dev/                     # 팀 내부 개발 문서
```

---

### 한 스텝당 데이터 흐름

```
카메라 프레임 (rgb_front, rgb_left, …)
        │
        ▼  ← _apply_image_enhancement()
   보정된 프레임
        │
        ▼
 upstream SensorAgent.run_step()  →  steer/throttle/brake
        │
        ▼
 ExtPipeline.run(obs, plan, base_control):
   1. vlm.infer(obs)          ← 선택적 VLM 추론
   2. risk.estimate(obs, plan) ← 위험도 점수 계산
   3. safety.filter(control, risk, obs)  ← 제어값 최종 보정
        │
        ▼
   최종 control → CARLA 시뮬레이터
```

파이프라인은 upstream 인지·계획을 **대체하지 않고**, 그 위에서 **보완·감시**합니다.

---

### 평가 실행 방법

```bash
# 1) CARLA 서버 실행 (별도 터미널)
cd /mnt/2/carla
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=30000

# 2) 평가 스크립트 실행
cd /mnt/2/carla_garage
bash Bench2Drive/leaderboard/scripts/run_evaluation_tf++_local.sh
```

**이미지 보정 켜기/끄기** — 스크립트 안의 한 줄로 조절:
```bash
# 켜기
export GARAGE_EXT_CONFIG=/mnt/2/carla_garage/configs/experiments/classic_enhance.yaml

# 끄기 (주석 처리하거나 줄 삭제)
# export GARAGE_EXT_CONFIG=...
```

**시각화 저장** (`DEBUG_CHALLENGE=1`이면 `SAVE_PATH` 아래에 프레임/비교 이미지 저장):
```bash
export DEBUG_CHALLENGE=1
export SAVE_PATH=/mnt/2/carla_metric_result/carla_viz
```

**평가 강제 종료**:
```bash
bash kill_eval.sh
```

---

### 모듈 추가하는 법 (예: 새 이미지 보정기)

#### 1단계 — 구현 파일 생성

`src/garage_ext/modules/image_enhancer/my_enhancer.py`:

```python
import numpy as np
from ...registry import register

@register("image_enhancer", "my_enhancer")
class MyEnhancer:
    def __init__(self, strength: float = 1.0, **_):
        self.strength = strength

    def enhance(self, bgr_uint8: np.ndarray) -> np.ndarray:
        # bgr_uint8: (H, W, 3) uint8 BGR 이미지를 받아 같은 shape/dtype 반환
        return bgr_uint8  # 여기에 로직 작성
```

**인터페이스 규칙**: `enhance(bgr_uint8)` 하나만 구현하면 됩니다. 상속 불필요.

#### 2단계 — `__init__.py`에 import 추가

`src/garage_ext/modules/image_enhancer/__init__.py`:

```python
from . import classic   # noqa: F401
from . import my_enhancer  # noqa: F401  ← 추가
```

#### 3단계 — 실험 YAML 작성

`configs/experiments/my_enhance_run.yaml`:

```yaml
extends: ../base.yaml

image_enhancer: my_enhancer
image_enhancer_kwargs:
  strength: 1.5

meta:
  owner: <본인 핸들>
  tag: my-enhance-v1
  notes: "테스트 설명"
```

#### 4단계 — 스모크 테스트 추가

`tests/smoke/test_my_enhancer.py`:

```python
def test_my_enhancer_registered():
    import garage_ext.modules  # noqa: F401
    from garage_ext.registry import available, build
    import numpy as np

    assert "my_enhancer" in available("image_enhancer")
    enh = build("image_enhancer", "my_enhancer", strength=2.0)
    dummy = np.zeros((4, 4, 3), dtype=np.uint8)
    out = enh.enhance(dummy)
    assert out.shape == dummy.shape
```

#### 5단계 — 환경변수 설정 후 실행

```bash
export GARAGE_EXT_CONFIG=configs/experiments/my_enhance_run.yaml
bash Bench2Drive/leaderboard/scripts/run_evaluation_tf++_local.sh
```

**upstream 코드는 한 줄도 건드리지 않습니다.**

---

### 다른 종류의 모듈 추가 (VLM / 위험도 / 안전 필터)

같은 패턴입니다. 각 모듈이 구현해야 하는 메서드:

| 종류 | 레지스트리 kind | 구현 메서드 | 위치 |
|------|----------------|-------------|------|
| 이미지 보정 | `image_enhancer` | `enhance(bgr_uint8) → ndarray` | `modules/image_enhancer/` |
| VLM | `vlm` | `infer(obs: Observation) → dict` | `modules/vlm/` |
| 위험도 추정 | `risk` | `estimate(obs, plan) → RiskReport` | `modules/risk/` |
| 안전 필터 | `safety` | `filter(control, risk, obs) → Control` | `modules/safety/` |

모든 dataclass 정의는 [`src/garage_ext/modules/base.py`](src/garage_ext/modules/base.py)에 있습니다.

YAML에서 모듈 이름만 바꾸면 실험 전환 완료:
```yaml
extends: ../base.yaml
vlm: my_vlm_v2
risk: heuristic_ttc
safety: brake_if_risk_gt_05
```

---

### 비교 이미지 저장 위치

`SAVE_PATH`가 설정돼 있으면 이미지 보정 전/후 비교 패널이 자동 저장됩니다:

```
$SAVE_PATH/enhance_compare/
  scenario_000/
    00004.png   # (4프레임마다 1장)
    00008.png
  scenario_001/
    ...
```

각 PNG는 `ORIGINAL [FRONT] | ENHANCED [FRONT]` 형식의 좌우 비교 이미지입니다.

---

### `overrides/` 폴더 사용 규칙

서브클래싱으로 해결이 **불가능한 경우에만** 사용합니다. 파일 첫 줄에 반드시 명시:

```python
# ORIGIN: team_code/<원본 경로>
# REASON: <서브클래싱 불가 이유>
# SYNC-CHECK: <분기 시점 upstream 커밋 해시>
```

`overrides/`가 늘어나는 건 설계 재검토 신호입니다.

---

### 더 읽기

- [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md) — 설계 원칙 전체
- [docs/dev/MODULE_GUIDE.md](docs/dev/MODULE_GUIDE.md) — 모듈 추가 5단계 가이드
- [docs/dev/CONTRIBUTING.md](docs/dev/CONTRIBUTING.md) — PR 규칙 및 브랜치 전략
- [docs/dev/ONBOARDING.ko.md](docs/dev/ONBOARDING.ko.md) — 처음 합류한 팀원용 온보딩

---

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hidden-biases-of-end-to-end-driving-models/carla-leaderboard-2-0-on-carla)](https://paperswithcode.com/sota/carla-leaderboard-2-0-on-carla?p=hidden-biases-of-end-to-end-driving-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hidden-biases-of-end-to-end-driving-models/bench2drive-on-bench2drive)](https://paperswithcode.com/sota/bench2drive-on-bench2drive?p=hidden-biases-of-end-to-end-driving-models)


<p align="center" style="font-size:20px;">
This repository contains the first complete starter kit for the CARLA leaderboard 2.0 where all components are open-source including the dataset, expert driver, evaluation and training code.
We additionally provide pre-trained model weights for TransFuser++ which is the best open-source model at the time of publication. The paper <a href="https://arxiv.org/abs/2306.07957"> Hidden Biases of End-to-End Driving Models </a> describes the method and the <a href="https://arxiv.org/abs/2412.09602"> LB2 Technical Report</a> discusses the changes we made to adapt TransFuser++ to the CARLA leaderboard 2.0. <br/><br/>
The leaderboard 1.0 code can be found on the <a href="https://github.com/autonomousvision/carla_garage/tree/leaderboard_1"> leaderboard_1</a> branch.
</p>

## Citations
If you find CARLA garage useful, please consider giving us a star &#127775;.
Please cite the following papers for the respective components of the repo:

TransFuser++ Method:
```BibTeX
@InProceedings{Jaeger2023ICCV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Bernhard Jaeger and Kashyap Chitta and Andreas Geiger},
  booktitle={Proc. of the IEEE International Conf. on Computer Vision (ICCV)},
  year={2023}
}
```

TransFuser++ Leaderboard 2.0 changes
```BibTeX
@article{Zimmerlin2024ArXiv,
  title={Hidden Biases of End-to-End Driving Datasets},
  author={Julian Zimmerlin and Jens Beißwenger and Bernhard Jaeger and Andreas Geiger and Kashyap Chitta},
  journal={arXiv.org},
  volume={2412.09602},
  year={2024}
}

@mastersthesis{Zimmerlin2024thesis,
  title={Tackling CARLA Leaderboard 2.0 with End-to-End Imitation Learning},
  author={Julian Zimmerlin},
  school={University of Tübingen},
  howpublished={\textsc{url:}~\url{https://kashyap7x.github.io/assets/pdf/students/Zimmerlin2024.pdf}},
  year={2024}
}
```

PDM-Lite expert:
```BibTeX
@inproceedings{Sima2024ECCV,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2024}
}
```

Bench2Drive benchmark:

```BibTeX
@inproceedings{Jia2024NeurIPS,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

## Acknowledgements
The original code in this repository was written by Bernhard Jaeger, Julian Zimmerlin and Jens Beißwenger. Andreas Geiger and Kashyap Chitta have contributed as technical advisors.

Open source code like this is build on the shoulders of many other open source repositories.
Particularly, we would like to thank the following repositories for their contributions:
* [scenario_runner](https://github.com/carla-simulator/scenario_runner)
* [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
* [leaderboard](https://github.com/carla-simulator/leaderboard)
* [simple_bev](https://github.com/aharley/simple_bev)
* [transfuser](https://github.com/autonomousvision/transfuser)
* [InterFuser](https://github.com/opendilab/InterFuser)
* [DriveLM](https://github.com/OpenDriveLab/DriveLM)
* [roach](https://github.com/zhejz/carla-roach/)
* [plant](https://github.com/autonomousvision/plant)
* [king](https://github.com/autonomousvision/king)
* [WorldOnRails](https://github.com/dotchen/WorldOnRails)
* [TCP](https://github.com/OpenDriveLab/TCP)
* [LearningByCheating](https://github.com/dotchen/LearningByCheating)

We also thank the creators of the numerous pip libraries we use. Complex projects like this would not be feasible without your contribution.

