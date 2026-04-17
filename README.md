<p align="center">
  <img src="assets/carla_garage_white.png" alt="CARLA garage" width="500"/>
  <h3 align="center">
        <a href="https://arxiv.org/abs/2412.09602"> LB2 Technical Report</a> | <a href="https://arxiv.org/abs/2306.07957.pdf"> Paper</a> | <a href="https://youtu.be/ChrPW8RdqQU">Video</a> | <a href="https://youtu.be/x_42Fji1Z2M?t=1073">Talk</a> | <a href="https://www.cvlibs.net/shared/common_misconceptions.pdf"> Slides</a> | <a href="https://github.com/autonomousvision/carla_garage/tree/main/assets/Jaeger2023ICCV_Poster.pdf">Poster</a>
  </h3>
</p>

<p align="center" style="font-size:40px;">
<b> A starter kit for the <a href="https://leaderboard.carla.org/">CARLA leaderboard 2.0</a> </b>
</p>

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

## Contents

1. [Setup](#setup)
2. [Pre-Trained Models](#pre-trained-models)
3. [Local Evaluation and Debugging](#local-evaluation-and-debugging)
4. [Benchmarking](#benchmarking)
5. [Dataset](#dataset)
6. [Data Generation](#data-generation)
7. [Training](#training)
8. [Additional Documentation](#additional-documentation)
9. [Citations](#citations)

## Setup

Clone the repo, setup CARLA 0.9.15, and build the conda environment:
```Shell
git clone https://github.com/autonomousvision/carla_garage.git
cd carla_garage
git checkout leaderboard_2
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate garage_2
```

Before running the code, you will need to add the following paths to PYTHONPATH on your system:
```Shell
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```
You can add this in your shell scripts or directly integrate it into your favorite IDE. \
E.g. in PyCharm: Settings -> Project -> Python Interpreter -> Show all -> garage (need to add from existing conda environment first) -> Show Interpreter Paths -> add all the absolute paths above (without pythonpath).

## Pre-Trained Models
We provide a set of [pretrained models](https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip).
The models are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).

These are not the exact model weights we used in the CVPR CARLA 2024 challenge but re-trained models with similar performance.
We currently provide 1 set of models with 3 different training seeds trained on all towns (i.e., including Town13) for the Bench2Drive and CARLA leaderboard 2.0 test routes benchmarks.
We will add models for the validation benchmark at a later point.

Each folder has an `args.txt` containing the training hyperparameters, a `config.json` containing all hyperparameters which will automatically be loaded and `model_0030_0.pth` files containing the model weights. The last number in the model file indicates the seed/training repetition.

## Local Evaluation and Debugging

To evaluate a model, you need to start a CARLA server:
```Shell
cd /path/to/CARLA/root
./CarlaUE4.sh
```
Afterwards, run [leaderboard_evaluator_local.py](leaderboard/leaderboard/leaderboard_evaluator_local.py) as the main python file.

The leaderboard in the leaderboard folder is the original leaderboard (with some minor changes like extra seeding options and an upgrade to python 3.10). The leaderboard_autopilot is a modified version of the leaderboard that stores extra information which helps the privileged expert driver solve the scenarios. It is used for data collection.
The leaderboard in the Bench2Drive folder is a modified version of the leaderboard which was adapted by the Bench2Drive team for short evaluation routes.

Set the `--agent-config` option to a folder containing a `config.json` and one or more `model_0030.pth` files. If multiple models are present in the folder, their predictions will be combined to an ensemble. <br>
Set the `--agent` to [sensor_agent.py](team_code/sensor_agent.py). <br>
The `--routes` option should be set to a route file, for example [debug.xml](leaderboard/data/debug.xml). <br>
Set `--checkpoint ` to `/path/to/results/result.json`


Set `export SAVE_PATH=/path/to/results` to save additional logs or visualizations

The code has the optional feature to generate visualizations. To turn it on, set the environment variables `export DEBUG_CHALLENGE=1`, and the path where you want the visualization to be stored `export SAVE_PATH=/path/to/visualization`. <br>

## Benchmarking

### Bench2Drive
Bench2Drive is a CARLA benchmark proposed by the paper [Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://arxiv.org/abs/2406.03877). It consists of 220 very short (~150m) routes split across all towns with 1 safety critical scenario in each route.
Since it uses all towns for training, the methods have seen the test towns during training, so it can be considered a 'training' benchmark (reminiscent of level 4 driving).
The benchmark also comes with a training dataset generated by the [Think2Drive](https://arxiv.org/abs/2402.16720) expert, but we don't use it here since we observe that TransFuser++ trained with data from the [PDM-Lite](https://arxiv.org/abs/2312.14150) expert achieves much better results than all the other methods trained with Think2Drive (see picture below).
The benchmark and additional instructions can be found in the [Bench2Drive](Bench2Drive) folder.
It is run by executing the bash script [run_evaluation_tf++.sh](Bench2Drive/leaderboard/scripts/run_evaluation_tf++.sh). You need to adjust the script with your paths and number of GPUs (by default, the script assumes 8 GPUs are available on the node.) To give a rough estimate, it takes around ~4 hours to evaluate TF++ on Bench2Drive with 8x2080ti. For more details on how to aggregate the results, see the [README](Bench2Drive/README.md).

<img src="assets/Bench2Drive.png" alt="Bench2Drive" width="1000"/>

The Bench2Drive folder is a copy of version 0.0.3 of the [Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive). Please cite the [Bench2Drive paper](https://arxiv.org/abs/2406.03877) when using the benchmark.

### CARLA leaderboard 2.0 validation routes
The [CARLA leaderboard 2.0 validation routes](leaderboard/data/routes_validation.xml) is a set of 20 long (~12 km) routes in Town 13. While driving along the routes, the agent has to solve around 90 safety critical scenarios per route consisting of [21 different types](https://leaderboard.carla.org/scenarios/) (38 counting variations).
As its name suggests, this is a 'validation' benchmark, so data from Town 13 may not be used during training (reminiscent of level 5 driving).
To train a model for this benchmark, use the training command line option `--setting 13_withheld`.
We recommend running 3 seed repetitions of the 20 routes with different seeds to reduce the impact of evaluation variance (which is quite high).

Due to the length of the routes as well as the large number of scenarios per route, the scores on this benchmark are much lower than on Bench2Drive and other benchmarks.
The CARLA leaderboard 2.0 validation routes are probably the most challenging public autonomous driving benchmark at the time of writing (Dec. 2024), so they are ideal for showcasing improvements over the state-of-the-art method(s).

Evaluation is best done by evaluating the 20*3 routes in parallel. We use a SLURM cluster with 2080ti GPUs for this. We provide our [evaluation script](evaluate_routes_slurm_tfpp.py) for this. It parallelizes the 60 routes across 60 jobs and monitors if any jobs crashed, restarting them as needed. You need to set your paths with the console arguments. The script is started via `sbatch run_evaluation_slurm_tfpp.sh` which starts the evaluation for every training seed/repetition you have (change as needed). Make sure your conda environment is active. For other types of clusters, you need to adapt the script accordingly. To give you a rough idea, with 14 GPUs, evaluation is typically done within a day. The [max_num_jobs.txt](max_num_jobs.txt) file specifies the maximum number of jobs the script will spawn and can be edited while running the evaluation. Keep it low initially and increase the number once your setup works.

The benchmark revealed a flaw in the common Driving Score metric, which for long routes and lower scores (that SOTA is currently at) can assign a lower driving score to a better method. To fix this problem, we propose the Normalized Driving Score metric, which does not have this issue but otherwise is similar in difficulty. For an in depth discussion of the problem and solution, please read [Zimmerlin 2024, Chapter 6](https://kashyap7x.github.io/assets/pdf/students/Zimmerlin2024.pdf).

The normalized driving score and other metrics for a detailed analysis can be computed by running the following python script, which will aggregate all given result files (in case the evaluation was distributed across GPUs) and compute the additional metrics.
```Shell
python ${WORK_DIR}/tools/result_parser.py --xml ${WORK_DIR}/leaderboard/data/routes_validation.xml --results /path/to/folder/containing_json_files
```

### CARLA leaderboard 2.0 test routes
The CARLA leaderboard test routes are similar to the validation routes, with the difference being that the test town 14 was not publicly released, and the evaluation is done by a third party ensuring fair results.
It works by creating a docker container with your model and code and uploading it to a third party evaluation server.
Instructions for submitting your model to the test server can be found on the [CARLA leaderboard website](https://leaderboard.carla.org/submit/) as well as [eval.ai](https://eval.ai/web/challenges/challenge-page/2098/overview), where you can view your results.
The CARLA leaderboard test routes are frequently used for competitions at workshops of top tier conferences.
TransFuser++ achieved second place at the [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/News/Workshop-Winners) [CARLA Autonomous Driving Challenge](https://opendrivelab.com/cvpr2024/workshop/) (Team Tuebingen_AI).

![CVPR_Challenge_2024](assets/CVPR_Challenge_2024.png)

At the time of writing (Dec. 2024) the CARLA leaderboard test server is unfortunately temporarily closed and does not accept submissions. 
By the terms of the competition, it is not allowed to evaluate privileged methods like PDM-Lite on the CARLA leaderboard 2.0 test routes.

To submit to the CARLA leaderboard, you need docker installed on your system (as well as the nvidia-container-toolkit to test it). To generate and test the dockerfile we provide the scripts [make_docker.sh](tools/make_docker.sh) and [run_docker.sh](tools/run_docker.sh). You need to change the paths to fit your system and edit variables in [Dockerfile.master](tools/Dockerfile.master).

Note that the CARLA leaderboard 2.0 test routes should not be confused with the devtest routes, which are a set of 2 routes for development testing aka debugging. The devtest routes are in the training town, miss many scenarios and lack diversity, so they are not suitable for benchmarking.

### Longest6 v2
[Longest6](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf) is a benchmark consisting of 36 medium length routes (~1-2 km) from leaderboard 1.0 in towns 1-6. We have adapted the benchmark to the new CARLA version 0.9.15 and leaderboard/scenario runner code. The benchmark features the 7 scenario types from leaderboard 1.0 (but implemented with the leaderboard 2.0 logic). The scenario descriptions were created by converting the leaderboard 1.0 scenarios with the [CARLA route bridge](tools/route_bridge.py) converter. It can serve as a benchmark with intermediate difficulty.
Note that the results of models on Longest6 v2 are not directly comparable to the leaderboard 1.0 longest6 numbers.
The benchmark can be found [here](leaderboard/data/longest6.xml) and the individual route files [here](leaderboard/data/longest6_split). Unlike the leaderboard 1.0 version, there are no modifications to the CARLA leaderboard code. Longest6 is a training benchmark, so training on Town 01-06 is allowed.

### Common mistakes in benchmarking autonomous driving
Benchmarking entire autonomous driving stacks is hard, and it is easy to get important subtle details wrong.
Unfortunately, the literature is riddled with these methodological mistakes. As an attempt to improve this situation we have written a guide on [common mistakes in benchmarking](docs/common_mistakes_in_benchmarking_ad.md), for authors to avoid them and reviewers to catch them.

## Dataset
We released the dataset we used to train the released model.
Note that this dataset was generated with a slightly older version of PDM-Lite than we have in the repository.
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
You can download it via.
```Shell
cd /path/to/carla_garage/tools
bash download_data.sh
```

The script will download the data to `/path/to/carla_garage/data`. This is also the path you need to set `--root_dir` to for training. The script will download and unzip the data with 39 parallel processes. The download is roughly 364 GB large. To download from China, you might need a high speed VPN.

 ## Data Generation
This repository uses the open source expert PDM-Lite from the paper [DriveLM](https://arxiv.org/abs/2312.14150) to generate the dataset above.
To re-generate the data we provide a script for a SLURM cluster which parallelizes data collection across many GPUs (2080ti in our case). You need to change the path etc. in the script. The script is started via `sbatch 0_run_collect_dataset_slurm.sh`.
Again, increase the number in [max_num_jobs.txt](max_num_jobs.txt) once your setup works.

Dataset generation is similar to evaluation. You can debug data collection by changing the `--agent` option of [leaderboard_evaluator_local.py](leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py) to [data_agent.py](team_code/data_agent.py) and the `--track` option to `MAP`. In addition, you need to set the following environment flags:
```Shell
export DATAGEN=1
export SAVE_PATH=/path/to/dataset/Routes_{route}_Repetition{repetition}
export TOWN=Town12
export REPETITION=0
export SCENARIO_RUNNER_ROOT=/path/to/scenario_runner_autopilot
```
PDM-Lite uses a modified version of the CARLA leaderboard that exposes additional information about the scenarios and makes data collection easier. They can be found in the [leaderboard_autopilot](leaderboard_autopilot) and [scenario_runner_autopilot](scenario_runner_autopilot) folders.
The routes for data collection are stored in [data](data).
The dataset provided in this repository is not perfect. At some point while improving the model you will likely need to collect an improved version.

 ## Training

Agents are trained via the file [train.py](team_code/train.py). Examples of how to use it are provided for [shell](team_code/shell_train.sh) and [SLURM](team_code/slurm_train.sh). You need to activate the garage conda environment before running it. It first sets the relevant environment variables and then launches the training with torchrun. Torchrun is a pytorch tool that handles multi-gpu training. If you want to debug on a single gpu simply set --nproc_per_node=1. The training script has many options to configure your training, you can list them with python train.py --help or look through the code. The most important ones are:

```Shell
--id your_model_000 # Name of your experiment
--batch_size 16 # Batch size per GPU
--setting all # Which towns to withhold during training. Use 'all' for leaderboard test routes and bench2drive, 13_withheld for the leaderboard validation routes.
--root_dir /path/to/dataset # Path to the root_dir of your dataset
--logdir /path/to/models # Root dir where the training files will be stored
--cpu_cores 20 # Total number of cpu cores on your machine
```

Training is normally done in 2 stages. For the perception pre-training stage, first turn off the checkpoint prediction and classification by setting:
```Shell
--use_controller_input_prediction 0
```
After training the model, run the script a second time for stage 2 with:
```Shell
--use_controller_input_prediction 1
--continue_epoch 0
--load_file /path/to/model/from/stage1/model_030.pth
```
The load_file option is usually used to resume a crashed training run, but with --continue_epoch 0 the training will start from scratch with the pre-trained weights used for initialization.

Training takes roughly 3 days per stage on 4 A100 (40GB) GPUs.
If you are compute constrained, we recommend using only 1 stage of training, and using a smaller backbone (with a larger batch size). This will reduce the training cost, but may result in lower performance.
```Shell
--image_architecture resnet34
--lidar_architecture resnet34
```

### Training in PyCharm
You can also run and debug torchrun in PyCharm. To do that, you need to set your run/debug configuration as follows:\
In the run configuration change script to module and type in: `torch.distributed.run` \
Set the parameters to:
```Shell
--nnodes=1
--nproc_per_node=1
--max_restarts=0
/path/to/train.py
--id test_tf_000_0
```
and fill in the parameters for train.py afterward.
Environment variable can be set in `Environment Variables:`.

## Additional Documentation
- The documentation on **Coordinate systems** systems can be found [here](docs/coordinate_systems.md).

- The TransFuser model family has grown quite a lot with different variants, which can be confusing for new community members. The **[history](docs/history.md)** file explains the different versions and which paper you should cite to refer to them.

- Building a full autonomous driving stack involves quite some [**engineering**](docs/engineering.md). The documentation explains some of the techniques and design philosophies we used in this project.

- The leaderboard_1 branch can run any experiment presented in the ICCV paper. It also supports some additional features that we did not end up using. They are documented [here](docs/additional_features.md).

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

