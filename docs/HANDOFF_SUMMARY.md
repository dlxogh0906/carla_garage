# 프로젝트 작업 인수인계 문서

> 작성일: 2026-04-18 (최종 업데이트)
> 작성자: kwy00 (aldatitus02@hotmail.com)
> 대상 저장소: `/mnt/2/carla_garage`

---

## 1. 문서 목적

이 문서는 **carla_garage 프로젝트에 팀 확장 레이어(garage_ext)를 추가한 작업**을 인수인계하기 위해 작성됐다.

- upstream(autonomousvision/carla_garage)의 TransFuser++ 코드를 건드리지 않고,
  VLM·위험도 추정·안전 필터·이미지 보정 모듈을 붙일 수 있는 플러그인 구조를 설계·구현했다.
- 이미지 보정(classic_cv)을 첫 번째 실모듈로 구현하고, Bench2Drive 220 전체 평가(220개 루트)를 완료했다.
- 다음 담당자가 새 모듈을 추가하거나 실험을 이어받을 수 있도록 실행 방법·설계·결과를 모두 기록한다.

---

## 2. 전체 요약

### 핵심 작업 목록

| # | 작업 | 상태 |
|---|------|------|
| 1 | `src/garage_ext/` 패키지 스캐폴드 구축 (registry, pipeline, config, modules 골격) | ✅ 완료 |
| 2 | 모듈 Protocol 기반 인터페이스 정의 (`modules/base.py`) | ✅ 완료 |
| 3 | YAML 오버레이 기반 실험 설정 레이어 구현 (`ExtConfig`) | ✅ 완료 |
| 4 | VLM·risk·safety noop 스텁 + 파이프라인 wiring 완성 | ✅ 완료 |
| 5 | `ExtSensorAgent`: upstream `SensorAgent` 서브클래스, 파이프라인 주입 | ✅ 완료 |
| 6 | 이미지 보정 모듈 구현 (`classic_cv`: 저조도·과노출·안개·흐림 자동 판별·보정) | ✅ 완료 |
| 7 | 비교 이미지 자동 저장 (원본\|보정 패널, 루트별 폴더 분리) | ✅ 완료 |
| 8 | 스모크 테스트 4개 + GitHub Actions CI 구성 | ✅ 완료 |
| 9 | CODEOWNERS, PR 템플릿, 개발 문서 (ARCHITECTURE, MODULE_GUIDE, CONTRIBUTING, ONBOARDING) | ✅ 완료 |
| 10 | `auto_eval_restart.sh`: CARLA 크래시 자동 감지·재시작 스크립트 | ✅ 완료 |
| 11 | Bench2Drive 220 이미지 보정 실험 전체 평가 (task 0~3, 총 220개 루트) | ✅ 완료 |

### 현재 상태

- **코드**: `src/garage_ext/` 전체 구현 완료. 팀원이 새 모듈을 추가하는 구조 안정적으로 동작 확인.
- **실험**: `classic_enhance` 실험 Bench2Drive 220 전 루트 완료. task 0~3 각 55개씩 총 220개 루트.
- **다음 할 일**: 베이스라인(보정 없음) 220개 평가 → 정량 비교 / VLM·risk·safety 실제 모듈 구현.

### 가장 중요한 결과 — Bench2Drive 220 이미지 보정 실험

| Task | 루트 수 | DS | RC | Penalty |
|------|---------|-----|-----|---------|
| Task 0 | 55 | 89.291 | 99.042% | 0.8987 |
| Task 1 | 55 | 87.884 | 100.000% | 0.8788 |
| Task 2 | 55 | 85.833 | 99.099% | 0.8616 |
| Task 3 | 55 | 82.316 | 94.617% | 0.8729 |
| **평균** | **220** | **86.331** | **98.190%** | **0.8780** |

> DS = Driving Score, RC = Route Completion  
> 베이스라인(보정 없음) 비교 결과는 아직 없음 → **추후 동일 조건으로 베이스라인 평가 필요**

---

## 3. 프로젝트 구조

```
carla_garage/
│
├── team_code/                        # ★ upstream — 절대 수정 금지
├── leaderboard/                      # ★ upstream
├── Bench2Drive/                      # ★ upstream (Bench2Drive 벤치마크)
│   └── leaderboard/scripts/
│       ├── run_evaluation_tf++_local.sh   ← 로컬 평가 스크립트 (수정 가능)
│       └── run_evaluation.sh             ← upstream 평가 래퍼 (수정 금지)
│
├── src/garage_ext/                   # ★ 팀 코드 전체 (여기서만 작업)
│   ├── agents/
│   │   └── ext_sensor_agent.py       ← leaderboard 진입점 (SensorAgent 서브클래스)
│   ├── config/
│   │   └── ext_config.py             ← ExtConfig + YAML 오버레이 로더
│   ├── modules/
│   │   ├── base.py                   ← Observation/Plan/Control/RiskReport + Protocol
│   │   ├── image_enhancer/
│   │   │   ├── base.py               ← ImageEnhancer Protocol
│   │   │   ├── classic.py            ← classic_cv 구현 (저조도/과노출/안개/흐림)
│   │   │   └── __init__.py
│   │   ├── vlm/
│   │   │   ├── dummy.py              ← noop VLM 템플릿
│   │   │   └── __init__.py
│   │   ├── risk/
│   │   │   ├── dummy.py              ← noop risk 템플릿
│   │   │   └── __init__.py
│   │   └── safety/
│   │       ├── dummy.py              ← noop safety 템플릿
│   │       └── __init__.py
│   ├── overrides/                    ← upstream 강제 수정 격리 (현재 미사용)
│   ├── pipeline.py                   ← vlm → risk → safety 파이프라인
│   └── registry.py                   ← (kind, name) → class 레지스트리
│
├── configs/
│   ├── base.yaml                     ← 기본값 (모든 모듈 noop/비활성)
│   └── experiments/
│       ├── classic_enhance.yaml      ← 이미지 보정 실험 ← 현재 사용된 설정
│       └── example_vlm_risk.yaml     ← VLM+risk 예제 템플릿
│
├── tests/
│   └── smoke/
│       └── test_imports.py           ← 스모크 테스트 4개 (CARLA 없이 CI 실행)
│
├── docs/dev/
│   ├── ARCHITECTURE.md               ← 설계 원칙
│   ├── MODULE_GUIDE.md               ← 모듈 추가 5단계 가이드
│   ├── CONTRIBUTING.md               ← 브랜치/PR 규칙
│   └── ONBOARDING.ko.md              ← 신규 팀원 30분 온보딩
│
├── .github/
│   ├── CODEOWNERS                    ← @dlxogh0906 전체 담당
│   ├── pull_request_template.md
│   └── workflows/ci.yml              ← lint(yapf) + smoke test 자동 실행
│
├── pyproject.toml                    ← garage-ext 패키지 설정 (Python ≥ 3.8)
├── auto_eval_restart.sh              ← CARLA 크래시 자동 재시작 스크립트
├── kill_eval.sh                      ← 평가 프로세스 강제 종료
└── HANDOFF_SUMMARY.md                ← 이 문서
```

---

## 4. 내가 지금까지 한 작업

### 작업 1: garage_ext 패키지 스캐폴드 구축

- **목적**: upstream 코드를 건드리지 않고 팀 모듈을 붙이는 플러그인 구조 설계
- **한 일**: `src/garage_ext/` 디렉토리 + `pyproject.toml` 생성
- **생성 파일**: `pyproject.toml`, `src/garage_ext/__init__.py`
- **결과**: `pip install -e ".[dev]"` 한 번으로 팀 코드 설치 가능

### 작업 2: registry + Protocol 인터페이스 구현

- **목적**: YAML 한 줄로 모듈을 교체할 수 있는 플러그인 구조
- **한 일**:
  - `registry.py`: `@register(kind, name)` 데코레이터 + `build(kind, name)` 팩토리
  - `modules/base.py`: `Observation`, `Plan`, `Control`, `RiskReport` 데이터클래스 + `VLMModule`, `RiskEstimator`, `SafetyFilter` Protocol 정의
- **결과**: 팀원이 `@register` 하나로 모듈 등록, YAML에서 이름으로 교체 가능

### 작업 3: YAML 오버레이 설정 레이어 구현

- **목적**: 실험마다 base.yaml을 상속해서 변경점만 기록, 재현성 확보
- **한 일**: `ExtConfig` 데이터클래스 + `load_experiment_config()` 로더 구현 (`extends:` 키 지원)
- **생성 파일**: `src/garage_ext/config/ext_config.py`, `configs/base.yaml`, `configs/experiments/example_vlm_risk.yaml`

### 작업 4: VLM / risk / safety noop 스텁 + 파이프라인 구현

- **목적**: 실제 모듈 없이도 파이프라인 전체가 돌아가는 구조 확인
- **한 일**: 각 종류별 noop 클래스 구현 + `pipeline.py`로 vlm → risk → safety 연결
- **생성 파일**: `modules/vlm/dummy.py`, `modules/risk/dummy.py`, `modules/safety/dummy.py`, `pipeline.py`

### 작업 5: ExtSensorAgent 구현

- **목적**: leaderboard 평가에서 우리 파이프라인이 실행되도록 진입점 연결
- **한 일**: `SensorAgent` 서브클래스 작성. `run_step()` 안에서 이미지 보정 → upstream 실행 → 파이프라인 순서
- **생성 파일**: `src/garage_ext/agents/ext_sensor_agent.py`
- **비고**: `GARAGE_EXT_CONFIG` 환경변수로 설정 경로 전달. 미설정 시 upstream과 동일하게 동작

### 작업 6: 이미지 보정 모듈 (classic_cv) 구현

- **목적**: 카메라 이미지 품질 저하(저조도·과노출·안개·흐림) 자동 감지·보정으로 모델 입력 품질 향상
- **한 일**:
  - `_analyze()`: 밝기·채도·선명도·dark channel 기반 4가지 모드 점수화 및 판별
  - 저조도 → 감마(0.65) + CLAHE, 과노출 → 하이라이트 압축 + 대비 조정, 안개 → 채도+CLAHE, 흐림 → 언샤프 마스크
  - upstream 모델이 이미지를 보기 **전에** 보정 적용
- **생성 파일**: `modules/image_enhancer/classic.py`, `configs/experiments/classic_enhance.yaml`
- **결과**: Bench2Drive 220 전체 평가에서 정상 동작 확인

### 작업 7: 비교 이미지 자동 저장

- **목적**: 보정 전후를 시각적으로 비교해서 모듈 효과 검증
- **한 일**: `_save_compare()` 구현. 원본|보정 좌우 패널 PNG 저장. 루트마다 폴더 분리
- **저장 경로**: `$SAVE_PATH/task_N/enhance_compare/scenario_XXX/FFFFF.png` (4프레임마다 1장)

### 작업 8: 스모크 테스트 + CI 구성

- **목적**: CARLA 없이 PR마다 빠른 검증
- **한 일**: 4개 스모크 테스트 + `ci.yml` (yapf lint + pytest)
- **생성 파일**: `tests/smoke/test_imports.py`, `.github/workflows/ci.yml`

### 작업 9: 팀 개발 문서 작성 + README 업데이트

- **목적**: 팀원이 문서만 보고 새 모듈을 추가할 수 있도록
- **생성 파일**: `docs/dev/` 아래 4개 md 파일, `README.md` 앞부분에 "팀 확장 레이어" 섹션 추가

### 작업 10: auto_eval_restart.sh 구현

- **목적**: CARLA 크래시(Signal=11) 발생 시 수동 재시작 없이 자동 복구
- **원리**:
  1. 평가 스크립트 실행
  2. exit code ≠ 0이면 크래시로 판단
  3. 잔존 CARLA 프로세스 `pkill` 후 30초 대기
  4. 같은 명령어로 재실행 (최대 20회)
  5. `--resume=True`가 leaderboard에 설정되어 있어 JSON 체크포인트에서 자동으로 이어받음
- **비고**: 로그는 `tee -a` append 모드 → 기존 로그 보존. task별 `TASK_SAVE_PATH` 분리로 시각화 덮어쓰기 방지

### 작업 11: Bench2Drive 220 전체 평가 완료

- **목적**: 이미지 보정 모듈의 실제 성능 검증
- **한 일**: task 0~3 순차 실행, CARLA 크래시 시 auto_eval_restart.sh로 자동 복구
- **결과**: 4개 task 모두 55/55 루트 완료, 전체 평균 DS 86.33

---

## 5. 데이터셋 정리

### 결과 파일 (JSON)

| 파일명 | 경로 | 루트 수 | 상태 | 비고 |
|--------|------|---------|------|------|
| `eval_bench2drive220_0.json` | `carla_metric_result/tfpp_b2d_traj/` | 10/55 | 미완료 | 베이스라인 시도, 중단됨 |
| `eval_bench2drive220_image_enhancer_0417_0.json` | 동일 | 55/55 | ✅ 완료 | 이미지 보정 task 0 |
| `eval_bench2drive220_image_enhancer_0417_1.json` | 동일 | 55/55 | ✅ 완료 | 이미지 보정 task 1 |
| `eval_bench2drive220_image_enhancer_0417_2.json` | 동일 | 55/55 | ✅ 완료 | 이미지 보정 task 2 |
| `eval_bench2drive220_image_enhancer_0417_3.json` | 동일 | 55/55 | ✅ 완료 | 이미지 보정 task 3 |

### JSON 파일 구조

```json
{
  "_checkpoint": {
    "global_record": {
      "scores_mean": {
        "score_composed": 89.291,   // Driving Score (핵심 지표)
        "score_route":   99.042,    // Route Completion (%)
        "score_penalty":  0.899     // 위반 패널티 계수 (1.0에 가까울수록 좋음)
      },
      "infractions": {
        "collisions_vehicle":    2.147,   // 차량 충돌 횟수/km
        "min_speed_infractions": 188.969, // 최저속도 위반 횟수/km
        ...
      },
      "meta": { "total_length": ..., "duration_game": ..., "exceptions": [...] }
    },
    "progress": [55, 55],   // [완료 루트 수, 전체 루트 수]
    "records": [...]        // 루트별 개별 결과
  }
}
```

### 로그 파일

| 파일명 | 경로 | 내용 |
|--------|------|------|
| `eval_bench2drive220_image_enhancer_0417_0.log` | `carla_metric_result/` | task 0 전체 실행 로그 (append 누적) |
| `eval_bench2drive220_image_enhancer_0417_1.log` | `carla_metric_result/` | task 1 로그 |
| `eval_bench2drive220_image_enhancer_0417_2.log` | `carla_metric_result/` | task 2 로그 |
| `eval_bench2drive220_image_enhancer_0417_3.log` | `carla_metric_result/` | task 3 로그 |

### 루트 분할 파일 (이미 생성됨, 재생성 불필요)

| 파일명 | 루트 수 | 비고 |
|--------|---------|------|
| `bench2drive220_0_tfpp_traj.xml` | 55 | task 0 |
| `bench2drive220_1_tfpp_traj.xml` | 55 | task 1 |
| `bench2drive220_2_tfpp_traj.xml` | 55 | task 2 |
| `bench2drive220_3_tfpp_traj.xml` | 55 | task 3 |

> 분할 완료 여부: `Bench2Drive/leaderboard/data/bench2drive220_tfpp_traj_split_done.flag` 파일 존재로 확인

---

## 6. 실험 / 분석 / 결과

### 실험 1: 이미지 보정 (classic_cv) — Bench2Drive 220 전체

**실험 조건**

| 항목 | 값 |
|------|----|
| 모델 | TransFuser++ (all_towns, 3 seed ensemble) |
| 체크포인트 | `/mnt/2/pretrained_models/all_towns/model_0030_{0,1,2}.pth` |
| 벤치마크 | Bench2Drive 220 (총 220개 루트, 4개 task) |
| 설정 파일 | `configs/experiments/classic_enhance.yaml` |
| 이미지 보정 | `ClassicCVEnhancer` (저조도/과노출/안개/흐림 자동 판별) |
| 시각화 | `DEBUG_CHALLENGE=1`, `SAVE_PATH=/mnt/2/carla_metric_result/carla_viz/task_N` |

**전체 결과**

| Task | 루트 | DS | RC | Penalty | 차량충돌/km | 최저속도위반/km |
|------|------|-----|-----|---------|------------|----------------|
| 0 | 55 | 89.291 | 99.042% | 0.8987 | 2.147 | 188.969 |
| 1 | 55 | 87.884 | 100.000% | 0.8788 | 2.630 | 196.469 |
| 2 | 55 | 85.833 | 99.099% | 0.8616 | 2.260 | 158.789 |
| 3 | 55 | 82.316 | 94.617% | 0.8729 | 2.828 | 186.263 |
| **평균** | **220** | **86.331** | **98.190%** | **0.8780** | **2.466** | **182.623** |

**관찰 사항**

- `MinSpeedTest FAILURE` 다수 발생 → TF++ 베이스 모델 특성이며 이미지 보정 코드와 무관
- task 3에서 RC가 94.617%로 낮음 → 해당 루트에 어려운 시나리오가 집중됐을 가능성 (확인 필요)
- 베이스라인(보정 없음) 결과가 없어 이미지 보정의 효과를 아직 정량적으로 비교 불가

**아직 검증 안 된 부분**

- 베이스라인 vs 이미지 보정 정량 비교
- task별 DS 편차 원인 (task 0: 89.3 vs task 3: 82.3)
- 어떤 시나리오 유형에서 충돌이 집중 발생하는지

---

## 7. 핵심 코드 및 로직 설명

### 실행 흐름 (한 스텝)

```
카메라 프레임 (rgb_front, rgb_left, rgb_right, ...)
        │
        ▼  ExtSensorAgent._apply_image_enhancement()
   ClassicCVEnhancer.enhance(bgr)
     ├── _analyze() → mode 판별 (저조도/과노출/안개/흐림/정상)
     └── _process() → 모드별 보정 파이프라인 적용
        │
        ▼  upstream SensorAgent.run_step() [team_code/sensor_agent.py]
   TransFuser++ 인지·계획 → steer/throttle/brake
        │
        ▼  ExtPipeline.run(obs, plan, base_control)
   1. vlm.infer(obs)           → 현재 noop (빈 dict 반환)
   2. risk.estimate(obs, plan)  → 현재 noop (score=0)
   3. safety.filter(control, risk, obs) → 현재 noop (control 그대로 통과)
        │
        ▼
   최종 control → CARLA 시뮬레이터
```

### 파일별 핵심 역할

| 파일 | 역할 | 핵심 인터페이스 |
|------|------|----------------|
| `registry.py` | `(kind, name) → class` 매핑. YAML 기반 모듈 교체의 핵심 | `@register`, `build()` |
| `pipeline.py` | vlm → risk → safety 순서로 모듈 실행 | `ExtPipeline.run()` |
| `ext_sensor_agent.py` | leaderboard 진입점. upstream 위에 파이프라인 주입 | `get_entry_point()`, `run_step()` |
| `ext_config.py` | YAML → ExtConfig 변환. `extends:` 상속 지원 | `load_experiment_config()` |
| `modules/base.py` | 모든 모듈이 따를 데이터클래스 + Protocol 정의 | `Observation`, `Control` 등 |
| `modules/image_enhancer/classic.py` | 적응형 이미지 보정 구현 | `ClassicCVEnhancer.enhance()` |

### 모듈 추가 시 건드려야 하는 파일 (예: 새 risk 모듈)

```
1. src/garage_ext/modules/risk/my_risk.py       ← 새 파일 생성
2. src/garage_ext/modules/risk/__init__.py       ← import 한 줄 추가
3. tests/smoke/test_my_risk.py                   ← 스모크 테스트 추가
4. configs/experiments/my_experiment.yaml        ← 실험 설정 작성
   (그 외 파일은 수정 불필요)
```

상세 가이드: `docs/dev/MODULE_GUIDE.md`

---

## 8. 의사결정 및 변경 이력

### 왜 `team_code/`를 수정하지 않았는가

upstream은 지속 업데이트되며 직접 수정하면 매번 머지 충돌이 발생한다. `src/garage_ext/`를 완전히 분리함으로써 `git merge upstream/leaderboard_2`만으로 upstream 업데이트를 받을 수 있다.

### 왜 ABC 대신 Protocol을 사용했는가

Protocol(duck typing)은 상속 없이 메서드 시그니처만 맞추면 모듈로 인정된다. 불필요한 상속 보일러플레이트가 없어 팀원이 자유롭게 구현할 수 있다.

### 왜 환경변수(`GARAGE_EXT_CONFIG`)로 설정을 전달했는가

leaderboard가 `--agent` 경로만 받고 추가 CLI 인수를 허용하지 않는다. 환경변수를 사용하면 실험 스크립트에서 한 줄만 바꾸면 되고, 미설정 시 upstream과 완전히 동일하게 동작한다.

### 왜 이미지 보정을 pipeline이 아니라 ExtSensorAgent에서 직접 처리했는가

파이프라인은 upstream이 control을 계산한 **이후**에 실행된다. 이미지 보정은 upstream 모델이 입력을 보기 **전에** 적용해야 효과가 있으므로, `run_step()` 최상단에서 처리했다.

### 왜 auto_eval_restart.sh에서 task별 SAVE_PATH를 분리했는가

`enhance_compare/scenario_000/` 같은 폴더 번호가 task마다 0부터 다시 시작하므로, 같은 경로를 쓰면 task 1이 task 0의 시각화를 덮어쓴다. `task_N` 서브폴더로 분리해서 방지했다.

### git 주요 커밋 이력

| 커밋 해시 | 내용 |
|-----------|------|
| `a84cb24` | README 팀 확장 레이어 섹션 추가 |
| `2e5a0a4` | CI 의존성 및 포맷팅 수정 |
| `3d24f8f` | 이미지 보정 모듈 + 루트별 시각화 저장 |
| `35ba8d9` | 한국어 온보딩 문서 추가 |
| `ca82720` | garage_ext 확장 스캐폴드 최초 구축 |

---

## 9. 실행 방법

### 환경 설정

```bash
conda activate garage_2

# numpy 버전 확인 — 2.x이면 transforms3d 충돌 발생
python -c "import numpy; print(numpy.__version__)"
# 2.x인 경우 다운그레이드
pip install "numpy<2.0"

# garage_ext 패키지 설치 (최초 1회)
cd /mnt/2/carla_garage
pip install -e ".[dev]"

# 스모크 테스트 통과 확인
pytest tests/smoke -q
```

### 새 실험 처음부터 돌리기 (220개 전체)

`auto_eval_restart.sh` 수정:
```bash
GPU_RANK_LIST=(0 0 0 0)
TASK_LIST=(0 1 2 3)
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220_<실험명>_<날짜>
```
실행:
```bash
bash /mnt/2/carla_garage/auto_eval_restart.sh
```

### 이미지 보정 끄기 (베이스라인)

`auto_eval_restart.sh`에서 아래 줄 주석 처리:
```bash
# export GARAGE_EXT_CONFIG=...
```

### CARLA 크래시 후 이어받기

그냥 다시 실행하면 됨. `--resume=True`가 자동으로 JSON 체크포인트에서 이어받음:
```bash
bash /mnt/2/carla_garage/auto_eval_restart.sh
```

### 220개 전체 최종 점수 집계

```bash
python Bench2Drive/tools/result_parser.py \
  --xml Bench2Drive/leaderboard/data/bench2drive220.xml \
  --results /mnt/2/carla_metric_result/tfpp_b2d_traj/
```

### 평가 강제 종료

```bash
bash /mnt/2/carla_garage/scripts/eval/kill_eval.sh
```

---

## 10. 현재 상태 / 남은 이슈

### 완료된 것

- [x] `src/garage_ext/` 패키지 전체 구현
- [x] 이미지 보정 모듈 (classic_cv) 구현 및 검증
- [x] Bench2Drive 220 전체 (220개 루트) 이미지 보정 실험 완료
- [x] CI, 테스트, 개발 문서 완비
- [x] CARLA 크래시 자동 재시작 스크립트

### 진행 중 / 미완료

- [ ] 베이스라인(보정 없음) 220개 평가 → 이미지 보정 효과 정량 비교
- [ ] `result_parser.py`로 전체 220개 통합 DS 집계
- [ ] VLM 모듈 실제 구현
- [ ] risk 모듈 실제 구현 (TTC 기반 등)
- [ ] safety 필터 구현 (risk 점수 기반 brake)

### 알려진 이슈

| 이슈 | 원인 | 해결 방법 |
|------|------|-----------|
| `auto_eval_restart.sh` 실행 시 AttributeError (transforms3d) | numpy 2.0과 transforms3d 버전 충돌 | `pip install "numpy<2.0"` |
| MinSpeedTest FAILURE 다수 발생 | TF++ 베이스 모델 특성, 보정 코드와 무관 | 베이스라인과 비교 시 동일 조건이므로 무시 가능 |
| CARLA Signal=11 크래시 간헐적 발생 | CARLA 메모리 문제 (확인 필요) | `auto_eval_restart.sh`로 자동 복구 |
| task 3 RC 94.6% (타 task 대비 낮음) | 해당 루트 구성 문제 가능성 | 확인 필요 |

---

## 11. 다음 담당자가 바로 해야 할 일

| 우선순위 | 해야 할 일 | 관련 파일 | 시작 포인트 |
|---------|-----------|----------|------------|
| 1 | 베이스라인(보정 없음) 220개 평가 실행 | `auto_eval_restart.sh` | `GARAGE_EXT_CONFIG` 줄 주석 처리 후, `BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220_baseline_0418`, `TASK_LIST=(0 1 2 3)` 설정 후 실행 |
| 2 | 이미지 보정 vs 베이스라인 DS/RC 비교 | `carla_metric_result/tfpp_b2d_traj/*.json` | 양쪽 JSON의 `scores_mean` 비교 |
| 3 | 전체 220개 통합 DS 집계 | `Bench2Drive/tools/result_parser.py` | `python result_parser.py --xml ... --results ...` |
| 4 | VLM 모듈 실제 구현 | `src/garage_ext/modules/vlm/` | `docs/dev/MODULE_GUIDE.md` 참고, `dummy.py` 기반으로 새 파일 작성 |
| 5 | risk 모듈 실제 구현 (TTC 기반 등) | `src/garage_ext/modules/risk/` | 동일 |
| 6 | safety 필터 구현 | `src/garage_ext/modules/safety/` | risk score 기반으로 throttle 감소 또는 brake 강제 |

---

## 12. 추가 확인 필요 항목

1. **베이스라인 JSON 미완료**: `eval_bench2drive220_0.json`의 progress가 `[10, 55]`로 중단된 상태. 완전한 베이스라인 비교를 위해 새로 평가 필요.

2. **task 3 RC 저하 원인**: task 3의 RC가 94.617%로 타 task(99~100%) 대비 낮음. 해당 task의 루트 구성이나 특정 시나리오 유형 확인 필요.

3. **enhance_compare 시각화 저장 여부**: `SAVE_PATH=/mnt/2/carla_metric_result/carla_viz/task_N/enhance_compare/`에 실제로 이미지가 저장됐는지 확인 필요.

4. **CARLA Signal=11 크래시 패턴**: 특정 시나리오에서 재현되는지, 아니면 무작위인지 로그에서 확인 필요.

5. **result_parser.py 위치 확인**: 전체 통합 결과 집계 시 `Bench2Drive/tools/result_parser.py`를 써야 하는지, upstream `tools/result_parser.py`를 써야 하는지 확인 필요.

---

*저장 위치: `/mnt/2/carla_garage/HANDOFF_SUMMARY.md`*
