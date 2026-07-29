# 팀 온보딩 매뉴얼 (한국어)

> 처음 이 저장소에서 일하게 된 사람이 읽는 문서. 30분 안에 "내 모듈 붙여서 돌려보기"까지 간다.

---

## 1. 이 저장소가 뭔가

- **원본**: [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage) — TransFuser++ 기반 CARLA 자율주행 연구 코드베이스.
- **우리 쪽**: [dlxogh0906/carla_garage](https://github.com/dlxogh0906/carla_garage) — 위 저장소를 fork 해서 팀의 확장(VLM, 위험도, safety filter 등)을 붙여 쓰는 작업 공간.
- 원본은 지금도 계속 업데이트됨 (최근에 TFv6 추가). **우리는 원본 파일을 직접 수정하지 않는다** — 대신 옆에 새 디렉토리 `src/garage_ext/`를 만들어 거기서만 작업한다.

---

## 2. 무엇이 바뀌었나 (Before → After)

### Before
```
carla_garage/
├── team_code/              # 원본 코드 (sensor_agent.py, model.py, config.py, …)
├── leaderboard/            # 원본
├── scenario_runner/        # 원본
└── Bench2Drive/            # 원본
```
새 모듈을 붙이려면 `team_code/sensor_agent.py` 같은 파일을 직접 고쳐야 해서, 원본이 업데이트될 때마다 merge 충돌이 생긴다.

### After (지금 상태)
```
carla_garage/
├── team_code/              # 원본. ✋ 건드리지 않음
├── leaderboard/            # 원본. ✋
├── scenario_runner/        # 원본. ✋
├── Bench2Drive/            # 원본. ✋
│
├── src/garage_ext/         # ⭐ 우리 코드는 전부 여기
│   ├── agents/             # 원본 SensorAgent를 '상속(subclass)'해서 확장
│   ├── config/             # 원본 GlobalConfig 위에 우리 설정을 얹는 레이어
│   ├── modules/            # ★ 팀원들이 주로 작업할 곳
│   │   ├── base.py         # 모듈이 지켜야 할 인터페이스 정의
│   │   ├── vlm/            # Vision-Language 모듈
│   │   ├── risk/           # 위험도 판단 모듈
│   │   └── safety/         # 안전 필터 모듈
│   ├── overrides/          # 어쩔 수 없이 원본을 비틀어야 할 때만 (거의 쓸 일 없음)
│   ├── pipeline.py         # vlm → risk → safety 순서로 연결해주는 파이프라인
│   └── registry.py         # "이름 → 클래스" 레지스트리 (YAML 한 줄로 모듈 교체)
│
├── configs/                # 실험 YAML 설정
│   ├── base.yaml           # 기본값
│   └── experiments/        # 실험별 파일 (예: my_vlm_run.yaml)
│
├── tests/smoke/            # CARLA 없이 돌아가는 빠른 테스트 (CI에서 실행)
├── docs/dev/               # 개발자 문서
├── .github/                # PR 템플릿, CODEOWNERS, CI 워크플로
└── pyproject.toml          # 패키지 설치 설정
```

### 핵심 아이디어 3줄 요약
1. **원본은 읽기 전용 폴더**. 원본 팀이 업데이트해도 우리 쪽은 영향을 안 받는다.
2. **우리 확장은 전부 `src/garage_ext/` 안에서만** 자란다.
3. **실험은 YAML로** 바꾼다 — 코드를 고치는 게 아니라 `risk: my_module_v2` 한 줄을 수정한다.

---

## 3. 한 번만 하는 셋업

### 3.1 저장소 가져오기
```bash
git clone git@github.com:dlxogh0906/carla_garage.git
cd carla_garage
```

### 3.2 원본 remote 추가 (업스트림 업데이트를 받기 위함)
```bash
git remote add origin https://github.com/autonomousvision/carla_garage.git
git remote rename origin upstream 2>/dev/null || true
# ↑ 이미 origin이 팀 fork일 수도 있으니 상황에 맞게
git remote -v
```
목표 상태: `origin` = 팀 fork, `upstream` = autonomousvision 원본.

### 3.3 확장 패키지 설치
```bash
pip install -e ".[dev]"
```
이러면 `garage_ext`를 파이썬에서 `import garage_ext` 로 쓸 수 있게 된다.

### 3.4 동작 확인
```bash
pytest tests/smoke -q
```
4개 테스트가 통과하면 OK.

---

## 4. 평소 작업 루프

### 4.1 브랜치 규칙
| 접두어 | 용도 | 예시 |
|---|---|---|
| `main` | 팀 공용 안정 브랜치. PR로만 들어간다 | `main` |
| `feat/` | 리뷰 받아 공유할 기능 브랜치 | `feat/risk-ttc-v1` |
| `exp/` | 개인 실험 브랜치. 더러워도 OK. 머지 안 해도 됨 | `exp/kwy-20260417` |

### 4.2 내 실험 시작하기
```bash
git fetch origin
git checkout -b exp/<내이름>-<날짜> origin/main
# ↑ main이 아직 없으면 feat/ext-scaffold 또는 upstream/leaderboard_2 에서 시작
```

### 4.3 작업 중
```bash
# 코드는 src/garage_ext/ 안에서만 수정
# 수정 후
pytest tests/smoke -q
git add src/garage_ext/... configs/...
git commit -m "..."
git push origin exp/<내이름>-<날짜>
```

### 4.4 기능을 팀에 공유하고 싶을 때 (PR)
```bash
git checkout -b feat/<내모듈이름>
# 실험 브랜치에서 핵심 커밋만 골라 cherry-pick 하거나 정리
git push origin feat/<내모듈이름>
gh pr create --base main
```
PR 템플릿이 자동으로 떠서 체크리스트에 답하면 된다.

### 4.5 원본 업스트림 동기화 (월 1회, 당번제)
```bash
git checkout main
git fetch upstream
git merge upstream/leaderboard_2
# 보통 충돌 없음 — 우리가 team_code/를 안 건드렸기 때문
git push origin main
```

---

## 5. 새 모듈 붙이기 (VLM / risk / safety)

예: "TTC(충돌까지 남은 시간) 기반 risk 모듈"을 만든다고 가정.

### 5.1 파일 생성
`src/garage_ext/modules/risk/ttc_v1.py`:
```python
from typing import Any
from ...registry import register
from ..base import Observation, Plan, RiskReport


@register("risk", "ttc_v1")   # ← "risk 카테고리에 ttc_v1 이름으로 등록"
class TTCRisk:
    def __init__(self, threshold: float = 2.0, **_: Any) -> None:
        self.threshold = threshold

    def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
        # obs.data 안에 원본 agent 상태, input_data 등이 들어있음
        ttc = _compute_ttc(obs)
        score = 1.0 if ttc < self.threshold else 0.0
        return RiskReport(score=score, reasons=[f"ttc={ttc:.2f}"])
```

### 5.2 레지스트리에 연결
`src/garage_ext/modules/risk/__init__.py`에 한 줄 추가:
```python
from . import ttc_v1  # noqa: F401
```

### 5.3 스모크 테스트 추가
`tests/smoke/test_risk_ttc.py`:
```python
def test_ttc_registered():
    import garage_ext.modules  # noqa: F401
    from garage_ext.registry import available, build

    assert "ttc_v1" in available("risk")
    est = build("risk", "ttc_v1", threshold=1.5)
    assert est.threshold == 1.5
```

### 5.4 실험 YAML 만들기
`configs/experiments/my_ttc_run.yaml`:
```yaml
extends: ../base.yaml

risk: ttc_v1
risk_kwargs:
  threshold: 1.5

meta:
  owner: <내이름>
  tag: ttc-first-try
```

### 5.5 돌리기
```bash
export GARAGE_EXT_CONFIG=configs/experiments/my_ttc_run.yaml
# 평소처럼 leaderboard_evaluator를 돌리되, agent를
# src/garage_ext/agents/ext_sensor_agent.py 의 ExtSensorAgent 로 지정
```

**이게 전부다.** 원본 파일은 한 글자도 안 고쳤다.

---

## 6. 각 파일의 역할 한 줄 설명

| 파일 | 역할 |
|---|---|
| `src/garage_ext/registry.py` | `@register("risk", "ttc_v1")` 같은 데코레이터 구현 |
| `src/garage_ext/modules/base.py` | 모듈이 따라야 할 인터페이스 (Protocol) 와 데이터 클래스 정의 |
| `src/garage_ext/pipeline.py` | vlm → risk → safety 순서로 호출해주는 컨트롤러 |
| `src/garage_ext/config/ext_config.py` | YAML을 읽어서 ExtConfig 객체를 만듦 |
| `src/garage_ext/agents/ext_sensor_agent.py` | 원본 `SensorAgent`를 상속. `run_step()` 끝에 파이프라인을 실행 |
| `configs/base.yaml` | 실험 설정의 기본값 (모두 noop) |
| `configs/experiments/*.yaml` | 개별 실험 설정. `extends: ../base.yaml` 로 기본값 위에 얹음 |
| `tests/smoke/` | CARLA 없이 몇 초 안에 끝나는 테스트. CI에서 필수로 돈다 |
| `.github/workflows/ci.yml` | GitHub Actions — lint + smoke test |
| `docs/dev/ARCHITECTURE.md` | 구조를 왜 이렇게 짰는지 (영문) |
| `docs/dev/CONTRIBUTING.md` | 브랜치/PR 규칙 (영문) |
| `docs/dev/MODULE_GUIDE.md` | 새 모듈 5단계 가이드 (영문) |

---

## 7. 자주 만나는 상황 FAQ

**Q. 원본 `team_code/sensor_agent.py`를 딱 한 줄만 고치고 싶은데.**
A. 참는다. 대신 `src/garage_ext/agents/ext_sensor_agent.py`에서 그 메서드를 override 한다. 정말 override로도 안 되면 `src/garage_ext/overrides/`에 파일명과 이유 헤더를 붙여 격리한다.

**Q. 내 모듈이 원본 agent의 내부 상태(예: LiDAR 텐서)를 쓰고 싶은데.**
A. `ExtSensorAgent.run_step` 안에서 이미 `obs.data["agent"] = self`로 agent 객체를 넘긴다. 모듈 안에서 `obs.data["agent"].<attr>` 로 접근하면 된다.

**Q. 팀원이 같은 이름으로 risk 모듈을 등록해버렸다.**
A. `registry.register`가 중복 등록을 막아서 에러가 난다. 네이밍 규칙: `<알고리즘>_v<번호>` (예: `ttc_v1`, `learned_v2`).

**Q. 실험 설정에 없는 키를 YAML에 적어버렸다.**
A. `ExtConfig.apply_overlay`가 모르는 키를 `meta["_unknown"]`에 모아둔다. 크래시 없이 진행되지만 로그에서 확인 가능. 오타 방지용 안전장치.

**Q. CI가 실패했다.**
A. 대부분 (1) `yapf --diff`가 포맷 불일치를 잡았거나 (2) 새 모듈을 `__init__.py`에서 import 안 해서 등록 테스트가 깨졌거나. 로컬에서 `yapf -i -r src/ tests/` → `pytest tests/smoke -q` 순으로 돌리면 재현된다.

**Q. 원본이 크게 바뀌어서 merge 충돌이 났다.**
A. 우리 코드는 `src/garage_ext/`에만 있으므로 충돌은 보통 `team_code/` 내부에서 원본끼리의 충돌이다 — 우리가 해결할 게 거의 없다. 만약 `overrides/`가 깨졌다면 그 파일의 `SYNC-CHECK` 커밋 해시를 기준으로 diff를 다시 떠서 포팅한다.

---

## 8. 더 읽을 거리

- 구조적 의도·트레이드오프: [ARCHITECTURE.md](../ARCHITECTURE.md)
- 브랜치·PR 규칙 상세: [CONTRIBUTING.md](CONTRIBUTING.md)
- 새 모듈 추가 예제: [MODULE_GUIDE.md](MODULE_GUIDE.md)
- 원본 프로젝트 README: [../../README.md](../../README.md)

---

## 9. 막히면

1. `docs/dev/` 를 먼저 뒤진다.
2. 관련 GitHub Issue가 있는지 검색.
3. `.github/ISSUE_TEMPLATE/`에서 맞는 템플릿으로 새 Issue.
4. 개인 실험 브랜치(`exp/…`)에서 막힌 거라면 그냥 커밋해서 푸시한 뒤 링크 공유 — 팀원이 바로 재현할 수 있다.