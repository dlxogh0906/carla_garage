# Adding a new module in 5 steps

We'll add a fictional `ttc_v1` risk estimator.

## 1. Create the file

`src/garage_ext/modules/risk/ttc_v1.py`:

```python
from typing import Any

from ...registry import register
from ..base import Observation, Plan, RiskReport


@register("risk", "ttc_v1")
class TTCRisk:
    def __init__(self, threshold: float = 2.0, **_: Any) -> None:
        self.threshold = threshold

    def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
        # read whatever you need from obs.data; return a RiskReport
        ttc = _compute_ttc(obs)
        score = 1.0 if ttc < self.threshold else 0.0
        return RiskReport(score=score, reasons=[f"ttc={ttc:.2f}"])
```

## 2. Expose it to registry on import

Open `src/garage_ext/modules/risk/__init__.py` and add:

```python
from . import ttc_v1  # noqa: F401
```

(Any import of the file is enough. Our `modules/__init__.py` already
imports the `risk` package.)

## 3. Add a smoke test

`tests/smoke/test_risk_ttc.py`:

```python
def test_ttc_registered():
    import garage_ext.modules  # noqa: F401
    from garage_ext.registry import available, build
    assert "ttc_v1" in available("risk")
    est = build("risk", "ttc_v1", threshold=1.5)
    assert est.threshold == 1.5
```

## 4. Use it from an experiment YAML

`configs/experiments/my_ttc_run.yaml`:

```yaml
extends: ../base.yaml
risk: ttc_v1
risk_kwargs:
  threshold: 1.5
meta:
  owner: <your handle>
  tag: ttc-first-try
```

## 5. Run

```bash
export GARAGE_EXT_CONFIG=configs/experiments/my_ttc_run.yaml
# launch leaderboard evaluator pointing at ExtSensorAgent as usual
```

That's it — you never touched upstream code, the swap is one YAML line,
and teammates can review just your module file + test.

## Signals that you're fighting the architecture

- You want to modify a team_code file → instead, subclass it in
  `src/garage_ext/agents/` and override just the method you need.
- You want to add a parameter to `ExtConfig` → fine, add a field; keep
  it `Optional` with a safe default so existing YAMLs don't break.
- You want a module to read state from inside the agent → use
  `obs.data["agent"] = self` pattern (already wired in
  `ExtSensorAgent.run_step`).
