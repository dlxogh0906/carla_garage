# Architecture: carla_garage team extension layer

This document explains *why* the repo is laid out the way it is. Read this
first before adding a module.

## Problem

`carla_garage` upstream (autonomousvision) ships a large, monolithic
`team_code/` directory — e.g. `sensor_agent.py` (~1 kLOC),
`autopilot.py` (~2 kLOC), `model.py` (~1 kLOC). It is under active
development; merges from upstream are expected.

Our team wants to attach research modules (VLM, risk estimation, safety
filtering, …) and swap them per-experiment with low friction. Editing
`team_code/` directly would create perpetual merge conflicts and blur
"what's ours" during code review.

## Principle: additive, side-by-side

All team code lives in `src/garage_ext/`. `team_code/` is treated as
read-only. We reach into it by *importing and subclassing*, never by
modifying. Upstream merges become a normal fast-forward in the common
case.

```
carla_garage/
├── team_code/                 # upstream, do not edit
├── src/garage_ext/            # us
│   ├── agents/                # subclasses of upstream agents
│   ├── config/                # ExtConfig + YAML overlay loader
│   ├── modules/
│   │   ├── base.py            # Observation/Plan/Control/Risk dataclasses + Protocols
│   │   ├── vlm/
│   │   ├── risk/
│   │   └── safety/
│   ├── overrides/             # quarantine for forced divergences (with ORIGIN header)
│   ├── pipeline.py            # runs vlm -> risk -> safety around upstream control
│   └── registry.py            # (kind, name) -> class
├── configs/                   # YAML experiments (git-tracked)
├── tests/smoke/               # fast, no CARLA required (runs in CI)
├── docs/dev/
└── .github/                   # CI, templates, CODEOWNERS
```

## Data flow per step

```
input_data, timestamp
        │
        ▼
 upstream.SensorAgent.run_step ──► control (steer/throttle/brake)
        │                              │
        └─────────── obs, plan ────────┤
                                        ▼
                                 ExtPipeline.run:
                                   vlm.infer(obs)       (optional)
                                   risk.estimate(obs, plan)
                                   safety.filter(control, risk, obs)
                                        │
                                        ▼
                                   final control
```

The pipeline never replaces upstream perception/planning; it augments and
guards it.

## Why Protocol instead of a base class

`modules/base.py` uses `typing.Protocol`. A teammate can author a class
with the right method signature and skip inheritance boilerplate. The
pipeline only ever looks at method shape.

## Why a registry

`registry.py` maps `(kind, name)` to a class. YAML picks modules by name:

```yaml
vlm: my_vlm_v2
risk: heuristic_ttc
safety: brake_if_risk_gt_0_5
```

Teammates can add a new estimator without touching pipeline or other
modules, and experiments are reproducible from the YAML alone.

## Upstream sync policy

- One teammate rotates as "upstream warden" (monthly).
- `git fetch origin && git merge origin/leaderboard_2` onto `main`.
- Any file in `src/garage_ext/overrides/` pinned against a commit that
  upstream has since changed → rewrite or justify in the sync PR.

## When to use `overrides/`

Only when subclassing/wrapping is genuinely impossible: e.g. you need to
change the body of a function that isn't a hook point. Each file there
must start with:

```python
# ORIGIN: team_code/<path>
# REASON: <why we can't subclass>
# SYNC-CHECK: <upstream commit hash we branched from>
```

If `overrides/` is growing, that's a signal to upstream a hook point or
rethink the design.
