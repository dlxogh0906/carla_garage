# Contributing

## Branch model

| Branch               | Purpose                                       |
|----------------------|-----------------------------------------------|
| `main`               | Stable. Only receives PR merges + upstream syncs. |
| `feat/<module>-<topic>` | A merged, reviewed feature (e.g. `feat/risk-ttc`).  |
| `exp/<user>-<date>`  | Personal experiment branch. May be messy. Not required to merge. |
| upstream `leaderboard_2` | Tracked via `origin` remote; synced into `main`. |

Each teammate's personal fork remote is named e.g. `myfork`. Push
experiment branches there. PRs target `main` on the shared repo.

## Daily loop

```bash
# pull upstream into your exp branch before starting work
git fetch origin
git rebase origin/leaderboard_2        # OR: git merge, per team preference

# ... do work in src/garage_ext/ only ...

pytest tests/smoke -q                  # fast local check
yapf -i -r src/ tests/                 # autoformat

git push myfork exp/<user>-<date>
gh pr create --base main               # when ready to share
```

## Rules

1. **Never edit** `team_code/`, `leaderboard/`, `leaderboard_autopilot/`,
   `scenario_runner/`, `scenario_runner_autopilot/`, `Bench2Drive/`.
   Extend via subclassing in `src/garage_ext/agents/` instead.
2. Add new modules under `src/garage_ext/modules/<kind>/<name>.py` and
   register them with `@register(kind, name)`.
3. Ship experiments as YAML under `configs/experiments/` so results are
   reproducible from `git checkout` alone.
4. Every module PR needs at least one smoke test that imports it and
   checks its registration.
5. One PR, one idea. Mixed "new module + refactor" PRs get split.

## Reviewing

Check: did the PR touch `team_code/`? Did it update `configs/` if it
added a config key? Are module docstrings clear about what the module
reads / writes?
