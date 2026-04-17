"""Quarantine for files that had to diverge from upstream.

Only put something here when subclassing / wrapping is not enough (e.g.
you must change a function body inside team_code that isn't a hook
point). Each override file MUST start with a comment block containing:

    # ORIGIN: team_code/<path>
    # REASON: <why upstream can't be subclassed here>
    # SYNC-CHECK: <commit hash of upstream file this was branched from>

Reviewers can then diff against the pinned commit during upstream merges.
"""
