"""Sweep variant: jail enabled, beta=2.0, schedule=all (every token).

Anchor for the beta sweep and the shared baseline for the prefix/suffix/alt
sweeps. Equivalent to running bloom_beast_logitdiff.py with
jailbroken_output.use_during_rollout=True and the existing defaults.

See bloom_beast_logitdiff.py for the full config; this launcher only mutates the
fields that differ from the base.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/beta_sweep/beta_2"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
