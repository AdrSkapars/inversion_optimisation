"""Combo: β=1 + suffix_20 — moderate jail, clean first 20 tokens.

Pareto candidate: lower-strength jail only on later tokens. Hypothesis: less
elic than β=2 full, but better naturalness (clean opening + half-strength
intervention afterwards) — possibly a new Pareto point.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/combo/b1_suf20"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 1.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "suffix", "n": 20}

if __name__ == "__main__":
    main()
