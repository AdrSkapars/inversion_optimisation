"""Combo: β=2 + suffix_10 — minimal clean prefix with full-strength jail.

Tests whether even 10 clean tokens up front are enough to preserve naturalness
while keeping most of the elicitation of full β=2.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/combo/b2_suf10"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "suffix", "n": 10}

if __name__ == "__main__":
    main()
