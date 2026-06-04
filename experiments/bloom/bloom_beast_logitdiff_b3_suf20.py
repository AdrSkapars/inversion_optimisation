"""Combo: β=3 + suffix_20 — strong jail but only after clean first 20 tokens."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/combo/b3_suf20"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 3.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "suffix", "n": 20}

if __name__ == "__main__":
    main()
