"""Mode sweep: pure target first 20 tokens, then jail β=2 — Pareto candidate."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/suffix_sweep/suffix_20"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "suffix", "n": 20}

if __name__ == "__main__":
    main()
