"""Mode sweep: pure target first 50 tokens, then jail β=2 for the rest of each turn."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/suffix_sweep/suffix_50"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "suffix", "n": 50}

if __name__ == "__main__":
    main()
