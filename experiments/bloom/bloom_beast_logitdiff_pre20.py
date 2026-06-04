"""Mode sweep: jail β=2 for first 20 tokens of each turn, then pure target."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/prefix_sweep/prefix_20"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "prefix", "n": 20}

if __name__ == "__main__":
    main()
