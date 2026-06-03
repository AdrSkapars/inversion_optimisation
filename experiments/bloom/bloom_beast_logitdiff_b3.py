"""Sweep variant: jail enabled, beta=3.0, schedule=all. Beta sweep point."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/beta_sweep/beta_3"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 3.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
