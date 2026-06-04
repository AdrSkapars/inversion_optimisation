"""Beta sweep: β=0.25 — fills Pareto gap between baseline and β=0.5."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/beta_sweep/beta_0.25"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 0.25
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
