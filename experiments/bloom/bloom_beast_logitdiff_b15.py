"""Beta sweep: β=1.5 — fills Pareto gap between β=1 and β=2."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/beta_sweep/beta_1.5"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 1.5
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
