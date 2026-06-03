"""Mode sweep: jail β=2 on even-indexed tokens (0,2,4,...), pure target on odd."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/alt/alt"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "alternate"}

if __name__ == "__main__":
    main()
