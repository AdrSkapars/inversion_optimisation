"""Combo: β=1 + alternate — moderate jail every other token."""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/combo/b1_alt"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 1.0
cfg["jailbroken_output"]["token_schedule"] = {"mode": "alternate"}

if __name__ == "__main__":
    main()
