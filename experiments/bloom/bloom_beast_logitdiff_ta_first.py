"""Turn-level annealing: jail only on first turn.

beta = [2, 0, 0] — jail sets direction in turn 0, then pure target for turns 1-2.
Tests whether the model naturally elaborates once the initial assistant reply
has been steered, with no further intervention.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/turn_anneal/first_only"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = [2, 0, 0]
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
