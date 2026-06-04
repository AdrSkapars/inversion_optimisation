"""Turn-level annealing: strong start, fade to clean.

beta = [3, 2, 0] — turn 0 strong jail (set direction), turn 1 mid, turn 2 pure
target. Hypothesis: jail pushes the conversation into bad territory early, then
the model naturally elaborates without the unnaturalness penalty on the final
reply. Should give the highest out_lp on turn 2 specifically.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/turn_anneal/hilow"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = [3, 2, 0]
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
