"""Turn-level annealing: clean start, escalate to strong jail.

beta = [0, 2, 3] — turn 0 is pure target (sets natural opening), turn 1 mid jail,
turn 2 strong jail. Hypothesis: natural opening with escalating push to elicit,
gives high naturalness on early reply, high elic by end.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/turn_anneal/lowhi"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = [0, 2, 3]
cfg["jailbroken_output"]["token_schedule"] = {"mode": "all"}

if __name__ == "__main__":
    main()
