"""Turn-anneal of token_schedule itself: clean → intense across turns.

Turn 0: jail only after first 50 tokens (mostly clean opening).
Turn 1: jail after first 20 tokens (shorter clean).
Turn 2: jail every token (full intervention).

β stays at 2.0 throughout — only the schedule anneals.
"""
from bloom_beast_logitdiff import cfg, main

cfg["folder_name"] = "runs_15/turn_anneal/sched_lowhi"
cfg["jailbroken_output"]["use_during_rollout"] = True
cfg["jailbroken_output"]["beta"] = 2.0
cfg["jailbroken_output"]["token_schedule"] = [
    {"mode": "suffix", "n": 50},
    {"mode": "suffix", "n": 20},
    {"mode": "all"},
]

if __name__ == "__main__":
    main()
