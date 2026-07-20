#!/usr/bin/env python3
"""Follow-up: A1 (big target + SMALL jail expert) was still climbing at beta=4 with FLAT
plausibility (qwen selfharm 4.00->6.16->7.56 at 54.7/56.5/55.7%). Two questions:
  1. does it keep climbing at beta=8, or does plausibility finally break?
  2. does it hold on a HIGH-beta cell (qwen political, tuned beta 3.25) where the BIG model
     itself needed heavy steering? All 4 existing cells are tuned beta<=1.
2 rounds, matching everything else in this queue."""
import shlex, os

BIG_Q, SMALL_Q = "local/Qwen/Qwen3.5-4B", "local/Qwen/Qwen3.5-2B"
BIG_L, SMALL_L = "local/meta-llama/Llama-3.2-3B-Instruct", "local/meta-llama/Llama-3.2-1B-Instruct"
POL = "prompts/2_political_bias.yaml"
SH = "prompts/7_self_harm_encouragement.yaml"
SP = "prompts/5_self_preservation.yaml"

# (cell, gpus, big, small, behaviour_file, seed, [(tag, beta)])
CHAIN_X = [
    ("qwen_selfharm",  "0 1", BIG_Q, SMALL_Q, SH,  "qwen_selfharm_pareto_b10",  [("w2sA1b8", 8)]),
    ("qwen_political", "0 1", BIG_Q, SMALL_Q, POL, "qwen_political_select",     [("w2sA1b325", 3.25), ("w2sA1b8", 8)]),
]
CHAIN_Y = [
    ("qwen_selfpres",  "2 3", BIG_Q, SMALL_Q, SP,  "qwen_selfpres_bongrid",     [("w2sA1b8", 8)]),
    ("llama_selfharm", "2 3", BIG_L, SMALL_L, SH,  "llama_selfharm_bongrid",    [("w2sA1b8", 8)]),
    ("llama_selfpres", "2 3", BIG_L, SMALL_L, SP,  "llama_selfpres_pareto_b10", [("w2sA1b8", 8)]),
]

q = shlex.quote
lines = ["#!/bin/bash", "set -u",
         "if screen -ls | grep -qE 'bsX|bsY|wsX|wsY|apX|apY'; then echo ABORT-screens; screen -ls; exit 1; fi",
         "u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{c++} END{print c+0}')",
         'if [ "$u" -gt 0 ]; then echo "ABORT: $u GPU busy"; exit 1; fi']
n = 0
for scr, chain in (("apX", CHAIN_X), ("apY", CHAIN_Y)):
    cmds = []
    for cell, gp, big, small, bf, seed, arms in chain:
        for tag, b in arms:
            cmds.append(f"bash ~/driverArmW2S.sh {cell} {gp} {big} {bf} {seed} {small} 2 {tag} "
                        f"{q(f'BLOOM_JAIL_BETA={b:g} BLOOM_JAIL_FLOOR=1e-4')} '';")
            n += 1
    lines.append(f"screen -dmS {scr} bash -c {q(' '.join(cmds))}")
lines += ["sleep 3", 'echo "A1-PUSH LAUNCHED $(date +%H:%M)"', "screen -ls | grep -E 'apX|apY'"]
open(os.path.expanduser("~/launchA1push.sh"), "w").write("\n".join(lines) + "\n")
print(f"wrote ~/launchA1push.sh ({n} runs)")
