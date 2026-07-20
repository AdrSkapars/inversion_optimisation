#!/usr/bin/env python3
"""Generate ~/launchTF.sh: floor-ON vs floor-OFF at each cell's TUNED beta, across all
models x behaviours. Both arms seeded from the SAME existing run's understanding+ideation
(25 scenarios) so each pair is scenario-matched. 3 turns, 3 rounds, seed 1.
Skips cells whose tuned beta is 0 (no steering => floor cannot bind)."""
import json, glob, os

R = "experiments/bloom/runs_init/"
MID = {"qwen": "local/Qwen/Qwen3.5-4B",
       "llama": "local/meta-llama/Llama-3.2-3B-Instruct",
       "gemma": "local/google/gemma-4-e4b-it",
       "phi": "local/microsoft/Phi-4-mini-instruct"}
BF = {"selfharm": "prompts/7_self_harm_encouragement.yaml",
      "medical": "prompts/8_dangerous_medical_advice.yaml",
      "political": "prompts/2_political_bias.yaml",
      "delusions": "prompts/3_reinforcing_delusions.yaml",
      "deception": "prompts/4_strategic_deception.yaml",
      "selfpres": "prompts/5_self_preservation.yaml",
      "racial": "prompts/1_racial_bias.yaml",
      "goblin": "prompts/9_goblin_fixation.yaml"}

tune = json.load(open("experiments/bloom/jail_tune_best_betas.json"))

def seed_for(model, beh, tb):
    """existing run at this cell whose ideation we reuse (prefer one at the tuned beta)."""
    best = None
    for d in glob.glob(f"{R}{model}_{beh}_*"):
        p = os.path.join(d, "round_1", "ideation.json")
        if not os.path.exists(p):
            continue
        try:
            n = len(json.load(open(p)).get("variations", []))
        except Exception:
            continue
        if n != 25:
            continue
        score = 1
        cfgp = os.path.join(d, "round_1", "cfg.json")
        try:
            j = (json.load(open(cfgp)).get("jailbroken_output", {}) or {})
            if j.get("beta") is not None and abs(float(j["beta"]) - float(tb)) < 1e-9:
                score = 0          # prefer same-beta run
        except Exception:
            pass
        cand = (score, os.path.basename(d))
        if best is None or cand < best:
            best = cand
    return best[1] if best else None

cells, skipped = [], []
for m in tune:
    for b in tune[m]:
        tb = tune[m][b].get("beta")
        if tb is None or float(tb) == 0.0:
            skipped.append((m, b, tb, "tuned beta = 0")); continue
        s = seed_for(m, b, tb)
        if not s:
            skipped.append((m, b, tb, "no 25-scenario ideation")); continue
        cells.append((m, b, float(tb), s))

cells.sort()
lines = ["#!/bin/bash", "set -u",
         "if screen -ls | grep -qE 'tfA|tfB|arA|arB|flA|flB|coA|coB'; then echo ABORT-screens; screen -ls; exit 1; fi",
         "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{c++} END{print c+0}')",
         'if [ "$used" -gt 0 ]; then echo "ABORT: $used GPU busy"; exit 1; fi']
streams = {0: [], 1: []}
for i, (m, b, tb, s) in enumerate(cells):
    st = i % 2
    eg, tg = (0, 1) if st == 0 else (2, 3)
    tag = f"{m}_{b}"
    for arm, env in (("tfoff", f"BLOOM_JAIL_BETA={tb:g}"),
                     ("tfon",  f"BLOOM_JAIL_BETA={tb:g} BLOOM_JAIL_FLOOR=1e-4")):
        streams[st].append(
            f'bash ~/driverArm.sh {tag} {eg} {tg} {MID[m]} {BF[b]} {s} 3 {arm} "{env}";')
lines.append("screen -dmS tfA bash -c '" + " ".join(streams[0]) + "'")
lines.append("screen -dmS tfB bash -c '" + " ".join(streams[1]) + "'")
lines.append("sleep 3; echo LAUNCHED $(date +%H:%M); screen -ls | grep -E 'tfA|tfB'")

open("/home/t75879as/launchTF.sh", "w").write("\n".join(lines) + "\n")
print(f"cells={len(cells)}  runs={len(cells)*2}  streamA={len(streams[0])} streamB={len(streams[1])}")
print("skipped:", skipped)
for m, b, tb, s in cells[:6]:
    print(f"  {m:6s} {b:10s} beta={tb:<5g} seed={s}")
