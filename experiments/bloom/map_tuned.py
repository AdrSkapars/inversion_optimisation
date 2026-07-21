#!/usr/bin/env python3
"""For each (model, behaviour) in jail_tune_best_betas.json, find an existing FLOOR-OFF jail run
at exactly the tuned beta, and report its round count + scenario count so a floor-ON run can be
launched with a MATCHED config (pairing per-cell, not globally)."""
import json, glob, os

R = "experiments/bloom/runs_init/"
tune = json.load(open("experiments/bloom/jail_tune_best_betas.json"))

# model key -> folder prefix
MODEL = {"qwen": "qwen", "llama": "llama", "gemma": "gemma", "phi": "phi"}

rows = []
for m in tune:
    for b in tune[m]:
        tb = tune[m][b].get("beta")
        if tb is None:
            continue
        pref = f"{MODEL[m]}_{b}_"
        cands = []
        for d in glob.glob(R + pref + "*"):
            cfg = os.path.join(d, "round_1", "cfg.json")
            if not os.path.exists(cfg):
                continue
            try:
                c = json.load(open(cfg))
            except Exception:
                continue
            j = c.get("jailbroken_output", {}) or {}
            if not j.get("use_during_rollout"):
                continue
            fl = j.get("target_floor")
            fl = float(fl) if fl not in (None, "") else 0.0
            if fl > 0:                       # want the FLOOR-OFF arm
                continue
            if j.get("beta") is None or abs(float(j["beta"]) - float(tb)) > 1e-9:
                continue
            if j.get("b1") is not None:      # exclude b1=0 (floor-only/jailonly variants)
                continue
            nr = len(glob.glob(os.path.join(d, "round_*", "judgment.json")))
            try:
                nv = len(json.load(open(os.path.join(d, "round_1", "judgment.json"))).get("judgments", []))
            except Exception:
                nv = 0
            cands.append((nr, nv, os.path.basename(d)))
        cands.sort(reverse=True)             # prefer most rounds
        if cands:
            nr, nv, name = cands[0]
            rows.append((m, b, tb, nr, nv, name))
        else:
            rows.append((m, b, tb, 0, 0, None))

have = [r for r in rows if r[5]]
miss = [r for r in rows if not r[5]]
print(f"{'model':6s} {'behaviour':11s} {'beta':>5s} {'rnds':>4s} {'scen':>4s}  floor-off run")
for r in sorted(rows):
    print(f"{r[0]:6s} {r[1]:11s} {r[2]!s:>5s} {r[3]:4d} {r[4]:4d}  {r[5] or '*** NONE ***'}")
print(f"\nmatched: {len(have)}/{len(rows)}   missing: {len(miss)}")
tot = sum(r[3] for r in have)
print(f"total rounds to replicate for floor-ON: {tot}  (~{tot*4+len(have)*3} min serial, /2 GPUs => ~{(tot*4+len(have)*3)/120:.1f} h)")
