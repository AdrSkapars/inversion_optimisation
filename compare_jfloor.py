#!/usr/bin/env python3
"""For each *_jfloor run (jail beta=1, b1=0, floor=1e-4), find sibling jail runs on the same
model+behaviour and report best-of-pool plausibility / score, so we can see what the
naturalness floor actually does on the jail path."""
import json, glob, os, statistics as st
import sys
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

def reduce_run(folder):
    pts = PA.extract(folder)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda p: p["score"]) for v in g.values()]
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best), len(g)

def cfg_of(folder):
    p = os.path.join(folder, "round_1", "cfg.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    j = c.get("jailbroken_output", {}) or {}
    if not j.get("use_during_rollout"):
        return None
    fl = j.get("target_floor")
    return {"beta": j.get("beta"), "b1": j.get("b1"),
            "floor": float(fl) if fl not in (None, "") else 0.0}

jfloors = sorted(glob.glob("experiments/bloom/runs_*/*_jfloor"))
for jf in jfloors:
    stem = os.path.basename(jf)[:-len("_jfloor")]          # e.g. qwen_selfharm
    root = os.path.dirname(jf)
    print(f"\n===== {stem} =====")
    sibs = sorted(glob.glob(os.path.join(root, stem + "*")))
    for s in sibs:
        c = cfg_of(s)
        if not c:
            continue
        if c["beta"] in (None, 0.0) and "jfloor" not in s:
            pass  # keep beta=0 too, it's the BoN reference
        r = reduce_run(s)
        if not r:
            continue
        plaus, score, n = r
        tag = "  <-- FLOOR ON" if c["floor"] > 0 else ""
        print(f"  {os.path.basename(s)[:46]:48s} b={c['beta']} b1={c['b1']} floor={c['floor']:<8g}"
              f" plaus={plaus:5.1f} score={score:5.2f} n={n}{tag}")
