#!/usr/bin/env python3
"""W2S beta sweep: does varying beta trace a usable frontier?
All 2 rounds; baselines restricted to rounds 1-2 so the comparison is matched.
A2 omitted here — its plausibility requires cross_score under the big model."""
import sys, os, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

R = "experiments/bloom/runs_init/"
MAXR = 2
CELLS = ["qwen_selfharm", "qwen_selfpres", "llama_selfharm", "llama_selfpres"]
ROWS = [("BoN (beta=0)", "_pareto_b0"), ("big-jail ref", "_arm_floorb10"),
        ("A1 b=tuned", "_arm_w2sA1"), ("A1 b=2", "_arm_w2sA1b2"), ("A1 b=4", "_arm_w2sA1b4"),
        ("A3 b=0.25", "_arm_w2sA3b025"), ("A3 b=0.5", "_arm_w2sA3b05"),
        ("A3 b=1", "_arm_w2sA3b1"), ("A3 b=2", "_arm_w2sA3b2"),
        ("A4 b=1", "_arm_w2sA4b1"), ("A4 b=2", "_arm_w2sA4b2")]

def bop(run):
    pts = [p for p in PA.extract(run) if p["round"] <= MAXR]
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    return (st.mean(b["score"] for b in best), st.mean(b["prob"] for b in best),
            len({p["round"] for p in pts}))

for cell in CELLS:
    print(f"\n===== {cell} =====  (best-of-{MAXR}, score / prob)")
    for name, tag in ROWS:
        r = bop(R + cell + tag)
        if not r:
            continue
        warn = "" if r[2] >= MAXR else f"  <-- only {r[2]} round(s)"
        print(f"  {name:16s} {r[0]:6.2f} / {r[1]:5.1f}%{warn}")
