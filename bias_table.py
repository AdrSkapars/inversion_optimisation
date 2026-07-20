#!/usr/bin/env python3
"""Logit-bias arms, best-of-pool on BOTH axes, vs BoN and the beta=1 reference.
All rows here are JAIL-path runs, so PA.extract (ARITHMETIC mean token-prob) is internally
consistent. corrupt-only is NOT included: corruption persists a GEOMETRIC mean in a different
file, and mixing the two statistics produced a fake ~20pp gap once already."""
import sys, os, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

R = "experiments/bloom/runs_init/"
CELLS = ["qwen_selfharm", "llama_selfpres"]
ARMS = [("BoN (beta=0)", "_pareto_b0"), ("beta=1 jail ref", "_arm_floorb10"),
        ("raw log p  L=1", "_arm_bias1"), ("raw log p  L=3", "_arm_bias3"),
        ("contrast   L=1", "_arm_biasC1"), ("contrast   L=3", "_arm_biasC3"),
        ("contrast top200", "_arm_biasT3"), ("manual words L=5", "_arm_biasW5"),
        ("jail-prompt frozen", "_arm_biasJ3"),
        ("avg contrast 8x4", "_arm_biasA3"), ("avg contrast 16x8", "_arm_biasA8")]

def bop(run):
    pts = PA.extract(run)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    return (st.mean(b["score"] for b in best), st.mean(b["prob"] for b in best),
            len({p["round"] for p in pts}))

for cell in CELLS:
    base = bop(R + cell + "_pareto_b0")
    print(f"\n===== {cell} =====")
    print(f"  {'arm':20s} {'score / prob':>15s} {'vs BoN':>7s}  rounds")
    for name, tag in ARMS:
        r = bop(R + cell + tag)
        if not r:
            print(f"  {name:20s} {'(no data)':>15s}"); continue
        flag = "" if r[2] == 5 else f"  <-- PARTIAL {r[2]}/5, do not quote"
        d = f"{r[0]-base[0]:+.2f}" if base else "-"
        print(f"  {name:20s} {r[0]:6.2f} / {r[1]:5.1f}% {d:>7s}  r{r[2]}{flag}")
