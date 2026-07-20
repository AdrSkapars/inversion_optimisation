#!/usr/bin/env python3
"""W2S arms vs baselines, best-of-pool on BOTH axes.

W2S runs are 2 rounds, so the 5-round baselines are RESTRICTED to rounds 1-2 (MAXR) — a
matched best-of-2 reduction rather than best-of-2 vs best-of-5.

A1/A3/A4 generate on the BIG target, so their stored gen_token_probs are the big model's and
are valid plausibility. A2 generates on the SMALL model: its stored probs are the SMALL
model's and are MEANINGLESS here — it needs cross_score.py under the big model, so its prob
is reported as '--' rather than a misleading number."""
import sys, os, glob, json, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

R = "experiments/bloom/runs_init/"
MAXR = 2
CELLS = ["qwen_selfharm", "qwen_selfpres", "llama_selfharm", "llama_selfpres"]
ARMS = [("BoN (beta=0)", "_pareto_b0", True), ("big-jail ref (tuned b)", "_arm_floorb10", True),
        ("A1 big tgt + small jail", "_arm_w2sA1", True),
        ("A2 both small", "_arm_w2sA2", False),          # prob invalid without cross_score
        ("A3 W2S neutral diff", "_arm_w2sA3b1", True),
        ("A4 W2S refusal diff", "_arm_w2sA4b1", True)]

def bop(run, prob_valid=True):
    pts = [p for p in PA.extract(run) if p["round"] <= MAXR]
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    return (st.mean(b["score"] for b in best),
            st.mean(b["prob"] for b in best) if prob_valid else None,
            len({p["round"] for p in pts}))

for cell in CELLS:
    base = bop(R + cell + "_pareto_b0")
    print(f"\n===== {cell} =====   (best-of-{MAXR})")
    print(f"  {'arm':26s} {'score':>6s} {'prob':>8s} {'vs BoN':>7s}  rnds")
    for name, tag, pv in ARMS:
        r = bop(R + cell + tag, pv)
        if not r:
            print(f"  {name:26s} {'(no data)':>6s}"); continue
        pr = f"{r[1]:7.1f}%" if r[1] is not None else "     --"
        d = f"{r[0]-base[0]:+.2f}" if base else "-"
        print(f"  {name:26s} {r[0]:6.2f} {pr:>8s} {d:>7s}  r{r[2]}")
