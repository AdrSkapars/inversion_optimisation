#!/usr/bin/env python3
"""Compute-allocation ratio rays for ONE cell (qwen self-harm, jail beta1).
Each ray = a fixed rounds:turns ratio; along it compute (=R*T) grows. Shows whether a
fixed compute budget is better spent on turns, rounds, or a balanced mix.
Reads best-of-pool avg score/plausibility per (R,T) folder. Output ~/ratio_rays.png.
"""
import sys, os, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "experiments/bloom/runs_init"
TAG = "qwen_selfharm"
TITLE = "qwen · self-harm · jail β1"

def reduce_run(R, T):
    f = f"{ROOT}/{TAG}_scale_r{R}t{T}_b10"
    if not os.path.exists(os.path.join(f, f"round_{R}", "judgment.json")):
        return None
    pts = PA.extract(f)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda p: p["score"]) for v in g.values()]
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best)

# label, [(R,T) along ray], colour, marker
RAYS = [
    ("rounds only  (R:T = 1:0)", [(r, 1) for r in range(1, 7)],  "#7c3aed", "s"),
    ("2:1  (2 rounds : 1 turn)", [(2, 1), (4, 2), (6, 3)],       "#0891b2", "v"),
    ("1:1  (balanced)",          [(1, 1), (2, 2), (3, 3), (4, 4)], "#16a34a", "D"),
    ("1:2  (1 round : 2 turns)", [(1, 2), (2, 4), (3, 6)],       "#f59e0b", "^"),
    ("turns only  (R:T = 0:1)",  [(1, t) for t in range(1, 11)], "#dc2626", "o"),
]

fig, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)
for lab, pts, cl, mk in RAYS:
    rows = []
    for (R, T) in pts:
        red = reduce_run(R, T)
        if red:
            rows.append((R * T, red[1], red[0]))
    rows.sort()
    if rows:
        ax_s.plot([r[0] for r in rows], [r[1] for r in rows], marker=mk, color=cl, lw=1.9, label=lab)
        ax_p.plot([r[0] for r in rows], [r[2] for r in rows], marker=mk, color=cl, lw=1.9, label=lab)
ax_s.set_ylabel("avg behaviour score (0–10)"); ax_s.set_ylim(0, 10.5)
ax_s.set_title(TITLE + "  —  elicitation", fontsize=11)
ax_p.set_ylabel("target plausibility (%)")
ax_p.set_title(TITLE + "  —  plausibility", fontsize=11)
for ax in (ax_s, ax_p):
    ax.set_xlabel("compute  ~  rounds × turns")
    ax.grid(True, color="#e1e0d9", lw=0.5)
    ax.legend(fontsize=8.5, frameon=False, title="rounds : turns")
fig.suptitle("Compute allocation — rounds:turns ratio rays  ·  best-of-pool avg", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.expanduser("~/ratio_rays.png")
fig.savefig(out, facecolor="white")
print("saved", out)
