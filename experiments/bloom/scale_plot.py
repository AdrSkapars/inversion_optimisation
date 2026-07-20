#!/usr/bin/env python3
"""Rounds-vs-turns compute-scaling plot for the two cells.
Reads *_scale_r{R}t{T}_b{bc} run folders, reduces each to (compute=R*T, plaus, score)
via best-of-pool-by-score per scenario, and plots elicitation & plausibility vs compute
for turns-scaling (R=1) vs rounds-scaling (T=1), jail-beta1 vs BoN-beta0.
"""
import sys, os, glob, re, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RX = re.compile(r"_scale_r(\d+)t(\d+)_b(\d+)$")

def reduce_run(folder, R):
    if not os.path.exists(os.path.join(folder, f"round_{R}", "judgment.json")):
        return None
    pts = PA.extract(folder)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda p: p["score"]) for v in g.values()]
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best)

CELLS = [("qwen · self-harm", "qwen_selfharm"), ("llama · self-pres", "llama_selfpres")]
# beta_code, arm, marker, colour, label
SERIES = [
    ("10", "turns",  "o", "#dc2626", "jail β1 · scale turns (R=1)"),
    ("10", "rounds", "s", "#7c3aed", "jail β1 · scale rounds (T=1)"),
    ("0",  "turns",  "o", "#f59e0b", "BoN β0 · scale turns (R=1)"),
    ("0",  "rounds", "s", "#0891b2", "BoN β0 · scale rounds (T=1)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), dpi=140)
counts = {}
for col, (title, tag) in enumerate(CELLS):
    runs = {}
    for f in glob.glob(f"experiments/bloom/runs_init/{tag}_scale_r*t*_b*"):
        m = RX.search(os.path.basename(f))
        if not m:
            continue
        R, T, bc = int(m.group(1)), int(m.group(2)), m.group(3)
        runs[(bc, R, T)] = f
    counts[tag] = 0
    ax_s, ax_p = axes[0][col], axes[1][col]
    for bc, arm, mk, cl, lab in SERIES:
        pts = []
        for (b, R, T), f in runs.items():
            if b != bc:
                continue
            if arm == "turns" and R != 1:
                continue
            if arm == "rounds" and T != 1:
                continue
            red = reduce_run(f, R)
            if red:
                pts.append((R * T, red[1], red[0]))  # compute, score, plaus
                counts[tag] += 1
        pts.sort()
        if pts:
            ax_s.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=cl, lw=1.9, label=lab)
            ax_p.plot([p[0] for p in pts], [p[2] for p in pts], marker=mk, color=cl, lw=1.9, label=lab)
    for ax, yl, ylim in ((ax_s, "avg behaviour score (0–10)", (0, 10.5)), (ax_p, "target plausibility (%)", None)):
        ax.set_title(title + ("  —  elicitation" if ax is ax_s else "  —  plausibility"), fontsize=10.5)
        ax.set_xlabel("compute  ~  rounds × turns")
        ax.set_ylabel(yl)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, color="#e1e0d9", lw=0.5)
        ax.legend(fontsize=7.5, frameon=False)
fig.suptitle("Rounds vs turns compute-scaling  ·  25 scenarios  ·  best-of-pool avg", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.expanduser("~/scale_turns_rounds.png")
fig.savefig(out, facecolor="white")
print("saved", out, " run-points per cell:", counts)
