#!/usr/bin/env python3
"""Error-bar version of the turns-vs-rounds plot for jail beta1, aggregating over
sampling seeds. Seed 1 = canonical unsuffixed folders; seeds 2,3 = _s2/_s3 folders.
Per (R,T) point: mean +/- std of best-of-pool avg score (and plausibility) across seeds.
Separate output ~/scale_seeds.png; does NOT touch scale_plot.py / scale_turns_rounds.png.
"""
import sys, os, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "experiments/bloom/runs_init"

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

def seed_folders(tag, R, T):
    base = f"{ROOT}/{tag}_scale_r{R}t{T}_b10"
    return [base, base + "_s2", base + "_s3"]

def agg(tag, R, T):
    """mean/std of (plaus, score) over available seeds for one (R,T) cell."""
    vals = [reduce_run(f, R) for f in seed_folders(tag, R, T)]
    vals = [v for v in vals if v]
    if not vals:
        return None
    ps = [v[0] for v in vals]; ss = [v[1] for v in vals]
    sd = lambda x: st.pstdev(x) if len(x) > 1 else 0.0
    return (st.mean(ps), sd(ps), st.mean(ss), sd(ss), len(vals))

CELLS = [("qwen · self-harm", "qwen_selfharm"), ("llama · self-pres", "llama_selfpres")]
# arm, marker, colour, label, (R,T) generator over compute c in 1..6
ARMS = [
    ("turns",  "o", "#dc2626", "jail β1 · scale turns (R=1)", lambda c: (1, c)),
    ("rounds", "s", "#7c3aed", "jail β1 · scale rounds (T=1)", lambda c: (c, 1)),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), dpi=140)
maxn = 0
for col, (title, tag) in enumerate(CELLS):
    ax_s, ax_p = axes[0][col], axes[1][col]
    for arm, mk, cl, lab, rt in ARMS:
        xs, sc_m, sc_e, pl_m, pl_e = [], [], [], [], []
        for c in range(1, 7):
            R, T = rt(c)
            a = agg(tag, R, T)
            if not a:
                continue
            pm, pe, sm, se, n = a
            maxn = max(maxn, n)
            xs.append(c); sc_m.append(sm); sc_e.append(se); pl_m.append(pm); pl_e.append(pe)
        if xs:
            ax_s.errorbar(xs, sc_m, yerr=sc_e, marker=mk, color=cl, lw=1.9, capsize=3, label=lab)
            ax_p.errorbar(xs, pl_m, yerr=pl_e, marker=mk, color=cl, lw=1.9, capsize=3, label=lab)
    ax_s.set_title(title + "  —  elicitation", fontsize=10.5)
    ax_s.set_ylabel("avg behaviour score (0–10)"); ax_s.set_ylim(0, 10.5)
    ax_p.set_title(title + "  —  plausibility", fontsize=10.5)
    ax_p.set_ylabel("target plausibility (%)")
    for ax in (ax_s, ax_p):
        ax.set_xlabel("compute  ~  rounds × turns")
        ax.grid(True, color="#e1e0d9", lw=0.5)
        ax.legend(fontsize=8, frameon=False)
fig.suptitle(f"Rounds vs turns compute-scaling  ·  jail β1  ·  mean ± std over up to {maxn} seeds", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.expanduser("~/scale_seeds.png")
fig.savefig(out, facecolor="white")
print("saved", out, " max seeds/point:", maxn)
