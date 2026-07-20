#!/usr/bin/env python3
"""Plot the jail-beta family of Pareto frontiers (one curve per beta) per cell,
plus the BoN baseline (beta=0, best-of-N by score) as a single dot.
Reads every completed *_pareto_b<bc> run folder, extracts its frontier via
pareto_analysis, colours by beta (blue=low -> red=high), saves a PNG.
Run from the repo root with the venv python.
"""
import sys, os, glob, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

def beta_of(folder):
    bc = os.path.basename(folder).rsplit("_b", 1)[-1]
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

def col(b, lo=0.25, hi=3.0):
    t = max(0.0, min(1.0, (b - lo) / (hi - lo)))
    return mc.hsv_to_rgb((0.66 * (1 - t), 0.70, 0.80))   # blue(low) -> red(high)

def bon_dot(folder):
    """BoN baseline operating point: per scenario take the best-of-N (max-score)
    sample from the no-jail pool, then average -> (mean plausibility, mean score)."""
    pts = PA.extract(folder)
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda p: p["score"]) for v in g.values()]
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best)

CELLS = [("qwen · self-harm", "qwen_selfharm_pareto_b*", 1.0),
         ("llama · self-pres", "llama_selfpres_pareto_b*", 1.0)]  # 3rd = jail-tune "optimal" beta (dashed)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=140)
for ax, (title, patt, win) in zip(axes, CELLS):
    folders = glob.glob("experiments/bloom/runs_init/" + patt)
    folders = [f for f in folders if os.path.exists(os.path.join(f, "round_5", "judgment.json"))]
    for f in sorted(folders, key=beta_of):
        b = beta_of(f)
        opt = abs(b - win) < 1e-9
        isbon = (b == 0)
        curve = PA.pareto_frontier(PA.extract(f))
        if isbon:   # BoN baseline: same post-hoc Pareto selection, distinct style
            style, colr, w, lab, z = "-.", "#111111", 2.4, "BoN β0 (no steering)", 5
        else:
            style, colr = ("--" if opt else "-"), col(b)
            w, lab, z = (2.7 if opt else 1.9), f"β{b:g}" + (" (opt)" if opt else ""), 3
        ax.plot([p[0] for p in curve], [p[1] for p in curve], linestyle=style,
                color=colr, lw=w, label=lab, zorder=z)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("target plausibility (%)  →  more plausible")
    ax.set_ylabel("behaviour score (0–10)")
    ax.set_ylim(0, 10.5)
    ax.grid(True, color="#e1e0d9", lw=0.5)
    ax.legend(fontsize=8, frameon=False, ncol=2, title="jail β")
fig.suptitle("Pareto frontier vs jail β  ·  5 rounds · 3 turns · 25 scenarios  ·  min–max", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.expanduser("~/beta_family.png")
fig.savefig(out, facecolor="white")
print("saved", out, "  cells:", [(t, len([f for f in glob.glob('experiments/bloom/runs_init/'+p) if os.path.exists(os.path.join(f,'round_5','judgment.json'))])) for t, p, w in CELLS])
