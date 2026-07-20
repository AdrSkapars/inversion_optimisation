#!/usr/bin/env python3
"""Floor-ON jail beta family: Pareto frontier per beta, with the BoN (beta=0) baseline curve.
Same construction as beta_family_plot.py but globs the *_arm_floorb<bc> runs (naturalness
floor = 1e-4 active). Output ~/beta_family_floor.png
"""
import sys, os, glob, re
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

R = "experiments/bloom/runs_init/"

def beta_of_bc(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

def col(b, lo=0.25, hi=3.0):
    t = max(0.0, min(1.0, (b - lo) / (hi - lo)))
    return mc.hsv_to_rgb((0.66 * (1 - t), 0.70, 0.80))   # blue(low) -> red(high)

CELLS = [("qwen · self-harm", "qwen_selfharm", 1.0),
         ("llama · self-pres", "llama_selfpres", 1.0)]   # 3rd = jail-tune "optimal" beta (dashed)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=140)
counts = []
for ax, (title, tag, win) in zip(axes, CELLS):
    runs = []
    for f in glob.glob(R + tag + "_arm_floorb*"):
        m = re.search(r"_arm_floorb(\d+)$", os.path.basename(f))
        if not m:
            continue
        if not os.path.exists(os.path.join(f, "round_5", "judgment.json")):
            continue
        runs.append((beta_of_bc(m.group(1)), f))
    runs.sort()
    counts.append((title, len(runs)))
    for b, f in runs:
        opt = abs(b - win) < 1e-9
        curve = PA.pareto_frontier(PA.extract(f))
        ax.plot([p[0] for p in curve], [p[1] for p in curve], linestyle="--" if opt else "-",
                color=col(b), lw=2.7 if opt else 1.9,
                label=f"β{b:g}" + (" (opt)" if opt else ""), zorder=3)
    # BoN baseline: beta=0 -> z = target, so the floor can never bind; the no-floor run is valid
    bon = R + tag + "_pareto_b0"
    if os.path.exists(os.path.join(bon, "round_5", "judgment.json")):
        curve = PA.pareto_frontier(PA.extract(bon))
        ax.plot([p[0] for p in curve], [p[1] for p in curve], linestyle="-.",
                color="#111111", lw=2.4, label="BoN β0 (no steering)", zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("target plausibility (%)  →  more plausible")
    ax.set_ylabel("behaviour score (0–10)")
    ax.set_ylim(0, 10.5)
    ax.grid(True, color="#e1e0d9", lw=0.5)
    ax.legend(fontsize=8, frameon=False, ncol=2, title="jail β")
fig.suptitle("Pareto frontier vs jail β  ·  naturalness floor ON (1e-4)  ·  5 rounds · 3 turns · 25 scenarios",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.expanduser("~/beta_family_floor.png")
fig.savefig(out, facecolor="white")
print("saved", out, " curves:", counts)
