"""Plot the per-token target-probability histogram for 4 runs -> PNG.

Data: counts over log10(prob%) bins (edges -16..2 step 0.5), from score_token_hist.py.
Renders a grouped log-y bar histogram with the 1e-5 target-floor marked.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = [round(-16 + 0.5 * i, 2) for i in range(37)]
CENTERS = [(EDGES[i] + EDGES[i + 1]) / 2 for i in range(36)]

RUNS = {
    "corruption off · 1t (n=2014)": ([0]*27 + [1,0,3,2,7,18,31,118,1834], "#888780"),
    "floor 1e-5 · 1t (n=3067)":     ([0]*26 + [19,10,16,12,19,22,45,52,112,2760], "#378ADD"),
    "β=1 @1e-5 · 10rd (n=2972)":    ([0]*26 + [6,4,6,9,9,19,29,75,144,2671], "#1D9E75"),
    "β=5 @1e-5 · 10rd (n=5627)":    ([0]*25 + [1,57,59,51,52,71,74,101,101,142,4918], "#D85A30"),
}

LO, HI = -4.0, 2.0
mask = [i for i, c in enumerate(CENTERS) if LO <= c <= HI]
xc = np.array([CENTERS[i] for i in mask])

fig, ax = plt.subplots(figsize=(9, 5))
n = len(RUNS); bw = 0.45 / n
for k, (label, (counts, color)) in enumerate(RUNS.items()):
    y = np.array([counts[i] for i in mask], dtype=float)
    y[y == 0] = np.nan  # log y: hide empty bars
    ax.bar(xc + (k - (n - 1) / 2) * bw, y, width=bw, color=color, label=label,
           edgecolor="none", zorder=3)

ax.axvline(-3.0, color="#A32D2D", ls="--", lw=1.2, zorder=2)
ax.text(-3.0, 6500, "1e-5 target floor\n(=0.001%)", color="#A32D2D", fontsize=9,
        ha="center", va="bottom")

ax.set_yscale("log")
ax.set_ylim(0.7, 9000)
ax.set_xlim(LO - 0.3, HI + 0.3)
ax.set_xticks(range(int(LO), int(HI) + 1))
ax.set_xticklabels([f"$10^{{{p}}}$" for p in range(int(LO), int(HI) + 1)])
ax.set_xlabel("target probability of token  (%)")
ax.set_ylabel("token count  (log scale)")
ax.set_title("Per-token target probability — the least-token lives in the left tail")
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = __file__.rsplit("plot_token_hist.py", 1)[0] + "token_prob_histogram.png"
fig.savefig(out, dpi=160)
print("wrote", out)
