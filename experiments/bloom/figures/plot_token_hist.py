"""Per-token target-probability histograms -> PNG, one panel per run.

Data: counts over log10(prob%) bins (edges -16..2 step 0.5), from score_token_hist.py.
Five panels share a fixed x-axis (1e-16% .. 1e2%) so tails are directly comparable.
Each panel labels the run's avg behavior-presence and elicitation rate.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = [round(-16 + 0.5 * i, 2) for i in range(37)]
CENTERS = np.array([(EDGES[i] + EDGES[i + 1]) / 2 for i in range(36)])

# (label, counts[36], avg, elic, color, mean_prob_pct, median_prob_pct)
RUNS = [
    ("corruption off · 1t",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,2,7,18,31,118,1834],
     1.92, 0.04, "#888780", 84.66, 99.52),
    ("corruption ON, NO floor · 1t  (the fingerprint)",
     [0,0,0,1,0,2,1,1,1,3,3,1,0,2,2,5,4,7,4,3,9,12,13,12,12,14,26,23,36,47,56,69,91,127,194,4540],
     8.64, 0.84, "#E24B4A", 82.58, 99.99),
    ("floor 1e-5 · 1t",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,10,16,12,19,22,45,52,112,2760],
     3.04, 0.16, "#378ADD", 85.26, 99.98),
    ("β=1 @1e-5 · 10rd best-across",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,4,6,9,9,19,29,75,144,2671],
     7.12, 0.88, "#1D9E75", 84.81, 99.93),
    ("β=5 @1e-5 · 10rd best-across",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,57,59,51,52,71,74,101,101,142,4918],
     8.48, 1.00, "#BA7517", 85.21, 100.00),
]

LO, HI = -16.0, 2.0
import math
fig, axes = plt.subplots(len(RUNS), 1, figsize=(10, 11), sharex=True)
for ax, (label, counts, avg, elic, color, tmean, tmed) in zip(axes, RUNS):
    y = np.array(counts, dtype=float)
    y[y == 0] = np.nan
    ax.bar(CENTERS, y, width=0.42, color=color, edgecolor="none", zorder=3)
    ax.axvline(-3.0, color="#A32D2D", ls="--", lw=1.0, zorder=2)  # 1e-5 target floor = 0.001%
    ax.axvline(math.log10(tmean), color="#26215C", ls="-", lw=1.4, zorder=4,
               label=f"mean {tmean:.0f}%")
    ax.axvline(math.log10(tmed), color="#26215C", ls=":", lw=1.6, zorder=4,
               label=f"median {tmed:.1f}%")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.30, 1.02),
              handlelength=1.6, ncol=1)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 9000)
    ax.set_xlim(LO - 0.4, HI + 0.4)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.text(0.012, 0.86, label, transform=ax.transAxes, fontsize=11, va="top")
    ax.text(0.012, 0.60, f"avg {avg:.2f}   ·   elic {elic:.2f}", transform=ax.transAxes,
            fontsize=10, color="#5F5E5A", va="top")
    ax.set_ylabel("count")

axes[1].text(-9, 200, "tail reaches 1e-15%", color="#A32D2D", fontsize=9, ha="center")
axes[0].text(-3.0, 9500, "1e-5 floor (0.001%)", color="#A32D2D", fontsize=8.5, ha="center", va="bottom")
axes[-1].set_xticks(range(int(LO), int(HI) + 1, 2))
axes[-1].set_xticklabels([f"$10^{{{p}}}$" for p in range(int(LO), int(HI) + 1, 2)])
axes[-1].set_xlabel("target probability of token  (%)   —   log scale")
fig.suptitle("Per-token target probability per run  (least-token = left edge of each tail)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = __file__.rsplit("plot_token_hist.py", 1)[0] + "token_prob_histogram.png"
fig.savefig(out, dpi=160)
print("wrote", out)
