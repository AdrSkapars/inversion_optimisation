"""Per-token target-probability histograms -> PNG, one panel per experiment.

Data: counts over log10(prob%) bins (edges -16..2 step 0.5) from score_token_hist.py.
Six panels share a fixed x-axis (1e-16% .. 1e2%) so tails are directly comparable.
Each panel: verbose experiment description, the "average elicitation rate" (mean
behaviour-presence 0-10), and average token probability (arithmetic + geometric +
median). The red dashed line marks the 1e-5 target floor (= 0.001%). The least-token
is the left edge of each tail.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = [round(-16 + 0.5 * i, 2) for i in range(37)]
CENTERS = np.array([(EDGES[i] + EDGES[i + 1]) / 2 for i in range(36)])

# label, counts[36], avg_elic_rate(0-10), arith%, geom%, median%, least%, color
RUNS = [
    ("Corruption OFF (vanilla)  ·  1 turn  ·  single round",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,2,7,18,31,118,1834],
     1.92, 84.66, 73.05, 99.52, 5.96e-3, "#888780"),
    ("Corruption OFF (vanilla)  ·  1 turn  ·  10 rounds, best-across",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,8,28,45,132,1979],
     3.76, 83.12, 70.67, 99.16, 1.76e-1, "#5F5E5A"),
    ("Corruption ON  ·  β=5  ·  NO floor  ·  1 turn  ·  single round   (the fingerprint)",
     [0,0,0,1,0,2,1,1,1,3,3,1,0,2,2,5,4,7,4,3,9,12,13,12,12,14,26,23,36,47,56,69,91,127,194,4540],
     8.64, 82.58, 37.21, 99.99, 5.18e-15, "#E24B4A"),
    ("Corruption ON  ·  β=5  ·  target-floor 1e-5  ·  1 turn  ·  single round",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,10,16,12,19,22,45,52,112,2760],
     3.04, 85.26, 61.17, 99.98, 1.13e-3, "#378ADD"),
    ("Corruption ON  ·  β=5  ·  target-floor 1e-5  ·  1 turn  ·  10 rounds, best-across",
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,26,19,19,33,31,37,45,70,112,3604],
     5.56, 86.69, 59.76, 100.00, 9.85e-4, "#BA7517"),
]

LO, HI = -16.0, 2.0
fig, axes = plt.subplots(len(RUNS), 1, figsize=(11, 13.5), sharex=True)
for ax, (label, counts, avg, arith, geo, med, least, color) in zip(axes, RUNS):
    y = np.array(counts, dtype=float)
    y[y == 0] = np.nan
    ax.bar(CENTERS, y, width=0.42, color=color, edgecolor="none", zorder=3)
    ax.axvline(-3.0, color="#A32D2D", ls="--", lw=1.0, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 9000)
    ax.set_xlim(LO - 0.4, HI + 0.4)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.set_ylabel("count")
    ax.text(0.011, 0.90, label, transform=ax.transAxes, fontsize=10.5, va="top")
    ax.text(0.011, 0.66, f"average elicitation rate {avg:.2f}   (behaviour-presence, 0–10)",
            transform=ax.transAxes, fontsize=9.5, color="#444441", va="top")
    ax.text(0.011, 0.45,
            f"average token probability  {arith:.0f}% arith.  ·  {geo:.0f}% geom.  ·  {med:.1f}% median",
            transform=ax.transAxes, fontsize=9.5, color="#444441", va="top")
    least_str = (f"{least:.0e}" if least < 1e-2 else f"{least:.2f}").replace("e-0", "e-")
    ax.text(0.011, 0.24, f"least token  {least_str}%",
            transform=ax.transAxes, fontsize=9.5, color="#444441", va="top")

axes[0].text(-3.0, 9700, "1e-5 target floor (0.001%)", color="#A32D2D", fontsize=8.5,
             ha="center", va="bottom")
axes[-1].set_xticks(range(int(LO), int(HI) + 1, 2))
axes[-1].set_xticklabels([f"$10^{{{p}}}$" for p in range(int(LO), int(HI) + 1, 2)])
axes[-1].set_xlabel("target probability of token  (%)   —   log scale")
fig.suptitle("Per-token target probability per experiment  (least-token = left edge of each tail)", y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.99])
out = __file__.rsplit("plot_token_hist.py", 1)[0] + "token_prob_histogram.png"
fig.savefig(out, dpi=160)
print("wrote", out)
