"""Two-temperature PoE sweep: target prob vs offensiveness as T_t cools.

Knob-validation run (best-of-20, NOT the 100-sample compute-matched comparison).
STRONG is my manual judgment of the best-pick texts (no regex).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_two_temp.png")

T_t    = [1.0, 0.7, 0.5, 0.3]
p_t    = [40.91, 52.44, 54.19, 55.22]
strong = [8, 4, 2, 1]                       # manual judgment of best-picks
x = list(range(len(T_t)))                    # even spacing, cooling left->right

fig, axL = plt.subplots(figsize=(8, 5))
axR = axL.twinx()

l1, = axL.plot(x, p_t, "o-", color="tab:blue", lw=2.4, ms=8, label="mean best-of-20 P_t")
l2, = axR.plot(x, strong, "s--", color="tab:red", lw=2.4, ms=8, label="STRONG (manual, /15)")

axL.set_xlabel("target temperature  T_t   (cooling →)")
axL.set_xticks(x); axL.set_xticklabels([f"{t:g}" for t in T_t])
axL.set_ylabel("mean best-of-20 P_t (%)", color="tab:blue")
axL.tick_params(axis="y", labelcolor="tab:blue")
axL.set_ylim(0, 100)
axR.set_ylabel("STRONG scenarios (out of 15)", color="tab:red")
axR.tick_params(axis="y", labelcolor="tab:red")
axR.set_ylim(0, 15)

for xi, p in zip(x, p_t):
    axL.annotate(f"{p:.1f}%", (xi, p), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=8, color="tab:blue")
for xi, s in zip(x, strong):
    axR.annotate(str(s), (xi, s), xytext=(0, -14), textcoords="offset points",
                 ha="center", fontsize=8, color="tab:red")

axL.legend(handles=[l1, l2], loc="center right")
axL.grid(alpha=0.3)
plt.title("Two-temperature PoE (β=5): cooling T_t trades offensiveness for P_t\n"
          "best-of-20 knob test — not compute-matched to the 100-sample baselines")
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print("saved", OUT)
