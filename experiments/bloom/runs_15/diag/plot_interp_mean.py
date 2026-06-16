"""Average the interpolation sweep across all scenarios into one 3-line plot."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {0}  # scenario 0 is degenerate (benign refusal, corrupt parrots body)

betas = None
T, C, P = [], [], []
for f in sorted(glob.glob(os.path.join(HERE, "diag_interp_sweep_s*.json"))):
    idx = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
    if idx in EXCLUDE:
        continue
    rows = json.load(open(f, encoding="utf-8"))["rows"]
    b = [r["beta"] for r in rows]
    if betas is None:
        betas = b
    if b != betas:
        continue
    T.append([r["p_target"] for r in rows])
    C.append([r["p_corrupt"] for r in rows])
    P.append([r["p_poe"] for r in rows])

betas = np.array(betas)
T, C, P = np.array(T), np.array(C), np.array(P)
n = T.shape[0]

def mb(a):
    return a.mean(0), a.std(0)

(tm, ts), (cm, cs), (pm, ps) = mb(T), mb(C), mb(P)

plt.figure(figsize=(8, 5.2))
for m, s, col, lab, mk in [
    (tm, ts, "tab:blue",   "target-only  softmax(t)",          "o"),
    (cm, cs, "tab:red",    "corrupt-only  softmax(c)",         "s"),
    (pm, ps, "tab:purple", "equal PoE  softmax(0.5t+0.5c)",    "^"),
]:
    plt.plot(betas, m, mk + "-", color=col, label=lab, lw=2, ms=5)
    plt.fill_between(betas, m - s, m + s, color=col, alpha=0.13)

plt.xlabel("interpolation β   (0 = pure target,  1 = pure corrupt)")
plt.ylabel("geom-mean per-token probability of sample (%)")
plt.title(f"Interpolated sampling — mean over {n} scenarios (±1 std)")
plt.ylim(0, 100)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(HERE, "diag_interp_sweep_mean.png")
plt.savefig(out, dpi=140)
print("saved", out, "| n =", n)
