#!/usr/bin/env python3
"""All ablation arms on BOTH axes, vs the BoN baseline and the beta=1 reference.
Everything here is a JAIL-path run, so we use PA.extract throughout = ARITHMETIC mean
per-token probability (NOT comparable to corruption's stored geometric values).

Key test for jailonly: is it a genuine gain, or just a high-beta point on the same frontier?
We compute the floor-on jail family frontier and ask what score it achieves at jailonly's
plausibility. If the family matches/beats it, jailonly is not a better method."""
import sys, os, glob, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

R = "experiments/bloom/runs_init/"
CELLS = ["qwen_selfharm", "llama_selfpres"]
ARMS = [("BoN (beta=0)", "_pareto_b0"), ("beta=1 specific", "_arm_floorb10"),
        ("jailonly b1=0", "_arm_jailonly"), ("negative beta", "_arm_negb10"),
        ("unrelated(goblin)", "_arm_unrelgob"), ("generic-unsafe", "_arm_genbad"),
        ("unrel b2=2", "_arm_unrelgob20"), ("generic b2=2", "_arm_genbad20"),
        ("unrel b2=3", "_arm_unrelgob30"), ("generic b2=3", "_arm_genbad30")]

def bestpool(run):
    """Returns (plaus, score, n_rounds). n_rounds MATTERS: a run still in flight has fewer
    rounds and its numbers move a lot as it completes — reading partial runs produced two
    wrong conclusions on 2026-07-19. Always check n_rounds==5 before quoting."""
    pts = PA.extract(run)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    nr = len({p["round"] for p in pts})
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best), nr

def family_frontier(cell):
    pts = []
    for f in glob.glob(R + cell + "_arm_floorb*"):
        pts.extend(PA.extract(f))
    return PA.pareto_frontier(pts) if pts else []

for cell in CELLS:
    print(f"\n================ {cell} ================")
    print(f"  {'arm':20s} {'plaus%':>7s} {'score':>6s}   {'vs BoN':>7s}")
    base = bestpool(R + cell + "_pareto_b0")
    for name, tag in ARMS:
        run = R + cell + tag
        if not os.path.isdir(run):
            continue
        r = bestpool(run)
        if not r:
            print(f"  {name:20s}   (no data yet)")
            continue
        d = f"{r[1]-base[1]:+.2f}" if base else "-"
        warn = "" if r[2] == 5 else f"  <-- PARTIAL {r[2]}/5 rounds, do not quote"
        print(f"  {name:20s} {r[0]:7.2f} {r[1]:6.2f}   {d:>7s}  r{r[2]}{warn}")

    # is jailonly on the family frontier, or beyond it?
    fr = family_frontier(cell)
    jo = bestpool(R + cell + "_arm_jailonly")
    if fr and jo:
        at = max((p[1] for p in fr if p[0] >= jo[0]), default=None)
        print(f"\n  jailonly: plaus={jo[0]:.2f}% score={jo[1]:.2f}")
        if at is None:
            print(f"  -> family frontier never reaches plaus {jo[0]:.2f}%"
                  f" (family max plaus {max(p[0] for p in fr):.2f}%):"
                  f" jailonly operates BELOW the family's plausibility range.")
        else:
            verdict = ("jailonly is ON/BELOW the family frontier -> just a high-beta point"
                       if at >= jo[1] - 0.15 else
                       "jailonly BEATS the family frontier at matched plausibility -> genuine gain")
            print(f"  family frontier score at plaus>={jo[0]:.2f}%: {at:.2f}  ->  {verdict}")
