#!/usr/bin/env python3
"""Pareto frontier + optimal-point analysis for plausibility vs elicitation.

STANDALONE analysis/plot tool — NOT part of the pipeline. It only reads a run's
output files (judgment.json for behaviour score, transcript gen_token_probs for
plausibility); it never imports the pipeline.

A POINT = one transcript = (scenario, prob, score):
  prob  = mean token-probability the UNMODIFIED target gives its own tokens (plausibility %)
  score = judge behaviour_presence (0-10)                                   (elicitation)
Per scenario there is a pool of samples (rounds or resamples). The aggregate Pareto
frontier is traced by weighted-sum selection: for a grid of weights w, pick per scenario
the sample maximising w*score_norm + (1-w)*prob_norm, average over scenarios; sweeping w
gives the frontier. Then find the single operating point (closest-to-utopia / knee / TOPSIS).

Usage:
  python3 pareto_analysis.py <run_dir | points.json> [--norm minmax|zscore] [--plot out.png]
"""
import json, sys, os, glob, math, statistics as st
from collections import defaultdict


def extract(run_dir):
    """Read (scenario, prob, score) points from a BLOOM run folder (all rounds)."""
    pts = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        j = json.load(open(jp))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            d = json.load(open(tf)); v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in score:
                continue
            tp = [float(p) for m in d["messages"] if m.get("source") == "target"
                  for p in (m.get("gen_token_probs") or [])]
            if tp:
                pts.append({"scenario": v, "round": rnd, "prob": st.mean(tp), "score": score[v]})
    return pts


def _norm(vals, method):
    if method == "minmax":
        lo, hi = min(vals), max(vals); rng = (hi - lo) or 1.0
        return [(x - lo) / rng for x in vals]
    m = st.mean(vals); s = st.pstdev(vals) or 1.0
    return [(x - m) / s for x in vals]


def pareto_frontier(points, norm="minmax", wsteps=21):
    """Aggregate frontier via weighted-sum selection swept over weights. Returns [(prob, score)]."""
    pn = _norm([p["prob"] for p in points], norm); sn = _norm([p["score"] for p in points], norm)
    for p, a, b in zip(points, pn, sn):
        p["_pn"], p["_sn"] = a, b
    g = defaultdict(list)
    for p in points:
        g[p["scenario"]].append(p)
    curve = []
    for i in range(wsteps):
        w = i / (wsteps - 1)
        chosen = [max(pool, key=lambda p: w * p["_sn"] + (1 - w) * p["_pn"]) for pool in g.values() if pool]
        curve.append((round(st.mean(c["prob"] for c in chosen), 3), round(st.mean(c["score"] for c in chosen), 3)))
    dedup = []
    for p in curve:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return dedup


def optimal_points(curve):
    """Single operating points on a frontier curve [(prob, score)] (both maximised)."""
    C = curve
    xs = [p[0] for p in C]; ys = [p[1] for p in C]
    pmn, pmx, smn, smx = min(xs), max(xs), min(ys), max(ys)
    prg = (pmx - pmn) or 1.0; srg = (smx - smn) or 1.0
    nrm = lambda p: ((p[0] - pmn) / prg, (p[1] - smn) / srg)
    # closest-to-utopia: min normalised Euclidean distance to the ideal corner (1, 1)
    utopia = min(C, key=lambda p: math.hypot(1 - nrm(p)[0], 1 - nrm(p)[1]))
    # knee: max perpendicular distance from the chord joining the two endpoints
    A = max(C, key=lambda p: p[0]); B = min(C, key=lambda p: p[0])
    dx, dy = B[0] - A[0], B[1] - A[1]; L = math.hypot(dx, dy) or 1.0
    knee = max(C, key=lambda p: abs((p[0] - A[0]) * dy - (p[1] - A[1]) * dx) / L)
    # TOPSIS: max d(nadir) / [d(nadir) + d(ideal)]
    def topsis(p):
        n = nrm(p); di = math.hypot(1 - n[0], 1 - n[1]); dn = math.hypot(*n)
        return dn / (di + dn) if (di + dn) else 0.0
    top = max(C, key=topsis)
    return {"utopia": list(utopia), "knee": list(knee), "topsis": list(top)}


def plot(curve, opt, title="", out="pareto.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [p[0] for p in curve]; ys = [p[1] for p in curve]
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=130)
    ax.plot(xs, ys, "-", color="#ea580c", lw=2, label="Pareto frontier")
    u, k, t = opt["utopia"], opt["knee"], opt["topsis"]
    ax.scatter(*t, s=260, facecolors="none", edgecolors="#111", lw=2, label="TOPSIS", zorder=3)
    ax.scatter(*k, s=170, marker="*", color="#dc2626", label="knee", zorder=4)
    ax.scatter(*u, s=70, marker="D", color="#7c3aed", label="closest-to-utopia", zorder=5)
    ax.set_xlabel("target plausibility (%)  →  more plausible")
    ax.set_ylabel("behaviour score (0–10)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)
    return out


def main():
    a = sys.argv[1:]
    norm = a[a.index("--norm") + 1] if "--norm" in a else "minmax"
    out = a[a.index("--plot") + 1] if "--plot" in a else None
    src = a[0]
    pts = json.load(open(src)) if src.endswith(".json") and os.path.exists(src) else extract(src)
    curve = pareto_frontier(pts, norm=norm)
    opt = optimal_points(curve)
    print(f"points={len(pts)} scenarios={len(set(p['scenario'] for p in pts))} norm={norm}")
    print("pareto frontier (plaus%, score):")
    print("  " + "  ".join(f"{x:.1f}/{y:.2f}" for x, y in curve))
    print("optimal points (plaus%, score):")
    for k in ("utopia", "knee", "topsis"):
        print(f"  {k:8} {opt[k][0]:.1f}% / {opt[k][1]:.2f}")
    if out:
        try:
            print("saved plot ->", plot(curve, opt, title=os.path.basename(src), out=out))
        except ModuleNotFoundError:
            print("(matplotlib not installed in this interpreter — `pip install matplotlib`, or use a venv that has it; frontier data is printed above)")


if __name__ == "__main__":
    main()
