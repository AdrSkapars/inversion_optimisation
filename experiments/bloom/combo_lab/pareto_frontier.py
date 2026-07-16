#!/usr/bin/env python3
"""Map the plausibility<->elicitation Pareto frontier from a BLOOM multi-round run.

Data model: each POINT is one transcript = (scenario, round, prob, score).
  prob  = mean token-probability the UNMODIFIED target gives its own generated tokens
          (pooled over all target turns of the conversation) -- the plausibility axis.
  score = judge behaviour_presence (0-10) for that conversation -- the elicitation axis.
Pool per scenario = its samples across rounds (here 6). A "technique" selects ONE sample
per scenario; we average (prob, score) over scenarios; sweeping the technique's parameter
traces that technique's aggregate frontier.

Techniques:
  prob_rank    pick k-th most-probable per scenario         (k=1 == target_pick ... k=max == minplaus)
  elic_rank    pick k-th highest-score per scenario          (k=1 == oracle)             [needs judge]
  weighted_sum argmax  w*score_n + (1-w)*prob_n              (convex hull only)
  utopia_l2    argmin  Euclidean dist to ideal (weighted)     (compromise programming, p=2)
  tchebycheff  argmin  max weighted dev to ideal              (weighted L-inf; reaches non-convex)
  epsilon      argmax score s.t. prob >= eps                  (epsilon-constraint sweep)
Normalisation of the two axes (for the scalarising techniques): 'minmax' or 'zscore', global.

Usage:
  python3 pareto_frontier.py <run_dir|points.json> [--norm minmax|zscore] [--tag NAME]
Writes ~/points_<tag>.json (raw points) and ~/frontiers_<tag>_<norm>.json (all technique curves).
"""
import json, sys, os, glob, math, statistics as st
from collections import defaultdict

def extract(run_dir):
    pts = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        j = json.load(open(jp))
        score = {e["variation_number"]: e["behavior_presence"]
                 for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            d = json.load(open(tf)); v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in score: continue
            tp = [float(p) for m in d["messages"] if m.get("source") == "target"
                  for p in (m.get("gen_token_probs") or [])]
            if not tp: continue
            pts.append({"scenario": v, "round": rnd, "prob": st.mean(tp), "score": score[v]})
    return pts

def _norm(vals, method):
    if method == "minmax":
        lo, hi = min(vals), max(vals); rng = (hi - lo) or 1.0
        return [(x - lo) / rng for x in vals]
    m = st.mean(vals); s = st.pstdev(vals) or 1.0
    return [(x - m) / s for x in vals]

def by_scenario(pts):
    g = defaultdict(list)
    for p in pts: g[p["scenario"]].append(p)
    return g

def _agg(sel, g):
    ch = [sel(pool) for pool in g.values() if pool]
    return {"prob": round(st.mean(c["prob"] for c in ch), 2),
            "score": round(st.mean(c["score"] for c in ch), 3), "n": len(ch)}

def frontiers(pts, norm="minmax", wsteps=21, rho=0.05):
    pn = _norm([p["prob"] for p in pts], norm); sn = _norm([p["score"] for p in pts], norm)
    for p, a, b in zip(pts, pn, sn): p["pn"], p["sn"] = a, b
    g = by_scenario(pts); maxpool = max(len(v) for v in g.values())
    id_s, id_p = max(sn), max(pn)                          # ideal (utopia) point in normalised space
    out = {"prob_rank": [], "elic_rank": [], "weighted_sum": [], "utopia_l2": [], "tchebycheff": [], "aug_tchebycheff": [], "epsilon": []}
    for k in range(1, maxpool + 1):
        out["prob_rank"].append({"param": k, **_agg(lambda pool, k=k: sorted(pool, key=lambda p: p["prob"], reverse=True)[min(k-1, len(pool)-1)], g=g)})
        out["elic_rank"].append({"param": k, **_agg(lambda pool, k=k: sorted(pool, key=lambda p: p["score"], reverse=True)[min(k-1, len(pool)-1)], g=g)})
    for i in range(wsteps):
        w = i / (wsteps - 1)                                # weight on SCORE (elicitation)
        out["weighted_sum"].append({"param": round(w, 3), **_agg(lambda pool, w=w: max(pool, key=lambda p: w*p["sn"] + (1-w)*p["pn"]), g=g)})
        out["utopia_l2"].append({"param": round(w, 3), **_agg(lambda pool, w=w: min(pool, key=lambda p: math.hypot(w*(id_s-p["sn"]), (1-w)*(id_p-p["pn"]))), g=g)})
        out["tchebycheff"].append({"param": round(w, 3), **_agg(lambda pool, w=w: min(pool, key=lambda p: max(w*(id_s-p["sn"]), (1-w)*(id_p-p["pn"]))), g=g)})
        out["aug_tchebycheff"].append({"param": round(w, 3), **_agg(lambda pool, w=w: min(pool, key=lambda p: max(w*(id_s-p["sn"]), (1-w)*(id_p-p["pn"])) + rho*((id_s-p["sn"]) + (id_p-p["pn"]))), g=g)})
    def _eps_sel(pool, eps):                                # max score s.t. prob>=eps; else closest-to-feasible (max prob)
        feas = [p for p in pool if p["prob"] >= eps]
        return max(feas, key=lambda p: p["score"]) if feas else max(pool, key=lambda p: p["prob"])
    plo, phi = min(p["prob"] for p in pts), max(p["prob"] for p in pts)
    for i in range(wsteps):
        eps = plo + (phi - plo) * i / (wsteps - 1)
        out["epsilon"].append({"param": round(eps, 2), **_agg(lambda pool, eps=eps: _eps_sel(pool, eps), g=g)})
    return out

def main():
    a = sys.argv[1:]
    norm = a[a.index("--norm") + 1] if "--norm" in a else "minmax"
    tag = a[a.index("--tag") + 1] if "--tag" in a else "run"
    rho = float(a[a.index("--rho") + 1]) if "--rho" in a else 0.05
    src = a[0]
    pts = json.load(open(src)) if src.endswith(".json") and os.path.exists(src) else extract(src)
    json.dump(pts, open(os.path.expanduser(f"~/points_{tag}.json"), "w"))
    fr = frontiers(pts, norm=norm, rho=rho)
    outp = os.path.expanduser(f"~/frontiers_{tag}_{norm}.json")
    json.dump({"tag": tag, "norm": norm, "n_points": len(pts),
               "n_scenarios": len(by_scenario(pts)), "frontiers": fr}, open(outp, "w"), indent=1)
    print(f"points={len(pts)} scenarios={len(by_scenario(pts))} norm={norm} -> {outp}")
    for name, curve in fr.items():
        print(f"  {name:13} " + "  ".join(f"{d['prob']:.0f}/{d['score']:.1f}" for d in curve))

if __name__ == "__main__":
    main()
