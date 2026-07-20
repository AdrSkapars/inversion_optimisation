#!/usr/bin/env python3
"""Method-level Pareto comparison: jail (floor-on beta family) vs corruption (b2 sweep).

NOTE the two paths persist plausibility differently:
  jail       -> transcripts/*.json messages[].gen_token_probs   (per token)
  corruption -> corruption_pool.json pool[].target_p_pct        (mean per turn, per candidate)
so corruption needs its own extractor. Corruption gives mean target-prob only (no per-token
list), hence no min-token-prob for corruption here.

Per method we pool samples across its hyperparameter setting and take the Pareto frontier
(per-scenario selection swept over weights) = the method's achievable frontier."""
import sys, os, glob, json, math, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
import pareto_analysis as PA

R = "experiments/bloom/runs_init/"
CELLS = ["qwen_selfharm", "llama_selfpres"]

def extract_jail(run):
    """Like PA.extract but reports the GEOMETRIC mean per-token probability, exp(mean log p),
    because that is what corruption persists (target_p_pct = exp(mean_lp)). PA.extract uses the
    ARITHMETIC mean, which is systematically higher and NOT comparable to the corruption side."""
    pts = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        j = json.load(open(jp))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            d = json.load(open(tf))
            v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in score:
                continue
            tp = [float(p) for m in d["messages"] if m.get("source") == "target"
                  for p in (m.get("gen_token_probs") or [])]
            if tp:
                geo = math.exp(st.mean(math.log(max(p, 1e-12) / 100.0) for p in tp)) * 100.0
                pts.append({"scenario": v, "round": rnd, "prob": geo, "score": score[v]})
    return pts

def extract_corr(run):
    """[{scenario, round, prob, score}] for a corruption run, mirroring PA.extract."""
    pts = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rd = os.path.dirname(jp)
        rnd = int(os.path.basename(rd).split("_")[1])
        j = json.load(open(jp))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        cp = os.path.join(rd, "corruption_pool.json")
        if not os.path.exists(cp):
            continue
        pools = json.load(open(cp)).get("corruption_pools", [])
        # text emitted per (variation, turn), to pick the selected candidate when pool>1
        emitted = {}
        for tf in glob.glob(os.path.join(rd, "transcripts", "*.json")):
            d = json.load(open(tf))
            v = d.get("metadata", {}).get("variation_number")
            if v is None:
                continue
            k = 0
            for m in d.get("messages", []):
                if m.get("source") == "target":
                    k += 1
                    emitted[(v, k)] = (m.get("content") or "").strip()
        byvar = {}
        for entry in pools:
            v, turn = entry.get("variation_index"), entry.get("turn")
            pool = entry.get("pool") or []
            if v is None or not pool:
                continue
            want = emitted.get((v, turn))
            cand = None
            if want:
                for c in pool:
                    if (c.get("text") or "").strip() == want:
                        cand = c; break
            if cand is None:                      # fall back: the run's own selection rule
                ok = [c for c in pool if (c.get("d3") or 0) >= 0.8] or pool
                cand = max(ok, key=lambda c: c.get("target_lp", float("-inf")))
            p = cand.get("target_p_pct")
            if p is not None:
                byvar.setdefault(v, []).append(float(p))
        for v, ps in byvar.items():
            if v in score and ps:
                pts.append({"scenario": v, "round": rnd, "prob": st.mean(ps), "score": score[v]})
    return pts

def frontier(runs, extractor):
    pts = []
    for r in runs:
        pts.extend(extractor(r))
    return PA.pareto_frontier(pts) if pts else []

def score_at(curve, t):
    return max((p[1] for p in curve if p[0] >= t), default=None)

def runs_for(cell, pat, exclude_p10=True):
    rs = sorted(glob.glob(R + cell + pat + "*"))
    return [r for r in rs if (not exclude_p10 or "p10" not in r)
            and os.path.exists(os.path.join(r, "round_5", "judgment.json"))]

for cell in CELLS:
    print(f"\n================ {cell} ================")
    j = frontier(runs_for(cell, "_arm_floorb"), extract_jail)
    c = frontier(runs_for(cell, "_arm_corrb"), extract_corr)
    for nm, cur, rs in (("jail(floor)", j, runs_for(cell, "_arm_floorb")),
                        ("corruption", c, runs_for(cell, "_arm_corrb"))):
        if cur:
            print(f"{nm:12s} ({len(rs)} runs) plaus {min(p[0] for p in cur):.1f}-{max(p[0] for p in cur):.1f}%"
                  f"  score {min(p[1] for p in cur):.2f}-{max(p[1] for p in cur):.2f}")
            print("   " + "  ".join(f"{p[0]:.1f}/{p[1]:.2f}" for p in cur))
        else:
            print(f"{nm:12s} EMPTY ({len(rs)} runs found)")
    if j and c:
        lo = max(min(p[0] for p in j), min(p[0] for p in c))
        hi = min(max(p[0] for p in j), max(p[0] for p in c))
        if hi > lo:
            print(f"\n  matched-plausibility (overlap {lo:.1f}-{hi:.1f}%):")
            print(f"  {'plaus>=':>8s} {'jail':>7s} {'corrupt':>8s}  winner")
            for i in range(7):
                t = lo + (hi - lo) * i / 6
                a, b = score_at(j, t), score_at(c, t)
                if a is None or b is None:
                    continue
                w = "jail" if a > b + 0.15 else ("corruption" if b > a + 0.15 else "tie")
                print(f"  {t:8.1f} {a:7.2f} {b:8.2f}  {w}")
        else:
            print(f"\n  NO PLAUSIBILITY OVERLAP: jail {min(p[0] for p in j):.1f}-{max(p[0] for p in j):.1f}%"
                  f" vs corruption {min(p[0] for p in c):.1f}-{max(p[0] for p in c):.1f}%")

print("\n\n========= p10 pool vs single-sample (b2=6) — what 10x compute buys =========")
def summarise(run, extractor):
    pts = extractor(run)
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    return st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best)
for cell in CELLS:
    for tag in ("_arm_corrb6", "_arm_corrp10b6"):
        r = R + cell + tag
        s = summarise(r, extract_corr) if os.path.isdir(r) else None
        print(f"  {cell:16s} {tag:18s} " + (f"plaus={s[0]:6.2f}%  score={s[1]:5.2f}" if s else "(not finished)"))
