#!/usr/bin/env python3
"""Absolute best-of-pool avg SCORE / avg PROB per cell: floor-ON vs floor-OFF vs jail-off BoN.
Best-of-pool = per scenario take the max-score round, then average score AND that sample's
mean token-probability (arithmetic, jail-path). Marks the beta chosen by the original
jail-tune sweep (experiments/bloom/jail_tune_best_betas.json)."""
import json, glob, os, re, statistics as st

R = "experiments/bloom/runs_init/"

def bop(run):
    rows = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rd = os.path.dirname(jp)
        rnd = int(os.path.basename(rd).split("_")[1])
        j = json.load(open(jp))
        sc = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
              if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(rd, "transcripts", "*.json")):
            d = json.load(open(tf))
            v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in sc:
                continue
            tp = [float(p) for m in d["messages"] if m.get("source") == "target"
                  for p in (m.get("gen_token_probs") or [])]
            if tp:
                rows.append((rnd, v, float(sc[v]), st.mean(tp)))
    if not rows:
        return None
    by = {}
    for _, v, s, p in rows:
        by.setdefault(v, []).append((s, p))
    best = [max(v, key=lambda x: x[0]) for v in by.values()]
    return (st.mean(b[0] for b in best), st.mean(b[1] for b in best), len({r[0] for r in rows}))

def beta_of(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

tune = json.load(open("experiments/bloom/jail_tune_best_betas.json"))
CELLS = [("qwen_selfharm", "qwen", "selfharm"), ("llama_selfpres", "llama", "selfpres")]

def fmt(t):
    return f"{t[0]:5.2f} / {t[1]:5.1f}%" if t else f"{'--':>13s}"

for cell, mk, bk in CELLS:
    opt = (tune.get(mk, {}).get(bk) or {}).get("beta")
    b0 = bop(R + cell + "_pareto_b0")
    print(f"\n===== {cell} =====   jail-tune optimal beta = {opt}"
          f"   (tuned on: score {tune.get(mk,{}).get(bk,{}).get('score')}, "
          f"prob {tune.get(mk,{}).get(bk,{}).get('prob')}%)")
    print(f"  {'beta':>5s}  {'floor ON':>14s}  {'floor OFF':>14s}  {'BoN (jail off)':>14s}")
    betas = []
    for on in glob.glob(R + cell + "_arm_floorb*"):
        m = re.search(r"_arm_floorb(\d+)$", os.path.basename(on))
        if m:
            betas.append((beta_of(m.group(1)), m.group(1)))
    for b, bc in sorted(betas):
        a = bop(R + f"{cell}_arm_floorb{bc}")
        o = bop(R + f"{cell}_pareto_b{bc}")
        if not a or a[2] != 5:
            continue
        if o and o[2] != 5:
            o = None
        star = "  <== jail-tune optimal" if (opt is not None and abs(b - float(opt)) < 1e-9) else ""
        print(f"  {b:5g}  {fmt(a):>14s}  {fmt(o):>14s}  {fmt(b0):>14s}{star}")
