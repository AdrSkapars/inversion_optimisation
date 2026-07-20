#!/usr/bin/env python3
"""floor-on vs floor-off at matched beta / scenarios / seed.
Auto-pairs every *_arm_floorb<bc> run with its *_pareto_b<bc> counterpart.
Reports best-of-pool score, mean token-prob, and MIN token-prob (the floor's direct target)."""
import json, glob, os, re, statistics as st

R = "experiments/bloom/runs_init/"

def samples(run):
    out = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
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
                out.append((v, score[v], st.mean(tp), min(tp)))
    return out

def summarise(run):
    s = samples(run)
    if not s:
        return None
    g = {}
    for v, sc, mp, mn in s:
        g.setdefault(v, []).append((sc, mp, mn))
    best = [max(rows, key=lambda r: r[0]) for rows in g.values()]
    return {"score": st.mean(b[0] for b in best),
            "meanp": st.mean(b[1] for b in best),
            "minp":  st.mean(b[2] for b in best),
            "worst": min(x[3] for x in s),
            "n": len(g)}

def beta_of(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

pairs = []
for on in sorted(glob.glob(R + "*_arm_floorb*")):
    m = re.search(r"^(.*)_arm_floorb(\d+)$", os.path.basename(on))
    if not m:
        continue
    cell, bc = m.group(1), m.group(2)
    off = R + f"{cell}_pareto_b{bc}"
    if os.path.isdir(off):
        pairs.append((cell, beta_of(bc), on, off))
pairs.sort(key=lambda p: (p[0], p[1]))

hdr = f"{'cell':16s} {'beta':>5s} {'floor':5s} {'score':>6s} {'meanP%':>7s} {'minP%':>8s} {'worstP%':>11s}"
print(hdr); print("-" * len(hdr))
for cell, b, on, off in pairs:
    a, c = summarise(on), summarise(off)
    if not a or not c:
        print(f"{cell:16s} {b:5g}  incomplete (on={bool(a)} off={bool(c)})"); continue
    print(f"{cell:16s} {b:5g} {'ON':5s} {a['score']:6.2f} {a['meanp']:7.2f} {a['minp']:8.4f} {a['worst']:11.7f}")
    print(f"{'':16s} {'':5s} {'OFF':5s} {c['score']:6.2f} {c['meanp']:7.2f} {c['minp']:8.4f} {c['worst']:11.7f}")
    ratio = (a['worst'] / c['worst']) if c['worst'] > 0 else float('inf')
    print(f"{'':16s} {'':5s} {'D':5s} {a['score']-c['score']:+6.2f} {a['meanp']-c['meanp']:+7.2f}"
          f" {a['minp']-c['minp']:+8.4f}   worst x{ratio:,.0f}")
