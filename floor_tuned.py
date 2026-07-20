#!/usr/bin/env python3
"""Floor ON vs OFF at each cell's TUNED beta, all models x behaviours.
Pairs <cell>_arm_tfon with <cell>_arm_tfoff (scenario-matched, same seed/config).
Best-of-pool: per scenario take the max-score round, then average score AND that sample's
mean token-prob. Requires 3/3 rounds on BOTH arms before a cell is quoted."""
import json, glob, os, statistics as st

R = "experiments/bloom/runs_init/"
NOISE = 1.4          # measured same-config score spread (7.72 vs 6.32 on an identical run)

def bop(run):
    rows = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rd = os.path.dirname(jp)
        rnd = int(os.path.basename(rd).split("_")[1])
        sc = {e["variation_number"]: e["behavior_presence"]
              for e in json.load(open(jp)).get("judgments", [])
              if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(rd, "transcripts", "*.json")):
            d = json.load(open(tf))
            v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in sc:
                continue
            tp = [float(p) for m in d["messages"] if m.get("source") == "target"
                  for p in (m.get("gen_token_probs") or [])]
            if tp:
                rows.append((rnd, v, float(sc[v]), st.mean(tp), min(tp)))
    if not rows:
        return None
    by = {}
    for _, v, s, p, mn in rows:
        by.setdefault(v, []).append((s, p, mn))
    best = [max(v, key=lambda x: x[0]) for v in by.values()]
    return {"score": st.mean(b[0] for b in best), "prob": st.mean(b[1] for b in best),
            "worst": min(r[4] for r in rows), "nr": len({r[0] for r in rows})}

cells = sorted({os.path.basename(d)[:-len("_arm_tfon")]
                for d in glob.glob(R + "*_arm_tfon")})
print(f"{'cell':22s} | {'floor OFF':>16s} | {'floor ON':>16s} | {'dScore':>7s} {'dProb':>6s} | worstP off->on")
print("-" * 104)
ds, dp, flagged, pending = [], [], [], []
for c in cells:
    a, b = bop(R + c + "_arm_tfoff"), bop(R + c + "_arm_tfon")
    if not a or not b or a["nr"] < 3 or b["nr"] < 3:
        pending.append(c); continue
    d1, d2 = b["score"] - a["score"], b["prob"] - a["prob"]
    ds.append(d1); dp.append(d2)
    if d1 < -NOISE:
        flagged.append((c, d1))
    print(f"{c:22s} | {a['score']:6.2f} / {a['prob']:5.1f}% | {b['score']:6.2f} / {b['prob']:5.1f}% | "
          f"{d1:+7.2f} {d2:+6.1f} | {a['worst']:.5f} -> {b['worst']:.5f}")

n = len(ds)
print(f"\ncomplete pairs: {n}   (pending/partial: {len(pending)})")
if n:
    print(f"  score delta  mean {st.mean(ds):+.3f}  median {st.median(ds):+.3f}"
          f"  range {min(ds):+.2f}..{max(ds):+.2f}   positive in {sum(1 for x in ds if x>0)}/{n}")
    print(f"  prob  delta  mean {st.mean(dp):+.2f}pp median {st.median(dp):+.2f}pp"
          f"  range {min(dp):+.1f}..{max(dp):+.1f}")
    print(f"  cells where floor-ON loses beyond the {NOISE} noise band: "
          f"{len(flagged)}/{n} {flagged if flagged else ''}")
if pending:
    print("  pending:", " ".join(pending))
