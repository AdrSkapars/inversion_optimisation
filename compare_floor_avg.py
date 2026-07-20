#!/usr/bin/env python3
"""floor-on vs floor-off on PLAIN AVERAGE behaviour presence (the paper's headline metric),
not best-of-pool. Best-of-pool takes max-over-rounds per scenario, which compresses
differences; the plain mean does not. Also reports elicitation rate (presence > 6)."""
import json, glob, os, re, statistics as st

R = "experiments/bloom/runs_init/"

def rows(run):
    """(round, scenario, score) for every judged sample."""
    out = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        j = json.load(open(jp))
        for e in j.get("judgments", []):
            v, s = e.get("variation_number"), e.get("behavior_presence")
            if v is not None and s is not None:
                out.append((rnd, v, float(s)))
    return out

def stats(run):
    rs = rows(run)
    if not rs:
        return None
    nrounds = len({r[0] for r in rs})
    pooled = [r[2] for r in rs]
    last = max(r[0] for r in rs)
    lastr = [r[2] for r in rs if r[0] == last]
    byscen = {}
    for _, v, s in rs:
        byscen.setdefault(v, []).append(s)
    bop = [max(v) for v in byscen.values()]
    return {"nr": nrounds,
            "pool_avg": st.mean(pooled), "pool_elic": sum(1 for s in pooled if s > 6) / len(pooled),
            "last_avg": st.mean(lastr),  "last_elic": sum(1 for s in lastr if s > 6) / len(lastr),
            "bop_avg": st.mean(bop),     "bop_elic": sum(1 for s in bop if s > 6) / len(bop)}

def beta_of(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

pairs = []
for on in sorted(glob.glob(R + "*_arm_floorb*")):
    m = re.search(r"^(.*)_arm_floorb(\d+)$", os.path.basename(on))
    if not m:
        continue
    off = R + f"{m.group(1)}_pareto_b{m.group(2)}"
    if os.path.isdir(off):
        pairs.append((m.group(1), beta_of(m.group(2)), on, off))
pairs.sort(key=lambda p: (p[0], p[1]))

hdr = (f"{'cell':15s} {'beta':>4s} {'floor':5s} | {'poolAvg':>7s} {'poolElic':>8s} | "
       f"{'lastAvg':>7s} {'lastElic':>8s} | {'bopAvg':>7s} {'bopElic':>7s}")
print(hdr); print("-" * len(hdr))
dp, dl, db = [], [], []
for cell, b, on, off in pairs:
    a, c = stats(on), stats(off)
    if not a or not c:
        continue
    if a["nr"] != 5 or c["nr"] != 5:
        print(f"{cell:15s} {b:4g} PARTIAL on={a['nr']}/5 off={c['nr']}/5 — skipped"); continue
    for lbl, r in (("ON", a), ("OFF", c)):
        print(f"{cell:15s} {b:4g} {lbl:5s} | {r['pool_avg']:7.2f} {r['pool_elic']:8.2f} | "
              f"{r['last_avg']:7.2f} {r['last_elic']:8.2f} | {r['bop_avg']:7.2f} {r['bop_elic']:7.2f}")
    dp.append(a["pool_avg"] - c["pool_avg"]); dl.append(a["last_avg"] - c["last_avg"])
    db.append(a["bop_avg"] - c["bop_avg"])
    print(f"{'  delta':15s} {'':4s} {'':5s} | {dp[-1]:+7.2f} {a['pool_elic']-c['pool_elic']:+8.2f} | "
          f"{dl[-1]:+7.2f} {a['last_elic']-c['last_elic']:+8.2f} | {db[-1]:+7.2f} {a['bop_elic']-c['bop_elic']:+7.2f}")
print()
n = len(dp)
print(f"n={n} matched pairs.  MEAN DELTA (floor-on minus floor-off):")
print(f"  pooled avg  {st.mean(dp):+.3f}   (positive in {sum(1 for x in dp if x>0)}/{n})")
print(f"  last-round  {st.mean(dl):+.3f}   (positive in {sum(1 for x in dl if x>0)}/{n})")
print(f"  best-of-pool{st.mean(db):+.3f}   (positive in {sum(1 for x in db if x>0)}/{n})")
