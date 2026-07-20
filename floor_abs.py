#!/usr/bin/env python3
"""Absolute best-of-pool average behaviour score (best round per scenario, then averaged):
floor-ON vs floor-OFF vs jail-off BoN (beta=0). No deltas."""
import json, glob, os, re, statistics as st

R = "experiments/bloom/runs_init/"

def bop_avg(run):
    """avg over scenarios of (max score across rounds); plus n_rounds."""
    rs = []
    for jp in sorted(glob.glob(os.path.join(run, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        for e in json.load(open(jp)).get("judgments", []):
            v, s = e.get("variation_number"), e.get("behavior_presence")
            if v is not None and s is not None:
                rs.append((rnd, v, float(s)))
    if not rs:
        return None, 0
    by = {}
    for _, v, s in rs:
        by.setdefault(v, []).append(s)
    return st.mean(max(v) for v in by.values()), len({r[0] for r in rs})

def beta_of(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

cells = ["qwen_selfharm", "llama_selfpres"]
bon = {c: bop_avg(R + c + "_pareto_b0")[0] for c in cells}

hdr = f"{'cell':15s} {'beta':>5s} {'floorON':>8s} {'floorOFF':>9s} {'BoN(jail off)':>14s}"
print(hdr); print("-" * len(hdr))
for c in cells:
    betas = []
    for on in glob.glob(R + c + "_arm_floorb*"):
        m = re.search(r"_arm_floorb(\d+)$", os.path.basename(on))
        if m:
            betas.append((beta_of(m.group(1)), m.group(1)))
    for b, bc in sorted(betas):
        a, nra = bop_avg(R + f"{c}_arm_floorb{bc}")
        o, nro = bop_avg(R + f"{c}_pareto_b{bc}")
        if a is None or nra != 5:
            continue
        astr = f"{a:8.2f}"
        ostr = f"{o:9.2f}" if (o is not None and nro == 5) else f"{'--':>9s}"
        print(f"{c:15s} {b:5g} {astr} {ostr} {bon[c]:14.2f}")
    print()
