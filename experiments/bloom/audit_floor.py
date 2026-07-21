#!/usr/bin/env python3
"""Audit: for every BLOOM run on disk, was the target_floor (naturalness floor) active?
Reports per elicitation path (jail / corruption / plain) how many runs had a floor > 0."""
import json, glob, os
from collections import Counter, defaultdict

rows = []
for cfgp in glob.glob("experiments/bloom/runs_*/*/round_1/cfg.json"):
    run = cfgp.split(os.sep)[-3]
    root = cfgp.split(os.sep)[-4]
    try:
        c = json.load(open(cfgp))
    except Exception:
        continue
    j = c.get("jailbroken_output", {}) or {}
    co = c.get("corruption_output", {}) or {}
    jail_on = bool(j.get("use_during_rollout"))
    corr_on = bool(co.get("enabled"))
    jf = j.get("target_floor")
    cf = co.get("target_floor")
    jf = float(jf) if jf not in (None, "") else 0.0
    cf = float(cf) if cf not in (None, "") else 0.0
    if jail_on:
        mode, floor = "jail", jf
    elif corr_on:
        mode, floor = "corruption", cf
    else:
        mode, floor = "plain/BoN", 0.0
    rows.append((root, run, mode, floor, j.get("beta"), j.get("b1")))

print(f"total runs with cfg: {len(rows)}\n")
by_mode = defaultdict(list)
for r in rows:
    by_mode[r[2]].append(r)

for mode in ("jail", "corruption", "plain/BoN"):
    rs = by_mode.get(mode, [])
    if not rs:
        continue
    on = [r for r in rs if r[3] > 0]
    print(f"{mode}: {len(rs)} runs | floor ACTIVE in {len(on)} | floor OFF in {len(rs)-len(on)}")
    print(f"   distinct floor values: {dict(Counter(r[3] for r in rs))}")
    if on:
        print(f"   examples WITH floor: {[(r[1], r[3]) for r in on[:8]]}")
    print()

# every jail run, floor value, so nothing hides
jl = by_mode.get("jail", [])
print("ALL jail runs (name, floor, beta, b1):")
for r in sorted(jl, key=lambda x: x[1]):
    print(f"   {r[1][:52]:54s} floor={r[3]:<10g} beta={r[4]} b1={r[5]}")
