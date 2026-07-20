#!/usr/bin/env python3
"""Jail vs corruption vs BoN, per cell, as score / prob per row.
BOTH methods reported on the GEOMETRIC mean per-token probability, because that is what the
corruption path persists (target_p_pct = exp(mean log p)); jail's per-token list is converted
to match. Mixing the two raw statistics gives a fake ~20pp gap."""
import sys, os, glob, re, math, json, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
sys.path.insert(0, "/home/t75879as/inversion_optimisation")
from compare_methods import extract_jail, extract_corr

R = "experiments/bloom/runs_init/"

def bop(pts):
    if not pts:
        return None
    g = {}
    for p in pts:
        g.setdefault(p["scenario"], []).append(p)
    best = [max(v, key=lambda x: x["score"]) for v in g.values()]
    return st.mean(b["score"] for b in best), st.mean(b["prob"] for b in best), len({p["round"] for p in pts})

def beta_of(bc):
    frac = bc[1:] or "0"
    return int(bc[0]) + int(frac) / (10 ** len(frac))

for cell in ["qwen_selfharm", "llama_selfpres"]:
    b0 = bop(extract_jail(R + cell + "_pareto_b0"))
    print(f"\n===== {cell} =====   BoN (jail off) = {b0[0]:.2f} / {b0[1]:.1f}%")
    print(f"  {'arm':26s} {'score / prob':>16s}   {'BoN':>14s}")
    rows = []
    for d in glob.glob(R + cell + "_arm_floorb*"):
        m = re.search(r"_arm_floorb(\d+)$", os.path.basename(d))
        if m:
            rows.append((0, beta_of(m.group(1)), f"jail floor-ON  b={beta_of(m.group(1)):g}", d, extract_jail))
    for d in glob.glob(R + cell + "_arm_corrb*"):
        m = re.search(r"_arm_corrb(\d+)$", os.path.basename(d))
        if m:
            v = float(m.group(1))
            rows.append((1, v, f"corruption s1  b2={v:g}", d, extract_corr))
    d = R + cell + "_arm_corrp10b6"
    if os.path.isdir(d):
        rows.append((2, 6, "corruption p10 b2=6", d, extract_corr))
    for _, _, label, path, fn in sorted(rows):
        r = bop(fn(path))
        if not r or r[2] < 5:
            continue
        print(f"  {label:26s} {r[0]:6.2f} / {r[1]:5.1f}%   {b0[0]:6.2f} / {b0[1]:4.1f}%")
