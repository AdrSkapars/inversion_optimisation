#!/usr/bin/env python3
"""Recompute the gemma BoN-20 best-beta dict at the CURRENT WINNER_TOL (jail_tune.py) from the
cached -20 folders. Needed because the running m20 sweep baked best_beta at the old TOL. No GPU."""
import os, glob, json, importlib.util
spec = importlib.util.spec_from_file_location("jt", os.path.expanduser("~/jail_tune.py"))
jt = importlib.util.module_from_spec(spec); spec.loader.exec_module(jt)

OFFSET = 20
combos = [("gemma", "goblin"), ("gemma", "medical")]
VALID = {jt.bc(round(i * 0.25, 2)): round(i * 0.25, 2) for i in range(0, 17)}

best = {}
for mo, sl in combos:
    cache = {}
    for R in jt.REPOS:
        for d in glob.glob(f"{R}/{mo}_{sl}_jailb*"):
            code = os.path.basename(d).split("jailb")[-1]
            if code not in VALID:
                continue
            m = jt.measure(f"{mo}_{sl}_jailb{code}")
            if m is not None:
                cache[VALID[code]] = m
    base = jt.measure(f"{mo}_{sl}_base")
    if not cache or base is None:
        print(f"skip {mo} {sl} (no cache/base)"); continue
    target = round(base[1] - OFFSET, 2)
    lo, up = jt._bracket(cache, target)
    res = {"P0": base[1], "target": target, "cache": cache, "lower": lo, "upper": up, "bon": (base[0], base[1])}
    bb = jt.best_beta(res)
    if bb is not None:
        best.setdefault(mo, {})[sl] = bb

out = os.path.expanduser("~/jail_tune_best_betas_m20.json")
json.dump(best, open(out, "w"), indent=2)
print(f"wrote {sum(len(v) for v in best.values())} combos -> {out} (WINNER_TOL={jt.WINNER_TOL})")
for mo in best:
    for sl in best[mo]:
        c = best[mo][sl]
        print(f"  {mo} {sl}: b{c['beta']:g} {c['score']}/{c['prob']} winner={c['winner']}")
