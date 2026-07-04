#!/usr/bin/env python3
"""Full Gemma table: for each (label, folder, rounds) pull avg/elic/tok_avg/min-of-mins from the
log's Round-1 line (1-round) or best-of-N behaviour (multi-round). CPU only."""
import json, glob, re, collections
BASE = "experiments/bloom/runs_init"

def r1(folder):
    try: t = open(f"/workspace/{folder}.log").read()
    except Exception: return None
    m = re.findall(r"Round 1: avg=([0-9.]+), elicitation_rate=([0-9.]+), tok_avg=([0-9.]+)%, tok_meanmin=([0-9.]+)%, tok_min=([0-9.]+)%", t)
    return m[0] if m else None

def bestN(folder, rounds):
    best = collections.defaultdict(lambda: -1)
    for rr in range(1, rounds + 1):
        try: d = json.load(open(f"{BASE}/{folder}/round_{rr}/judgment.json"))
        except Exception: continue
        for j in d["judgments"]:
            v, s = j.get("variation_number"), j.get("behavior_presence")
            if v is not None and s is not None: best[v] = max(best[v], s)
    vs = list(best.values())
    return (sum(vs)/len(vs), sum(x>=5 for x in vs)/len(vs)) if vs else (None, None)

ROWS = [
 ("BoN=1","gemma4e_bon3",1),("BoN=3 (best-of-3)","gemma4e_bon3",3),
 ("Corr (1,6,0) no-prefill","gemma4e_corr",1),("Corr (1,6,0) prefill","gemma4e_corr_pf",1),
 ("Corr b3=1 no-prefill, orig","gemma4e_corr_b3np",1),("Corr b3=1 prefill, orig","gemma4e_corr_b3",1),
 ("Corr b3=1 prefill, nswap","gemma4e_neutralswap",1),("Corr b3=1 prefill, +input","gemma4e_inclinput",1),
 ("Corr b3=2 prefill, orig","gemma4e_orig_b3_2",1),("Corr b3=2 prefill, nswap","gemma4e_nswap_b3_2",1),
 ("Corr b3=3 prefill, orig","gemma4e_orig_b3_3",1),("Corr b3=3 prefill, nswap","gemma4e_nswap_b3_3",1),
 ("Corr (b2=12,b3=6) nswap","gemma4e_nswap_b2_12_b3_6",1),("Corr (b2=3,b3=1) nswap","gemma4e_nswap_b2_3_b3_1",1),
 ("Corr (6,3) nswap floor1e-3","gemma4e_nswap63_floor1e3",1),
 ("Corr (6,3) nswap b1=2","gemma4e_nswap_b1_2",1),("Corr (6,3) nswap b1=3","gemma4e_nswap_b1_3",1),
 ("Corr abliterated","gemma4e_ablit",1),
 ("Corr prefill best-of-3","gemma4e_corr3r",3),("Corr abliterated best-of-3","gemma4e_ablit3r",3),
]
print(f"{'label':32}| avg | elic | tokavg% | minmin%")
for label, folder, rounds in ROWS:
    d = r1(folder)
    tokavg = d[2] if d else "?"; minmin = d[4] if d else "?"
    if rounds > 1:
        avg, elic = bestN(folder, rounds)
        avg = f"{avg:.2f}" if avg is not None else "?"; elic = f"{elic:.2f}" if elic is not None else "?"
    else:
        avg = d[0] if d else "?"; elic = d[1] if d else "?"
    print(f"{label:32}| {avg:>4} | {elic:>4} | {tokavg:>6} | {minmin}")
