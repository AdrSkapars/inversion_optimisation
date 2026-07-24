"""Degeneracy analysis for WILT sweep transcripts (post-hoc; not part of the src/bloom method).

Two degeneracy signals, computed on the TARGET responses of a transcript:
  * cross-turn     : mean consecutive-turn word-set Jaccard within a transcript
                     -> catches turn-level repetition (the same answer re-emitted each turn).
                        This is the one that reaches the picks and inflates plausibility.
  * within-message : top-word share (max over responses)
                     -> catches single-word / phrase loops ("no no no"); rarely selected.

Two commands:
  scan   for each cell (behaviour x model) at its pm5 beta, reproduce the utopia-weighted
         band pick, then measure the degeneracy of the transcripts it actually SELECTS.
  sweep  drop transcripts whose cross-turn Jaccard exceeds a threshold BEFORE the per-scenario
         selection (betas unchanged), re-select, and report how flagged-picks / elicitation /
         plausibility / degeneracy / dropped-scenarios move across thresholds.

This is analysis-stage tooling: it re-selects over already-generated transcripts and never
re-runs the sweep. Run from experiments/bloom:
    python helpers/degeneracy.py scan   [--runs runs_new]
    python helpers/degeneracy.py sweep  [--runs runs_new]
"""
import argparse, json, glob, os, re, statistics as st
from collections import defaultdict, Counter

NW = 25                                     # weight-grid resolution (matches param_sweep._curve)
BEH = ["racial","political","delusions","deception","selfpres",
       "reward_hacking","self_harm","medical","goblin"]
XTURN_FLAG = 0.60                           # chosen transcript flagged if cross-turn Jaccard exceeds this
TOP_FLAG = 0.25                             # within-message flag: max top-word share over responses
SWEEP_THRS = [9.9, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]   # 9.9 == no filter

def _words(t): return re.findall(r"[a-z0-9']+", (t or "").lower())
def _top(t):
    w = _words(t); return max(Counter(w).values()) / len(w) if w else 0.0
def _jac(a, b):
    A, B = set(_words(a)), set(_words(b))
    return len(A & B) / len(A | B) if A and B else 0.0
def _beta_folder(b):
    return "bon" if not b else "jail_b" + f"{b:.1f}".rstrip("0").rstrip(".")

def _load(run_dir):
    """Per-transcript (var, score, arith, top, xturn) for one run folder."""
    rows = []
    for tf in glob.glob(f"{run_dir}/round_*/transcripts/*.json"):
        try: d = json.load(open(tf, encoding="utf-8"))
        except Exception: continue
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None: continue
        tg = [m.get("content", "") for m in d.get("messages", []) if m.get("source") == "target"]
        xj = [_jac(tg[i], tg[i + 1]) for i in range(len(tg) - 1)]
        rows.append({"var": d.get("metadata", {}).get("variation_number"),
                     "score": float(sc), "arith": float(ps["mean"]),
                     "top": max((_top(x) for x in tg), default=0.0),
                     "xturn": st.mean(xj) if xj else 0.0})
    return rows

def _select(rows, anchor, xturn_thr=9.9):
    """Reproduce the pm5 pick (max-elic curve point with plaus >= anchor-5) after dropping
    transcripts with cross-turn Jaccard > xturn_thr. Returns (elic, plaus, selected_rows) or None."""
    byv = defaultdict(list)
    for r in rows:
        if r["xturn"] <= xturn_thr: byv[r["var"]].append(r)
    if not byv: return None
    best = None
    for i in range(NW):
        lam = i / (NW - 1)
        sel = [max(p, key=lambda r: (1 - lam) * (r["arith"] / 100) + lam * (r["score"] / 10)) for p in byv.values()]
        plaus = st.mean([s["arith"] for s in sel]); elic = st.mean([s["score"] for s in sel])
        if plaus >= anchor - 5.0 and (best is None or elic > best[0]):
            best = (elic, plaus, sel)
    return best

def _iter_cells(runs):
    for beh in BEH:
        base = f"{runs}/{beh}"
        if not os.path.isdir(base): continue
        for model in sorted(os.listdir(base)):
            pj = f"{base}/{model}/param_selection.json"
            if not os.path.exists(pj): continue
            ps = json.load(open(pj))
            beta = ps["picks"]["arith"]["pm5"]["beta"]; anchor = ps["anchor"]["arith"]
            rows = _load(f"{base}/{model}/{_beta_folder(beta)}")
            if rows: yield beh, model, beta, anchor, rows

def cmd_scan(runs):
    """Degeneracy of the transcripts actually chosen at each cell's pm5 pick."""
    chosen = []
    for beh, model, beta, anchor, rows in _iter_cells(runs):
        r = _select(rows, anchor)
        if r is None: continue
        for s in r[2]:
            chosen.append((beh, model, s))
    fx = [c for c in chosen if c[2]["xturn"] > XTURN_FLAG]
    fw = [c for c in chosen if c[2]["top"] > TOP_FLAG]
    print(f"chosen transcripts: {len(chosen)}")
    print(f"  cross-turn flagged (Jaccard>{XTURN_FLAG}): {len(fx)}")
    print(f"  within-message flagged (top-word>{TOP_FLAG}): {len(fw)}")
    print("  cross-turn flags by behaviour:", dict(Counter(c[0] for c in fx)))
    print("  cross-turn flags by model:    ", dict(Counter(c[1].split('_')[0] for c in fx)))

def cmd_sweep(runs):
    """How flagged picks / elic / plaus / degen / dropped-scenarios move as the cross-turn
    filter threshold tightens (betas unchanged)."""
    per_cell = []
    for beh, model, beta, anchor, rows in _iter_cells(runs):
        n_scen = len(set(r["var"] for r in rows))
        rec = {"beh": beh, "n_scen": n_scen, "thr": {}}
        for thr in SWEEP_THRS:
            r = _select(rows, anchor, thr)
            rec["thr"][thr] = None if r is None else {
                "elic": r[0], "plaus": r[1], "n_kept": len(r[2]),
                "flag_x": sum(1 for s in r[2] if s["xturn"] > XTURN_FLAG),
                "mean_x": st.mean([s["xturn"] for s in r[2]])}
        per_cell.append(rec)
    tot = sum(c["n_scen"] for c in per_cell)
    print(f"{'xturn thr':10}{'elic':>7}{'plaus':>8}{'chosen xJac':>12}{'flagged':>9}{'scen dropped':>14}")
    for thr in SWEEP_THRS:
        rs = [c["thr"][thr] for c in per_cell if c["thr"].get(thr)]
        drop = sum(c["n_scen"] - c["thr"][thr]["n_kept"] for c in per_cell if c["thr"].get(thr))
        lbl = "none" if thr > 1 else f"{thr:.2f}"
        print(f"{lbl:10}{st.mean([r['elic'] for r in rs]):>7.2f}{st.mean([r['plaus'] for r in rs]):>8.2f}"
              f"{st.mean([r['mean_x'] for r in rs]):>12.3f}{sum(r['flag_x'] for r in rs):>9}"
              f"{str(drop)+'/'+str(tot):>14}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "sweep"])
    ap.add_argument("--runs", default="runs_new", help="runs root under experiments/bloom (default runs_new)")
    a = ap.parse_args()
    (cmd_scan if a.cmd == "scan" else cmd_sweep)(a.runs)
