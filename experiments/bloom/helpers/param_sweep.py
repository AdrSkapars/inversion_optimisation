#!/usr/bin/env python3
"""WILT hyperparameter sweep runner — adaptive beta search with band-selection.

Runs the vanilla/BoN baseline (beta=0) first, then increments beta by a fixed step,
each run a <scenarios> x <rounds> x <turns> WILT rollout that REUSES a shared per-round
kickoff bank (so the beta comparison is controlled). After every beta it scores the
arithmetic-mean selection curve and checks the plausibility GATE: once a beta's
max-plausibility operating point falls below (anchor_arith - 5pp) it can no longer enter
the +-5 band, and neither can any higher beta (beta is monotone in plausibility), so we
stop one beta past the gate (safety margin) or at --max-beta.

BoN (beta=0) is run with NO jail model, so it hits bloom_corrupt's target_only fast path
(plain target sampling, floor off) — this IS the anchor. Every beta>0 is self-jail
(jail model = target) at that beta. All runs share ONE kickoff bank (BoN generates it).

Selection curve: per scenario there is a pool of <rounds> transcripts; for a grid of
weights w in [0,1] we pick, per scenario, the transcript maximising
w*score_norm + (1-w)*plaus_norm, average over scenarios -> one (plaus, elic) point.
Sweeping w traces the frontier. The band pick is the highest-elic point, across all
betas' curves, whose plausibility lies in [anchor-band, anchor+band].

Writes ONE json per (model, behaviour) in the format documented in the repo.

Usage (run from the repo root, with the venv python and HF env already exported):
  python experiments/bloom/helpers/param_sweep.py \
      --model google/gemma-4-e4b-it --behaviour political \
      --behaviour-file prompts/2_political_bias.yaml \
      --eval-gpu 0 --target-gpu 1 \
      [--scenarios 15] [--rounds 5] [--turns 3] [--increment 0.25] [--max-beta 4.0] \
      [--out-base runs_new/WILT] [--resume]
"""
import argparse, json, os, shutil, subprocess, sys, time, glob, math, re, statistics as st
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # experiments/bloom/helpers
BLOOM = SCRIPT_DIR.parent / "bloom_corrupt.py"        # experiments/bloom/bloom_corrupt.py
REPO_BLOOM = "experiments/bloom"                      # BLOOM_FOLDER is relative to experiments/bloom
NW = 25                                               # weight-grid resolution for the selection curve


# ----------------------------- scoring --------------------------------------
def _load_points(run_dir: Path):
    """Per-transcript (var, score, arith%, geo%) over all rounds of a completed run."""
    rows = []
    for tf in glob.glob(str(run_dir / "round_*" / "transcripts" / "*.json")):
        try:
            d = json.load(open(tf, encoding="utf-8"))
        except Exception:
            continue
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None:
            continue
        rows.append({"var": d.get("metadata", {}).get("variation_number"),
                     "score": float(sc), "arith": float(ps["mean"]), "geo": float(ps["geomean"])})
    return rows


def _curve(rows, xkey):
    """Selection-weight frontier: list of (plaus, elic), one per weight in the grid."""
    byv = defaultdict(list)
    for r in rows:
        byv[r["var"]].append(r)
    out = []
    for i in range(NW):
        lam = i / (NW - 1)
        ps, ss = [], []
        for pool in byv.values():
            best = max(pool, key=lambda r: (1 - lam) * (r[xkey] / 100.0) + lam * (r["score"] / 10.0))
            ps.append(best[xkey]); ss.append(best["score"])
        out.append((round(st.mean(ps), 3), round(st.mean(ss), 3)))
    return out


def _best_of_pool(rows, xkey):
    """Per-scenario max-score selection; returns (avg score, avg plaus of the picks)."""
    byv = defaultdict(list)
    for r in rows:
        byv[r["var"]].append(r)
    picks = [max(p, key=lambda r: r["score"]) for p in byv.values()]
    return (round(st.mean([p["score"] for p in picks]), 3),
            round(st.mean([p[xkey] for p in picks]), 3),
            len(byv))


def _band_pick(curves_by_beta, xkey, anchor, band):
    """Highest-elic point across all betas' curves with plaus in [anchor-band, anchor+band].
    Returns {beta, score, plaus, runner_up, margin} or None if the band is empty."""
    per_beta_best = {}
    for beta, cv in curves_by_beta.items():
        inb = [(p, e) for (p, e) in cv[xkey] if anchor - band <= p <= anchor + band]
        if inb:
            p, e = max(inb, key=lambda t: t[1])
            per_beta_best[beta] = (e, p)
    if not per_beta_best:
        return None
    ranked = sorted(per_beta_best.items(), key=lambda kv: kv[1][0], reverse=True)
    (wb, (we, wp)) = ranked[0]
    ru = ranked[1][0] if len(ranked) > 1 else None
    margin = round(we - ranked[1][1][0], 3) if len(ranked) > 1 else None
    return {"beta": wb, "score": round(we, 3), "plaus": round(wp, 3),
            "runner_up": ru, "margin": margin}


# ----------------------------- running --------------------------------------
def _run(beta, cell, out_dir: Path, bank_dir: Path, resume: bool, bon_dir: Path = None) -> bool:
    """One WILT run. beta==0 -> BoN (no jail model, target_only). Returns True on success."""
    if resume and (out_dir / "round_1" / "judgment.json").exists() \
            and (out_dir / f"round_{cell['rounds']}" / "judgment.json").exists():
        print(f"  [beta {beta}] resume: all {cell['rounds']} rounds present, skipping run", flush=True)
        return True
    if beta > 0 and bon_dir is not None:
        # Reuse understanding + ideation from the BoN run (identical for the same behaviour+model);
        # bloom_corrupt skips any round-1 stage whose json already exists -> no re-derivation.
        _sr, _ds = bon_dir / "round_1", out_dir / "round_1"
        _ds.mkdir(parents=True, exist_ok=True)
        for _n in ("understanding.json", "ideation.json"):
            if (_sr / _n).exists() and not (_ds / _n).exists():
                shutil.copy2(_sr / _n, _ds / _n)
        if (_ds / "ideation.json").exists():
            print(f"  [beta {beta}] reusing understanding+ideation from bon/ (skips regeneration)", flush=True)
    env = dict(os.environ)
    env.update({
        "BLOOM_FOLDER": str(out_dir.relative_to(SCRIPT_DIR.parent)),   # relative to experiments/bloom
        "BLOOM_TARGET_MODEL": "local/" + cell["model"],
        "BLOOM_BEHAVIOR_FILE": cell["behaviour_file"],
        "BLOOM_EVAL_GPU": str(cell["eval_gpu"]), "BLOOM_TARGET_GPU": str(cell["target_gpu"]),
        "BLOOM_MAX_TURNS": str(cell["turns"]), "BLOOM_NUM_ROUNDS": str(cell["rounds"]),
        "BLOOM_NUM_SCENARIOS": str(cell["scenarios"]), "BLOOM_SEED": "1",
        "BLOOM_KICKOFF_BANK": str(bank_dir),
    })
    if beta > 0:                       # jail arm: self-jail at this beta (floor on by default)
        env["BLOOM_JAIL_MODEL"] = "local/" + cell["model"]
        env["BLOOM_JAIL_BETA"] = str(beta)
    # beta==0: no BLOOM_JAIL_MODEL -> bloom_corrupt reroutes to the beta=0 target_only BoN path
    log = out_dir.parent / f"{out_dir.name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [beta {beta}] running -> {out_dir}  (log: {log})", flush=True)
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(SCRIPT_DIR.parent.parent.parent),
                           env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and (out_dir / f"round_{cell['rounds']}" / "judgment.json").exists()
    print(f"  [beta {beta}] {'OK' if ok else 'FAILED (see log)'}", flush=True)
    return ok


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(SCRIPT_DIR)).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)                 # e.g. google/gemma-4-e4b-it
    ap.add_argument("--behaviour", required=True)             # short label, e.g. political
    ap.add_argument("--behaviour-file", required=True)        # prompts/2_political_bias.yaml
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--scenarios", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--increment", type=float, default=0.25)
    ap.add_argument("--max-beta", type=float, default=4.0)
    ap.add_argument("--out-base", default="runs_new")         # under experiments/bloom/ -> runs_new/<behaviour>/<model>/<method>
    ap.add_argument("--out", default=None)                    # json path; default beside the runs
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cell = {"model": a.model, "behaviour": a.behaviour, "behaviour_file": a.behaviour_file,
            "eval_gpu": a.eval_gpu, "target_gpu": a.target_gpu, "scenarios": a.scenarios,
            "rounds": a.rounds, "turns": a.turns}
    msan = a.model.replace("/", "_")
    base = SCRIPT_DIR.parent / a.out_base / a.behaviour / msan        # experiments/bloom/runs_new/WILT/<beh>/<model>
    bank = base / "_bank"
    out_json = Path(a.out) if a.out else (base / "param_selection.json")
    print(f"== WILT sweep: {a.model} x {a.behaviour} | {a.scenarios}x{a.rounds}x{a.turns} | "
          f"step {a.increment}, max {a.max_beta} ==", flush=True)

    curves = {}          # beta -> {"arith":[(p,e)...], "geo":[...]}
    per_beta = {}        # beta -> {score, arith, geo, n_scen}
    anchor = None
    gate_tripped_at = None
    ran_margin = None
    betas_run = []

    beta = 0.0
    while beta <= a.max_beta + 1e-9:
        out_dir = base / ("bon" if beta == 0 else f"jail_b{beta:g}")
        if not _run(beta, cell, out_dir, bank, a.resume, bon_dir=base / "bon"):
            print(f"  ABORT: beta {beta} failed.", flush=True)
            break
        rows = _load_points(out_dir)
        if not rows:
            print(f"  ABORT: beta {beta} produced no scorable transcripts.", flush=True)
            break
        curves[beta] = {"arith": _curve(rows, "arith"), "geo": _curve(rows, "geo")}
        sc, ar, n = _best_of_pool(rows, "arith")
        _, ge, _ = _best_of_pool(rows, "geo")
        per_beta[f"{beta:g}"] = {"score": sc, "arith": ar, "geo": ge, "n_scen": n}
        betas_run.append(beta)
        print(f"  [beta {beta}] best-of-pool score={sc} arith={ar}% geo={ge}%", flush=True)

        if beta == 0:                       # anchor = BoN best-of-pool (max-elicitation) plausibility
            anchor = {"source": "beta0_bon", "arith": ar, "geo": ge, "score": sc}
        else:
            # GATE on arithmetic +-5: beta's max-plausibility operating point (highest plaus on its curve).
            # Two CONSECUTIVE below-band betas -> exit (robust to a single noisy dip at 15 scenarios).
            max_plaus = max(p for (p, _) in curves[beta]["arith"])
            if max_plaus < anchor["arith"] - 5.0:
                if gate_tripped_at is None:
                    gate_tripped_at = beta
                    print(f"  GATE: beta {beta} max-plaus {max_plaus:.1f}% < anchor-5 "
                          f"({anchor['arith']-5:.1f}%). One margin beta then stop.", flush=True)
                else:
                    ran_margin = beta
                    print(f"  STOP: second consecutive below-band beta ({beta}).", flush=True)
                    break
            elif gate_tripped_at is not None:
                print(f"  beta {beta} recovered into the band -> reset gate.", flush=True)
                gate_tripped_at = None
        beta = round(beta + a.increment, 10)

    # ---- picks (both metrics, both bands) ----
    picks = {"primary_metric": "arith"}
    for m in ("arith", "geo"):
        aval = anchor[m]
        picks[m] = {"pm3": _band_pick(curves, m, aval, 3.0),
                    "pm5": _band_pick(curves, m, aval, 5.0)}

    def _auditor(run_dir):
        for tf in glob.glob(str(run_dir / "round_1" / "transcripts" / "*.json")):
            try:
                return json.load(open(tf, encoding="utf-8")).get("metadata", {}).get("evaluator_model")
            except Exception:
                pass
        return None

    result = {
        "model": a.model, "behaviour": a.behaviour, "behaviour_file": a.behaviour_file,
        "auditor": _auditor(base / "bon"), "kickoff_bank": str(bank.relative_to(SCRIPT_DIR.parent)),
        "settings": {"scenarios": a.scenarios, "rounds": a.rounds, "turns": a.turns,
                     "floor": 1e-4, "increment": a.increment, "max_beta": a.max_beta,
                     "gate_metric": "arith", "bands_pp": [3, 5]},
        "betas_run": betas_run,
        "stopped": {"reason": "gate" if gate_tripped_at is not None else "max_beta",
                    "gate_tripped_at": gate_tripped_at, "ran_margin": ran_margin},
        "anchor": anchor,
        "picks": picks,
        "per_beta": per_beta,
        "frontier": {"arith": {f"{b:g}": curves[b]["arith"] for b in curves},
                     "geo":   {f"{b:g}": curves[b]["geo"]   for b in curves}},
        "provenance": {"commit": _git_commit(),
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "pipeline": "WILT jail-only"},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n== wrote {out_json} ==", flush=True)
    print(f"  anchor arith={anchor['arith']}%  |  arith pm3={picks['arith']['pm3']}  "
          f"pm5={picks['arith']['pm5']}", flush=True)


if __name__ == "__main__":
    main()
