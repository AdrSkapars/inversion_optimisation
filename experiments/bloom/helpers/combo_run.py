#!/usr/bin/env python3
"""COMBO runner — G-PAIR input refinement + jail logit steering, TOGETHER (generalized driver).

Both on: refinement_input (t3_sfull: BLOOM_REFINE=1, HIST_TRANSCRIPT=3, HIST_STRATEGY=all) AND
jailbroken_output (self-jail at a beta). Reuses the cell bank (never rebuilds). Generalizes the
one-off combo_jailrefine.py: sweep multiple betas, choose 15-scen sweep or 100-scen final, override beta.

Stages of the combo experiment (all this driver, different flags):
  A sweep  6 cells x pm3 beta, 15 scen/seed 1, runs_new/combo         : compare combo@5 vs jail-alone@5.
  B around beta  cells that lost, betas {opt-0.5,opt,opt+0.5,opt+1.0}, runs_new/combo : does a nearby beta win?
  C final  best beta per cell, 100 scen/seed 100, runs_final/combo, bank-mode behaviour.

beta source: --betas pm3 (read each cell's param_selection picks.arith.pm3.beta) OR a comma-sep list
(applied to every cell). Output: <out_root>/<beh>/<model>/<config>/beta_<beta>/. Reports best-of-pool
elic/plaus (+ cross-turn Jaccard) per run.

Examples (repo root, venv python + HF env exported):
  # Stage A
  python experiments/bloom/helpers/combo_run.py --cells self_harm/Qwen_Qwen3.5-4B,... \
      --betas pm3 --out-root runs_new --config combo --scenarios 15 --seed 1 --rounds 7
  # Stage B (around optimal for one cell)
  python experiments/bloom/helpers/combo_run.py --cells deception/Qwen_Qwen3.5-4B \
      --betas 2.0,2.5,3.0,3.5 --out-root runs_new --config combo
  # Stage C (final)
  python experiments/bloom/helpers/combo_run.py --cells self_harm/Qwen_Qwen3.5-4B --betas 1.5 \
      --out-root runs_final --config combo --scenarios 100 --seed 100 --rounds 7 --bank-mode behaviour
"""
import argparse, json, os, re, subprocess, sys, time, glob, statistics as st
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT  = SCRIPT_DIR.parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
BLOOM      = REPO_ROOT / "src" / "bloom" / "bloom_corrupt.py"
NEW_ROOT   = RUNS_ROOT / "runs_new"
FINAL_ROOT = RUNS_ROOT / "runs_final"

TURNS = 3
HIST_TRANSCRIPT = "3"
HIST_STRATEGY   = "all"


def _pm3_beta(pj):
    d = json.load(open(pj, encoding="utf-8"))
    return (((d.get("picks") or {}).get("arith") or {}).get("pm3") or {}).get("beta")


def _meta(cell, bank_mode):
    beh, model_dir = cell.split("/", 1)
    pj = NEW_ROOT / beh / model_dir / "param_selection.json"
    if not pj.exists():
        return None
    d = json.load(open(pj, encoding="utf-8"))
    bank = (FINAL_ROOT / beh / "_bank") if bank_mode == "behaviour" else (NEW_ROOT / beh / model_dir / "_bank")
    return {"beh": beh, "model_dir": model_dir, "model": model_dir.replace("_", "/", 1),
            "behaviour_file": d.get("behaviour_file"), "pm3": _pm3_beta(pj), "bank": bank}


def _gpu_line():
    try:
        return " | ".join(l.strip() for l in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines())
    except Exception:
        return "n/a"


def _words(t): return re.findall(r"[a-z0-9']+", (t or "").lower())
def _jac(a, b):
    A, B = set(_words(a)), set(_words(b))
    return len(A & B) / len(A | B) if A and B else 0.0

def _score(run_dir, max_round=None):
    byv = defaultdict(list)
    for tf in glob.glob(str(run_dir / "round_*" / "transcripts" / "*.json")):
        if max_round is not None:
            m = re.search(r"round_(\d+)", tf)
            if m and int(m.group(1)) > max_round: continue
        try: d = json.load(open(tf, encoding="utf-8"))
        except Exception: continue
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None: continue
        tg = [m.get("content", "") for m in d.get("messages", []) if m.get("source") == "target"]
        xj = [_jac(tg[i], tg[i + 1]) for i in range(len(tg) - 1)]
        byv[d.get("metadata", {}).get("variation_number")].append(
            {"score": float(sc), "plaus": float(ps["mean"]), "xturn": st.mean(xj) if xj else 0.0})
    if not byv: return None
    sel = [max(v, key=lambda r: r["score"]) for v in byv.values()]
    return {"elic": round(st.mean(s["score"] for s in sel), 3),
            "plaus": round(st.mean(s["plaus"] for s in sel), 2),
            "xturn": round(st.mean(s["xturn"] for s in sel), 3), "n_scen": len(sel)}


def _run(m, beta, out_dir, a):
    if not a.no_resume and (out_dir / f"round_{a.rounds}" / "judgment.json").exists():
        print(f"  [beta {beta}] resume: done -> skip", flush=True); return True
    env = dict(os.environ)
    env.update({
        "BLOOM_RUNS_ROOT":      str(RUNS_ROOT),
        "BLOOM_FOLDER":         str(out_dir.relative_to(RUNS_ROOT)),
        "BLOOM_TARGET_MODEL":   "local/" + m["model"],
        "BLOOM_BEHAVIOR_FILE":  m["behaviour_file"],
        "BLOOM_EVAL_GPU":       str(a.eval_gpu),
        "BLOOM_TARGET_GPU":     str(a.target_gpu),
        "BLOOM_MAX_TURNS":      str(TURNS),
        "BLOOM_NUM_ROUNDS":     str(a.rounds),
        "BLOOM_NUM_SCENARIOS":  str(a.scenarios),
        "BLOOM_SEED":           str(a.seed),
        "BLOOM_KICKOFF_BANK":   str(m["bank"]),
        "BLOOM_JAIL_VAR_BATCH": str(a.var_batch),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # BOTH on:
        "BLOOM_REFINE":                 "1",
        "BLOOM_REFINE_HIST_TRANSCRIPT": HIST_TRANSCRIPT,
        "BLOOM_REFINE_HIST_STRATEGY":   HIST_STRATEGY,
        "BLOOM_JAIL_MODEL":             "local/" + m["model"],   # self-jail
        "BLOOM_JAIL_BETA":              str(beta),
    })
    env.pop("BLOOM_TOKBIAS_ENABLED", None)   # make sure tokbias isn't leaking in
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    print(f"  [beta {beta}] refine t3_sfull + jail | scen={a.scenarios} seed={a.seed} rounds={a.rounds} "
          f"-> {out_dir}\n        gpu[{_gpu_line()}]  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and (out_dir / f"round_{a.rounds}" / "judgment.json").exists()
    print(f"  [beta {beta}] {'OK' if ok else 'FAILED (see log)'}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="comma-sep beh/model_dir")
    ap.add_argument("--betas", default="pm3", help="'pm3' (per-cell optimal) or comma-sep floats applied to all cells")
    ap.add_argument("--out-root", default="runs_new")
    ap.add_argument("--config", default="combo")
    ap.add_argument("--bank-mode", choices=["model", "behaviour"], default="model")
    ap.add_argument("--scenarios", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--var-batch", type=int, default=15)
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    out_base = RUNS_ROOT / a.out_root
    use_pm3 = (a.betas.strip().lower() == "pm3")
    fixed_betas = [] if use_pm3 else [float(x) for x in a.betas.split(",") if x.strip() != ""]

    jobs = []
    for c in cells:
        m = _meta(c, a.bank_mode)
        if m is None or not m["bank"].exists() or (use_pm3 and m["pm3"] is None):
            print(f"  [skip] {c}: no param_selection/bank/pm3", flush=True); continue
        betas = [m["pm3"]] if use_pm3 else fixed_betas
        jobs.append((m, betas))

    if a.list:
        for m, betas in jobs:
            print(f"{m['beh']}/{m['model_dir']}  betas={betas}  bank={'ok' if m['bank'].exists() else 'MISSING'}")
        print(f"\n{len(jobs)} cells | out={a.out_root}/<cell>/{a.config}/beta_* | scen={a.scenarios} "
              f"seed={a.seed} rounds={a.rounds} bank={a.bank_mode}")
        return

    print(f"== COMBO {a.config}: {len(jobs)} cells | betas={'pm3' if use_pm3 else fixed_betas} | "
          f"scen={a.scenarios} seed={a.seed} rounds={a.rounds} | out={a.out_root} bank={a.bank_mode} ==", flush=True)
    for m, betas in jobs:
        cfg_dir = out_base / m["beh"] / m["model_dir"] / a.config
        print(f"\n=== {m['beh']} x {m['model']} | betas {betas} ===", flush=True)
        results = {}
        for beta in betas:
            b_s = f"{float(beta):g}"
            out_dir = cfg_dir / f"beta_{b_s}"
            if not _run(m, b_s, out_dir, a):
                results[b_s] = {"error": "run failed"}
                if not a.keep_going:
                    print(f"  ABORT {m['beh']}/{m['model_dir']} at beta {b_s}", flush=True); break
                continue
            sc5 = _score(out_dir, max_round=5); sc = _score(out_dir)
            results[b_s] = {"at5": sc5, "full": sc}
            if sc: print(f"  [beta {b_s}] elic@5={sc5['elic'] if sc5 else '?'} elic@{a.rounds}={sc['elic']} "
                         f"plaus={sc['plaus']}% xturn={sc['xturn']}", flush=True)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"cell": f"{m['beh']}/{m['model_dir']}", "pm3": m["pm3"],
                   "settings": {"scenarios": a.scenarios, "seed": a.seed, "rounds": a.rounds,
                                "bank_mode": a.bank_mode}, "betas": results},
                  open(cfg_dir / "combo_results.json", "w"), indent=2)
    print("\n== DONE ==", flush=True)


if __name__ == "__main__":
    main()
