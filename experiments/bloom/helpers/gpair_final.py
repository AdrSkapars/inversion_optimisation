#!/usr/bin/env python3
"""G-PAIR FINAL comparison runs (100 scenarios).

G-PAIR = merged PAIR-style input refinement (jailbroken_output OFF + refinement_input ON), single
generation model + audit-model input refinement. FINAL config chosen from the hyperparam sweep:
  history_transcript_rounds = 3   (3 full transcripts shown to the refiner)
  history_strategy_rounds    = all (all prior (round,score,strategy) rows)
= config "t3_sfull". 100 scenarios, seed 100, 3 turns, 7 rounds, var_batch 25.

Reuses the 100-scenario per-behaviour bank the WILT BoN built (runs_final/<beh>/_bank): understanding
+ ideation + round-1 kickoffs are reused so inputs start IDENTICAL to BoN/jail; rounds 2+ are refined
(NOT reused). Output drops in beside the WILT arms: runs_final/<beh>/<model>/gpair_t3_sfull/.

Cells: Qwen + gemma x {self_harm, deception, political} = 6 (headline = Qwen self_harm).
"""
import argparse, json, os, subprocess, sys, time, glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT  = SCRIPT_DIR.parent                     # experiments/bloom
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
BLOOM      = REPO_ROOT / "src" / "bloom" / "bloom_corrupt.py"
SWEEP_ROOT = RUNS_ROOT / "runs_new"                # source of behaviour_file/auditor
FINAL_ROOT = RUNS_ROOT / "runs_final"

SCENARIOS = 100
SEED      = 100
TURNS     = 3
ROUNDS    = 7
VAR_BATCH = 25
HIST_TRANSCRIPT = "3"      # history_transcript_rounds
HIST_STRATEGY   = "all"    # history_strategy_rounds ("all" -> None)
CONFIG_NAME = "gpair_t3_sfull"

MODELS     = ["Qwen/Qwen3.5-4B", "google/gemma-4-e4b-it"]
BEHAVIOURS = ["self_harm", "deception", "political"]


def _cell_meta(beh, model_dir):
    pj = SWEEP_ROOT / beh / model_dir / "param_selection.json"
    if not pj.exists():
        return None
    d = json.load(open(pj, encoding="utf-8"))
    return {"behaviour_file": d.get("behaviour_file"), "auditor": d.get("auditor"),
            "bank": FINAL_ROOT / beh / "_bank"}


def _gpu_line():
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"]).decode().strip()
        return " | ".join(l.strip() for l in q.splitlines())
    except Exception:
        return "nvidia-smi unavailable"


def _done(out_dir, rounds):
    return (out_dir / f"round_{rounds}" / "judgment.json").exists()


def _run(beh, model, meta, out_dir, rounds, var_batch, eval_gpu, target_gpu, resume):
    if resume and _done(out_dir, rounds):
        print(f"  [gpair] resume: round_{rounds}/judgment.json present -> skip", flush=True)
        return True
    env = dict(os.environ)
    env.update({
        "BLOOM_RUNS_ROOT":      str(RUNS_ROOT),
        "BLOOM_FOLDER":         str(out_dir.relative_to(RUNS_ROOT)),
        "BLOOM_TARGET_MODEL":   "local/" + model,
        "BLOOM_BEHAVIOR_FILE":  meta["behaviour_file"],
        "BLOOM_EVAL_GPU":       str(eval_gpu),
        "BLOOM_TARGET_GPU":     str(target_gpu),
        "BLOOM_MAX_TURNS":      str(TURNS),
        "BLOOM_NUM_ROUNDS":     str(rounds),
        "BLOOM_NUM_SCENARIOS":  str(SCENARIOS),
        "BLOOM_SEED":           str(SEED),
        "BLOOM_KICKOFF_BANK":   str(meta["bank"]),        # reuse the 100-scen WILT bank
        "BLOOM_JAIL_VAR_BATCH": str(var_batch),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "BLOOM_REFINE":                 "1",              # jailbroken_output stays OFF (no BLOOM_JAIL_MODEL)
        "BLOOM_REFINE_HIST_TRANSCRIPT": HIST_TRANSCRIPT,
        "BLOOM_REFINE_HIST_STRATEGY":   HIST_STRATEGY,
    })
    env.pop("BLOOM_JAIL_MODEL", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    print(f"  [gpair t3_sfull] rounds={rounds} scen={SCENARIOS} seed={SEED} batch={var_batch} -> {out_dir}\n"
          f"        gpu[{_gpu_line()}]  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(REPO_ROOT),
                           env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and _done(out_dir, rounds)
    print(f"  [gpair t3_sfull] {'OK' if ok else 'FAILED (see log)'}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behaviours", default=",".join(BEHAVIOURS))
    ap.add_argument("--models", default=None, help="substring filter, e.g. Qwen or gemma")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--var-batch", type=int, default=VAR_BATCH)
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--only", default=None, help="single 'beh/model_dir'")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    behaviours = [b.strip() for b in args.behaviours.split(",") if b.strip()]
    models = [m for m in MODELS if (args.models is None or args.models.lower() in m.lower())]
    resume = not args.no_resume

    jobs = []
    for model in models:                              # model-major (one model per box)
        model_dir = model.replace("/", "_")
        for beh in behaviours:
            meta = _cell_meta(beh, model_dir)
            if meta is None or not meta["bank"].exists():
                print(f"  [skip] {beh}/{model_dir}: no param_selection or bank", flush=True)
                continue
            out_dir = FINAL_ROOT / beh / model_dir / CONFIG_NAME
            if args.only and args.only != f"{beh}/{model_dir}":
                continue
            jobs.append((beh, model, meta, out_dir))

    if args.list:
        for beh, model, meta, out_dir in jobs:
            print(f"{beh}/{model}  [{'done' if _done(out_dir, args.rounds) else 'pending'}]  bank={meta['bank']}")
        print(f"\n{len(jobs)} cells.")
        return

    print(f"G-PAIR FINAL: {len(jobs)} cells | {CONFIG_NAME} | scen={SCENARIOS} seed={SEED} "
          f"rounds={args.rounds} batch={args.var_batch}", flush=True)
    ok_n = fail_n = 0
    for beh, model, meta, out_dir in jobs:
        print(f"\n=== G-PAIR {beh} x {model} | t=3 strat=all | {SCENARIOS}x{args.rounds}x{TURNS} ===", flush=True)
        if _run(beh, model, meta, out_dir, args.rounds, args.var_batch, args.eval_gpu, args.target_gpu, resume):
            ok_n += 1
            try:
                json.dump({"behaviour": beh, "model": model, "config": CONFIG_NAME,
                           "history_transcript_rounds": 3, "history_strategy_rounds": "all",
                           "scenarios": SCENARIOS, "seed": SEED, "turns": TURNS, "rounds": args.rounds,
                           "var_batch": args.var_batch, "auditor": meta["auditor"],
                           "bank": str(meta["bank"])},
                          open(out_dir / "gpair_cell.json", "w", encoding="utf-8"), indent=2)
            except Exception:
                pass
        else:
            fail_n += 1
            if not args.keep_going:
                print(f"  ABORT: {beh}/{model} failed.", flush=True); break
    print(f"\n== DONE: {ok_n}/{ok_n+fail_n} G-PAIR final cells OK ==", flush=True)


if __name__ == "__main__":
    main()
