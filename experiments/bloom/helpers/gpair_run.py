#!/usr/bin/env python3
"""G-PAIR hyperparameter sweep driver.

G-PAIR = merged PAIR-style input refinement (jailbroken_output OFF + refinement_input ON).
Single generation model + the audit model refines the input scenario across rounds. Routes the
same batched single-model hf_full decode as BoN (respects BLOOM_JAIL_VAR_BATCH).

This is the HYPERPARAM phase: 15 scenarios, seed 1 (the WILT-sweep convention), reusing the
EXISTING per-(behaviour,model) 15-scenario bank at runs_new/<beh>/<model_dir>/_bank so inputs
start identical to the sweep. Sweeps the two refiner history-depth knobs:
  - transcript depth  (history_transcript_rounds): {1,2,3}   env BLOOM_REFINE_HIST_TRANSCRIPT
  - strategy   depth  (history_strategy_rounds):   full(None) vs matched(=transcript depth)
                                                    env BLOOM_REFINE_HIST_STRATEGY ("all" or N)
Run ONCE at max rounds (default 7); score best-of-pool at each prefix (5/6/7) post-hoc.

Output: runs_new/<beh>/<model_dir>/gpair_t<T>_s<full|match>/  (sibling to the sweep's bon/jail arms).

Fixed to Qwen (the paper's representative slice) unless --model given. Behaviours default to the
3 headline ones. Reuses the sweep's param_selection.json for each cell's behaviour_file.
"""
import argparse, json, os, subprocess, sys, time, glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent              # experiments/bloom/helpers
RUNS_ROOT  = SCRIPT_DIR.parent                            # experiments/bloom
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent              # repo root
BLOOM      = REPO_ROOT / "src" / "bloom" / "bloom_corrupt.py"
SWEEP_ROOT = RUNS_ROOT / "runs_new"                       # completed sweep + the 15-scen banks

# Hyperparam-phase constants (match the WILT sweep so the bank is reusable).
SCENARIOS = 15
SEED      = 1
TURNS     = 3
ROUNDS    = 7                       # run once at 7; read 5/6/7 post-hoc (rounds are cumulative)
VAR_BATCH = 25                      # single generation model (no jail expert) -> BoN-sized batch

MODEL     = "Qwen/Qwen3.5-4B"       # representative slice
BEHAVIOURS = ["self_harm", "deception", "political"]
TRANSCRIPT_DEPTHS = [1, 2, 3]       # history_transcript_rounds; 3 may OOM/overflow -> --keep-going drops it
STRATEGY_MODES = ["full", "match"]  # full=all prior (round,score,strategy); match=same depth as transcripts


def _cell_meta(beh: str, model_dir: str):
    """Read the sweep's param_selection.json for this (beh, model) to recover behaviour_file/auditor."""
    pj = SWEEP_ROOT / beh / model_dir / "param_selection.json"
    if not pj.exists():
        return None
    d = json.load(open(pj, encoding="utf-8"))
    return {"behaviour_file": d.get("behaviour_file"), "auditor": d.get("auditor"),
            "bank": SWEEP_ROOT / beh / model_dir / "_bank"}


def _gpu_line():
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"]).decode().strip()
        return " | ".join(l.strip() for l in q.splitlines())
    except Exception:
        return "nvidia-smi unavailable"


def _done(out_dir: Path, rounds: int) -> bool:
    return (out_dir / f"round_{rounds}" / "judgment.json").exists()


def _run_one(beh, model, meta, t_depth, s_mode, out_dir: Path, rounds: int,
             var_batch: int, eval_gpu: int, target_gpu: int, resume: bool) -> bool:
    if resume and _done(out_dir, rounds):
        print(f"  [gpair t{t_depth} s{s_mode}] resume: round_{rounds}/judgment.json present -> skip", flush=True)
        return True
    s_env = "all" if s_mode == "full" else str(t_depth)   # full=None(all); match=same as transcript depth
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
        "BLOOM_KICKOFF_BANK":   str(meta["bank"]),          # reuse the 15-scen sweep bank
        "BLOOM_JAIL_VAR_BATCH": str(var_batch),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # G-PAIR: refinement ON, jail OFF (no BLOOM_JAIL_MODEL) -> single-model target_only decode.
        "BLOOM_REFINE":                 "1",
        "BLOOM_REFINE_HIST_TRANSCRIPT": str(t_depth),
        "BLOOM_REFINE_HIST_STRATEGY":   s_env,
    })
    env.pop("BLOOM_JAIL_MODEL", None)   # ensure jail stays off
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    print(f"  [gpair t{t_depth} s{s_mode}] rounds={rounds} scen={SCENARIOS} seed={SEED} -> {out_dir}\n"
          f"        gpu[{_gpu_line()}]  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(REPO_ROOT),
                           env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and _done(out_dir, rounds)
    print(f"  [gpair t{t_depth} s{s_mode}] {'OK' if ok else 'FAILED (see log)'}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def _configs(t_depths, s_modes):
    """Yield (t_depth, s_mode) grid, skipping the redundant t=1 duplicate (full==match at depth 1)."""
    for t in t_depths:
        for s in s_modes:
            if t == 1 and s == "match":   # match==full when depth is 1 -> skip the duplicate
                continue
            yield t, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behaviours", default=",".join(BEHAVIOURS), help="comma-sep behaviours")
    ap.add_argument("--model", default=MODEL, help="target model (default Qwen/Qwen3.5-4B)")
    ap.add_argument("--transcript-depths", default=",".join(map(str, TRANSCRIPT_DEPTHS)),
                    help="comma-sep history_transcript_rounds values (default 1,2,3)")
    ap.add_argument("--strategy-modes", default=",".join(STRATEGY_MODES),
                    help="comma-sep of full,match (default full,match)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--var-batch", type=int, default=VAR_BATCH)
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue on cell failure (e.g. t=3 OOM/overflow)")
    ap.add_argument("--only", default=None, help="single 'beh/gpair_tN_sMODE' to run one config")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    model = args.model
    model_dir = model.replace("/", "_")
    behaviours = [b.strip() for b in args.behaviours.split(",") if b.strip()]
    t_depths = [int(x) for x in args.transcript_depths.split(",") if x.strip()]
    s_modes = [x.strip() for x in args.strategy_modes.split(",") if x.strip()]
    resume = not args.no_resume

    # Build the job list: (beh, meta, t_depth, s_mode, out_dir)
    jobs = []
    for beh in behaviours:
        meta = _cell_meta(beh, model_dir)
        if meta is None:
            print(f"  [skip] {beh}/{model_dir}: no param_selection.json/bank in runs_new", flush=True)
            continue
        if not meta["bank"].exists():
            print(f"  [skip] {beh}/{model_dir}: bank missing at {meta['bank']}", flush=True)
            continue
        for t, s in _configs(t_depths, s_modes):
            name = f"gpair_t{t}_s{s}"
            out_dir = SWEEP_ROOT / beh / model_dir / name
            if args.only and args.only != f"{beh}/{name}":
                continue
            jobs.append((beh, meta, t, s, out_dir))

    if args.list:
        for beh, meta, t, s, out_dir in jobs:
            status = "done" if _done(out_dir, args.rounds) else "pending"
            print(f"{beh}/{model_dir} gpair_t{t}_s{s}  [{status}]  bank={meta['bank']}")
        print(f"\n{len(jobs)} configs.")
        return

    print(f"G-PAIR sweep: {len(jobs)} configs | model={model} | rounds={args.rounds} "
          f"scen={SCENARIOS} seed={SEED} batch={args.var_batch}", flush=True)
    ok_n = fail_n = 0
    for beh, meta, t, s, out_dir in jobs:
        print(f"\n=== G-PAIR {beh} x {model} | transcript-depth={t} strategy={s} | "
              f"{SCENARIOS}x{args.rounds}x{TURNS} ===", flush=True)
        ok = _run_one(beh, model, meta, t, s, out_dir, args.rounds, args.var_batch,
                      args.eval_gpu, args.target_gpu, resume)
        if ok:
            ok_n += 1
        else:
            fail_n += 1
            if not args.keep_going:
                print(f"  ABORT sweep: {beh} t{t} s{s} failed (use --keep-going to continue).", flush=True)
                break
    print(f"\n== DONE: {ok_n}/{ok_n+fail_n} G-PAIR configs OK ==", flush=True)


if __name__ == "__main__":
    main()
