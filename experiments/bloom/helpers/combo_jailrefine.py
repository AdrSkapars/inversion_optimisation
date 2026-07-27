#!/usr/bin/env python3
"""COMBO experiment: WILT jail steering + G-PAIR input refinement, TOGETHER.

Turns on BOTH jailbroken_output (jail, logit PoE at the cell's pm3 beta) AND refinement_input
(G-PAIR, config t3_sfull: 3 transcripts + all strategies). Tests whether refined inputs give jail
better material on cells where jail alone underperformed. 15 scenarios, seed 1, 7 rounds.

Reuses the 15-scen sweep bank (runs_new/<beh>/<model>/_bank): understanding+ideation+round-1
kickoffs; rounds 2+ are refined AND rolled out with jail steering. Output:
runs_new/<beh>/<model>/combo_jailrefine/. pm3 beta read from the cell's param_selection.json.

Default cells = the lowest-jail WILT cells. Override with --cells 'beh/model_dir,...'.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT  = SCRIPT_DIR.parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
BLOOM      = REPO_ROOT / "src" / "bloom" / "bloom_corrupt.py"
SWEEP_ROOT = RUNS_ROOT / "runs_new"

SCENARIOS = 15
SEED      = 1
TURNS     = 3
ROUNDS    = 7
VAR_BATCH = 15                          # jail path steps 2 models -> tighter batch
HIST_TRANSCRIPT = "3"
HIST_STRATEGY   = "all"
CONFIG_NAME = "combo_jailrefine"

DEFAULT_CELLS = [                       # lowest-jail WILT cells
    "self_harm/google_gemma-4-e4b-it",
    "medical/google_gemma-4-e4b-it",
    "goblin/google_gemma-4-e4b-it",
    "racial/Qwen_Qwen3.5-4B",
]


def _meta(cell):
    beh, model_dir = cell.split("/", 1)
    pj = SWEEP_ROOT / beh / model_dir / "param_selection.json"
    if not pj.exists():
        return None
    d = json.load(open(pj, encoding="utf-8"))
    pm3 = (((d.get("picks") or {}).get("arith") or {}).get("pm3")) or {}
    beta = pm3.get("beta")
    model = "/".join(model_dir.split("_", 1)) if "_" in model_dir else model_dir
    # model_dir like "google_gemma-4-e4b-it" -> "google/gemma-4-e4b-it"; "Qwen_Qwen3.5-4B" -> "Qwen/Qwen3.5-4B"
    return {"beh": beh, "model_dir": model_dir, "model": model_dir.replace("_", "/", 1),
            "behaviour_file": d.get("behaviour_file"), "beta": beta,
            "bank": SWEEP_ROOT / beh / model_dir / "_bank"}


def _gpu_line():
    try:
        return " | ".join(l.strip() for l in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines())
    except Exception:
        return "n/a"


def _done(out_dir, rounds):
    return (out_dir / f"round_{rounds}" / "judgment.json").exists()


def _run(m, out_dir, rounds, var_batch, eval_gpu, target_gpu, resume):
    if resume and _done(out_dir, rounds):
        print(f"  [combo] resume: round_{rounds}/judgment.json present -> skip", flush=True); return True
    env = dict(os.environ)
    env.update({
        "BLOOM_RUNS_ROOT":      str(RUNS_ROOT),
        "BLOOM_FOLDER":         str(out_dir.relative_to(RUNS_ROOT)),
        "BLOOM_TARGET_MODEL":   "local/" + m["model"],
        "BLOOM_BEHAVIOR_FILE":  m["behaviour_file"],
        "BLOOM_EVAL_GPU":       str(eval_gpu),
        "BLOOM_TARGET_GPU":     str(target_gpu),
        "BLOOM_MAX_TURNS":      str(TURNS),
        "BLOOM_NUM_ROUNDS":     str(rounds),
        "BLOOM_NUM_SCENARIOS":  str(SCENARIOS),
        "BLOOM_SEED":           str(SEED),
        "BLOOM_KICKOFF_BANK":   str(m["bank"]),
        "BLOOM_JAIL_VAR_BATCH": str(var_batch),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # BOTH on:
        "BLOOM_REFINE":                 "1",
        "BLOOM_REFINE_HIST_TRANSCRIPT": HIST_TRANSCRIPT,
        "BLOOM_REFINE_HIST_STRATEGY":   HIST_STRATEGY,
        "BLOOM_JAIL_MODEL":             "local/" + m["model"],   # self-jail
        "BLOOM_JAIL_BETA":              str(m["beta"]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    print(f"  [combo jail+refine] beta={m['beta']} t3_sfull rounds={rounds} scen={SCENARIOS} -> {out_dir}\n"
          f"        gpu[{_gpu_line()}]  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and _done(out_dir, rounds)
    print(f"  [combo] {'OK' if ok else 'FAILED (see log)'}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS), help="comma-sep beh/model_dir")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--var-batch", type=int, default=VAR_BATCH)
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    resume = not args.no_resume
    jobs = []
    for c in cells:
        m = _meta(c)
        if m is None or m["beta"] is None or not m["bank"].exists():
            print(f"  [skip] {c}: no param_selection/beta/bank", flush=True); continue
        jobs.append((m, SWEEP_ROOT / m["beh"] / m["model_dir"] / CONFIG_NAME))

    if args.list:
        for m, out in jobs:
            print(f"{m['beh']}/{m['model_dir']}  beta={m['beta']}  [{'done' if _done(out, args.rounds) else 'pending'}]")
        print(f"\n{len(jobs)} cells."); return

    print(f"COMBO jail+refine: {len(jobs)} cells | t3_sfull | scen={SCENARIOS} seed={SEED} "
          f"rounds={args.rounds} batch={args.var_batch}", flush=True)
    ok_n = fail_n = 0
    for m, out in jobs:
        print(f"\n=== COMBO {m['beh']} x {m['model']} | jail b={m['beta']} + refine t3_sfull | "
              f"{SCENARIOS}x{args.rounds}x{TURNS} ===", flush=True)
        if _run(m, out, args.rounds, args.var_batch, args.eval_gpu, args.target_gpu, resume):
            ok_n += 1
        else:
            fail_n += 1
            if not args.keep_going:
                print(f"  ABORT: {m['beh']}/{m['model_dir']} failed.", flush=True); break
    print(f"\n== DONE: {ok_n}/{ok_n+fail_n} combo cells OK ==", flush=True)


if __name__ == "__main__":
    main()
