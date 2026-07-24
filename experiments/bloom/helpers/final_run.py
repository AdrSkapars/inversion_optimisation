#!/usr/bin/env python3
"""Final WILT paper-experiments driver (NOT the sweep; the sweep is done).

For each (behaviour x model) cell it runs two arms into a distinct results tree:

  1. BoN baseline  : <scenarios> x <bon_rounds> x <turns>, beta=0 (no jail model -> the
                     target_only fast path). This arm GENERATES a fresh shared kickoff bank
                     of <bon_rounds> rounds (understanding + ideation + per-round kickoffs).
  2. WILT / jail   : <scenarios> x <jail_rounds> x <turns> at the cell's pm3 beta, self-jail,
                     REUSING the bank's first <jail_rounds> rounds. Same scenarios, same
                     per-round kickoffs, same per-round seed as BoN's first <jail_rounds>
                     rounds -- only the steering differs (controlled comparison; BoN gets the
                     extra rounds for compute-fairness).

The kickoff bank (understanding + ideation + per-round kickoffs) is generated entirely by the
AUDITOR and, with between_rounds_strategise=False (each round a fresh resample), is fully
TARGET-INDEPENDENT. So it is shared PER BEHAVIOUR across all models: bank lives at
<runs-final>/<beh>/_bank, and every model for that behaviour is evaluated on the identical
adversarial inputs. Cells run model-major (the first model builds each behaviour's bank; the
rest reuse it), which also makes bank creation deterministic.

The chosen beta per cell is read from the completed sweep:
    experiments/bloom/runs_new/<beh>/<model>/param_selection.json  ->  picks.arith.pm3.beta
(the +-3 band pick, NOT +-5). A pm3 beta of 0.0 means the sweep selected NO steering within
+-3pp of BoN, so the chosen method for that cell IS the BoN run -- the jail arm is skipped
(logged), not run at beta 0.

Out-of-sample by construction: seed=100 (rounds seeded 101.., a DIFFERENT seed than the
sweep's 1) and 100 scenarios -- the chosen beta is evaluated on fresh draws it was not tuned on.

Writes arms to experiments/bloom/<runs-final>/<beh>/<model>/{bon, jail_b<beta>, cell.json} and
the shared bank to experiments/bloom/<runs-final>/<beh>/_bank.
Idempotent / resume-aware: an arm whose final round's judgment.json already exists is skipped.

Run on the GPU box from the repo root, venv active + HF env exported:
    python experiments/bloom/helpers/final_run.py --list
    python experiments/bloom/helpers/final_run.py --only self_harm/Qwen_Qwen3.5-4B   # checkpoint
    python experiments/bloom/helpers/final_run.py                                    # all cells
"""
import argparse, json, os, subprocess, sys, time, glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent              # experiments/bloom/helpers
RUNS_ROOT  = SCRIPT_DIR.parent                            # experiments/bloom  (runs live here)
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent              # repo root
BLOOM      = REPO_ROOT / "src" / "bloom" / "bloom_corrupt.py"
SWEEP_ROOT = RUNS_ROOT / "runs_new"                       # completed sweep (source of the pm3 betas)

# Fixed experiment constants (the user's final-paper spec).
SCENARIOS  = 100
SEED       = 100
TURNS      = 3
BON_ROUNDS = 8
JAIL_ROUNDS = 5

# Model-major execution order: the FIRST model builds each behaviour's shared bank; the rest
# reuse it. Qwen (the paper's representative slice) is the bank-builder. self_harm leads so the
# natural first cell is the checkpoint cell (Qwen x self_harm). Any model/behaviour not listed
# sorts last, alphabetically -- so the plan never silently drops a cell.
MODEL_ORDER = ["Qwen/Qwen3.5-4B", "google/gemma-4-e4b-it",
               "meta-llama/Llama-3.2-3B-Instruct", "microsoft/Phi-4-mini-instruct"]
BEH_ORDER   = ["self_harm", "medical", "selfpres", "deception", "delusions",
               "political", "racial", "reward_hacking", "goblin"]


# ------------------------------- cell discovery ------------------------------
def _cells(runs_final: str):
    """Every cell that has a completed sweep param_selection.json, in model-major order.
    Yields dict(beh, model, model_dir, behaviour_file, pm3_beta, auditor, base, bank).
    The bank is PER BEHAVIOUR (shared across models): <runs_final>/<beh>/_bank."""
    out = []
    for pj in sorted(glob.glob(str(SWEEP_ROOT / "*" / "*" / "param_selection.json"))):
        d = json.load(open(pj, encoding="utf-8"))
        pm3 = (((d.get("picks") or {}).get("arith") or {}).get("pm3")) or {}
        beta = pm3.get("beta")
        beh, model = d.get("behaviour"), d.get("model")
        if beta is None or not beh or not model:
            print(f"  [skip] {pj}: missing behaviour/model/pm3 beta", flush=True)
            continue
        model_dir = model.replace("/", "_")
        base = RUNS_ROOT / runs_final / beh / model_dir
        out.append({"beh": beh, "model": model, "model_dir": model_dir,
                    "behaviour_file": d.get("behaviour_file"), "pm3_beta": float(beta),
                    "auditor": d.get("auditor"), "base": base,
                    "bank": RUNS_ROOT / runs_final / beh / "_bank"})
    mi = {m: i for i, m in enumerate(MODEL_ORDER)}
    bi = {b: i for i, b in enumerate(BEH_ORDER)}
    out.sort(key=lambda c: (mi.get(c["model"], len(mi)), bi.get(c["beh"], len(bi)),
                            c["model"], c["beh"]))
    return out


# ------------------------------- one arm -------------------------------------
def _gpu_line(eval_gpu, target_gpu):
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"]).decode().strip()
        return " | ".join(l.strip() for l in q.splitlines())
    except Exception:
        return "nvidia-smi unavailable"


def _arm_done(out_dir: Path, rounds: int) -> bool:
    return (out_dir / f"round_{rounds}" / "judgment.json").exists()


def _run_arm(cell, arm, out_dir: Path, rounds: int, beta: float,
             eval_gpu: int, target_gpu: int, resume: bool) -> bool:
    """One bloom_corrupt run. arm in {'bon','jail'}. Returns True on success (or resume-skip)."""
    if resume and _arm_done(out_dir, rounds):
        print(f"  [{arm}] resume: round_{rounds}/judgment.json present -> skip", flush=True)
        return True
    env = dict(os.environ)
    env.update({
        "BLOOM_RUNS_ROOT":     str(RUNS_ROOT),
        "BLOOM_FOLDER":        str(out_dir.relative_to(RUNS_ROOT)),
        "BLOOM_TARGET_MODEL":  "local/" + cell["model"],
        "BLOOM_BEHAVIOR_FILE": cell["behaviour_file"],
        "BLOOM_EVAL_GPU":      str(eval_gpu),
        "BLOOM_TARGET_GPU":    str(target_gpu),
        "BLOOM_MAX_TURNS":     str(TURNS),
        "BLOOM_NUM_ROUNDS":    str(rounds),
        "BLOOM_NUM_SCENARIOS": str(SCENARIOS),
        "BLOOM_SEED":          str(SEED),
        "BLOOM_KICKOFF_BANK":  str(cell["bank"]),
    })
    if arm == "jail":                       # self-jail at the chosen beta (floor on by default)
        env["BLOOM_JAIL_MODEL"] = "local/" + cell["model"]
        env["BLOOM_JAIL_BETA"]  = str(beta)
    # bon: no BLOOM_JAIL_MODEL -> jailbroken_output.enabled stays False -> target_only BoN path.
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    print(f"  [{arm}] beta={beta:g} rounds={rounds} -> {out_dir}\n"
          f"        gpu[{_gpu_line(eval_gpu, target_gpu)}]  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, str(BLOOM)], cwd=str(REPO_ROOT),
                           env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = (r.returncode == 0) and _arm_done(out_dir, rounds)
    print(f"  [{arm}] {'OK' if ok else 'FAILED (see log)'}  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def _run_cell(cell, eval_gpu, target_gpu, resume) -> bool:
    beh, model, beta = cell["beh"], cell["model"], cell["pm3_beta"]
    print(f"\n=== CELL {beh} x {model}  |  pm3 beta={beta:g}  |  "
          f"BoN {SCENARIOS}x{BON_ROUNDS}x{TURNS}, jail {SCENARIOS}x{JAIL_ROUNDS}x{TURNS} ===", flush=True)
    # 1) BoN (generates the shared bank).
    if not _run_arm(cell, "bon", cell["base"] / "bon", BON_ROUNDS, 0.0,
                    eval_gpu, target_gpu, resume):
        print(f"  ABORT cell: BoN failed.", flush=True)
        return False
    # 2) jail at pm3 beta (reuses the bank). beta==0 -> no steering selected; skip.
    if beta == 0.0:
        print(f"  [jail] pm3 beta is 0 -> sweep selected NO steering; chosen method is the "
              f"8-round BoN itself. Skipping jail arm.", flush=True)
    else:
        jdir = cell["base"] / f"jail_b{beta:g}"   # matches the sweep's folder convention
        if not _run_arm(cell, "jail", jdir, JAIL_ROUNDS, beta,
                        eval_gpu, target_gpu, resume):
            print(f"  ABORT cell: jail failed.", flush=True)
            return False
    # provenance stamp per cell.
    try:
        json.dump({"behaviour": beh, "model": model, "pm3_beta": beta,
                   "scenarios": SCENARIOS, "seed": SEED, "turns": TURNS,
                   "bon_rounds": BON_ROUNDS, "jail_rounds": JAIL_ROUNDS,
                   "auditor": cell["auditor"],
                   "jail_skipped_beta0": (beta == 0.0)},
                  open(cell["base"] / "cell.json", "w", encoding="utf-8"), indent=2)
    except Exception:
        pass
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-final", default="runs_final", help="output tree under experiments/bloom (default runs_final)")
    ap.add_argument("--only", default=None, help="run a single cell '<behaviour>/<model_dir>' (e.g. self_harm/Qwen_Qwen3.5-4B)")
    ap.add_argument("--eval-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true", help="re-run arms even if their final judgment.json exists")
    ap.add_argument("--keep-going", action="store_true", help="continue to later cells after a cell fails (default: HALT on first failure -- we do not advance past a broken cell)")
    ap.add_argument("--list", action="store_true", help="print the cell plan (betas, arms) and exit")
    a = ap.parse_args()

    cells = _cells(a.runs_final)
    if a.only:
        want = a.only.strip("/")
        cells = [c for c in cells if f"{c['beh']}/{c['model_dir']}" == want]
        if not cells:
            sys.exit(f"--only {a.only!r} matched no cell. Use --list to see valid ids.")

    if a.list:
        print(f"{'#':>2}  {'behaviour/model':52}{'pm3':>5}   jail arm")
        for i, c in enumerate(cells, 1):
            jail = "SKIP (beta0 -> BoN)" if c["pm3_beta"] == 0.0 else f"jail_b{c['pm3_beta']:g}, {JAIL_ROUNDS} rounds"
            print(f"{i:>2}  {c['beh']+'/'+c['model_dir']:52}{c['pm3_beta']:>5g}   {jail}")
        print(f"\n{len(cells)} cells | BoN {SCENARIOS}x{BON_ROUNDS}x{TURNS} + jail {SCENARIOS}x{JAIL_ROUNDS}x{TURNS} "
              f"| seed {SEED} | out=experiments/bloom/{a.runs_final}")
        return

    resume = not a.no_resume
    print(f"== FINAL WILT run | {len(cells)} cell(s) | out=experiments/bloom/{a.runs_final} | "
          f"eval-gpu {a.eval_gpu} target-gpu {a.target_gpu} | resume={resume} | "
          f"halt-on-error={not a.keep_going} ==", flush=True)
    ok, bad = 0, []
    for c in cells:
        if _run_cell(c, a.eval_gpu, a.target_gpu, resume):
            ok += 1
        else:
            bad.append(f"{c['beh']}/{c['model_dir']}")
            if not a.keep_going:
                print(f"\n== HALT: cell {c['beh']}/{c['model_dir']} failed. Not advancing to later "
                      f"cells (fix it, then re-run -- resume skips completed arms). Use --keep-going "
                      f"to override. ==", flush=True)
                sys.exit(1)
    print(f"\n== DONE: {ok}/{len(cells)} cells complete ==", flush=True)
    if bad:
        print("   FAILED cells:", ", ".join(bad), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
