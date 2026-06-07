"""Diagnostic step 1: sample 5 fresh target responses to the offensive query.

Same (sys, user) context as the loan-discrimination example we've been working
on. Target samples freely (no jail, no rephrase). For each sample: report the
text + per-token avg logprob under target (== logprob it just produced).

Goal: see what target naturally outputs when asked, both in content (is it
offensive on its own?) and in probability (calibrate against the rephrase
results — what's a 'natural' lp for this prompt?).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
from bloom_beast_logitdiff import _get_local_model  # type: ignore

SRC_RUN     = SCRIPT_DIR.parent / "runs_14" / "1r3t_bon25io_judgeloss" / "round_1"
OUT_PATH    = SCRIPT_DIR / "diag_natural_results.json"

TARGET_MODEL = "Qwen/Qwen3-4B"
GPU_ID       = 0
TARGET_UTIL  = 0.25

N_SAMPLES    = 5
TEMP         = 1.0
MAX_GEN_TOK  = 400
NO_THINK_SUFFIX = "<think>\n\n</think>\n"


def load_example() -> Dict:
    """Use the same TRS-source example as previous diags."""
    j = json.load(open(SRC_RUN / "judgment.json", encoding="utf-8"))
    judgments = sorted(j.get("judgments", []),
                       key=lambda e: -(e.get("behavior_presence", 0) or 0))
    for entry in judgments:
        v = entry["variation_number"]
        tf = SRC_RUN / "transcripts" / f"transcript_v{v}r1.json"
        if not tf.exists():
            continue
        td = json.load(open(tf, encoding="utf-8"))
        msgs = td.get("messages", [])
        sys_msg  = next((m for m in msgs if m["role"] == "system"), {"content": ""})
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        if user_msg is None:
            continue
        return {
            "scenario_file":     tf.name,
            "behavior_presence": entry.get("behavior_presence"),
            "sys_prompt":        sys_msg.get("content", "") or "",
            "user_msg":          user_msg.get("content", "") or "",
            "original_trs":      user_msg.get("targeted_response_start", "") or "",
        }
    raise RuntimeError("No example found")


def sample_n(lm, sys_prompt: str, user_msg: str, n: int) -> List[str]:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg}]
    full = lm.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
    prefix_ids = lm.tokenizer.encode(full, add_special_tokens=False)
    out_lists = lm.worker.generate_n_tokens(
        [prefix_ids],
        dict(max_tokens=MAX_GEN_TOK, temperature=TEMP, top_p=1.0,
             skip_special_tokens=False, n=n),
    )[0]
    return [lm.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in out_lists]


def score_under_target(lm, sys_prompt: str, user_msg: str, output: str) -> Optional[float]:
    ctx_msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_msg}]
    ctx_str = lm.tokenizer.apply_chat_template(
        ctx_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
    ctx_ids    = lm.tokenizer.encode(ctx_str, add_special_tokens=False)
    target_ids = lm.tokenizer.encode(output, add_special_tokens=False)
    if not target_ids:
        return None
    full_ids = ctx_ids + target_ids
    return lm.worker.compute_target_logprobs(
        [(full_ids, len(ctx_ids), len(target_ids))],
    )[0]


def main():
    t0 = time.time()
    ex = load_example()
    print(f"[{time.time()-t0:.0f}s] Example: {ex['scenario_file']} (BP={ex['behavior_presence']})", flush=True)
    print(f"  sys ({len(ex['sys_prompt'])} chars): {ex['sys_prompt'][:120]}...", flush=True)
    print(f"  user: {ex['user_msg']}", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Loading target...", flush=True)
    lm = _get_local_model(TARGET_MODEL, gpu_id=GPU_ID,
                          gpu_memory_utilization=TARGET_UTIL)

    print(f"\n[{time.time()-t0:.0f}s] Sampling {N_SAMPLES} responses (T={TEMP})...", flush=True)
    samples = sample_n(lm, ex["sys_prompt"], ex["user_msg"], N_SAMPLES)

    rows: List[Dict] = []
    for idx, text in enumerate(samples):
        lp = score_under_target(lm, ex["sys_prompt"], ex["user_msg"], text)
        print(f"\n--- sample #{idx} (lp={lp:.4f}, len={len(text)}) ---", flush=True)
        print(text, flush=True)
        rows.append({"idx": idx, "text": text, "lp_target": lp})

    summary = {
        "scenario": ex["scenario_file"],
        "behavior_presence_original": ex["behavior_presence"],
        "n_samples": len(rows),
        "lps": [r["lp_target"] for r in rows],
    }
    print(f"\n=== SUMMARY ===")
    print(f"  all lps: {[round(x, 3) for x in summary['lps']]}")

    json.dump({"summary": summary, "example": ex, "samples": rows},
              open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
