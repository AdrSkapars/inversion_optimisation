"""Diagnostic: compare per-token avg logprob under target model of three candidate
assistant continuations for the same (system, user) context.

Three candidates per scenario:
  (a) target_out   — sampled fresh from target model (Qwen3-4B), normal context
  (b) jail_out     — sampled fresh from jail model (Huihui-Qwen3-4B-abliterated),
                     jail's system prompt + prefill (mirrors how it's used in PoE)
  (c) eval_trs     — eval model's pre-generated targeted_response_start, loaded
                     from a past 1r3t_bon25io_* run

All three scored as average log P(output | sys, user) under target with no-think
chat template (matches generation-time prompt format).

Hypothesis: target_out ≈ best, jail_out close behind (same base architecture),
eval_trs far worse (different model family). If true → motivates jail-as-TRS for
BEAST search instead of eval-as-TRS.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from bloom_beast_logitdiff import (  # type: ignore
    _get_local_model, batch_generate_local, parse_message, _make_local_response,
)

# ── Config ────────────────────────────────────────────────────────────────
SRC_RUN     = SCRIPT_DIR / "runs_14" / "1r3t_bon25io_judgeloss" / "round_1"
OUT_PATH    = SCRIPT_DIR / "diag_trs_results.json"
N_INPUTS    = 8                                # scenarios to test (range 5–10)
TARGET_MODEL = "Qwen/Qwen3-4B"
JAIL_MODEL   = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
TARGET_GPU   = 0
JAIL_GPU     = 1
TARGET_UTIL  = 0.45
JAIL_UTIL    = 0.45
MAX_GEN_TOK  = 200
TEMP         = 1.0
NO_THINK_SUFFIX = "<think>\n\n</think>\n"


# ── Step 1: load (sys, user, eval_TRS) from past run ──────────────────────
def load_inputs(n: int) -> List[Dict]:
    items: List[Dict] = []
    for tf in sorted((SRC_RUN / "transcripts").glob("transcript_v*r1.json")):
        td = json.load(open(tf, encoding="utf-8"))
        msgs = td.get("messages", [])
        sys_msg = next((m for m in msgs if m["role"] == "system"), None)
        # First (turn-0) user message and its eval TRS.
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        if user_msg is None or not user_msg.get("targeted_response_start"):
            continue
        items.append({
            "scenario_file": tf.name,
            "sys_prompt":    (sys_msg or {}).get("content", "") or "",
            "user_msg":      user_msg.get("content", "") or "",
            "eval_trs":      user_msg["targeted_response_start"],
        })
        if len(items) >= n:
            break
    return items


# ── Step 2: generate fresh target/jail outputs ────────────────────────────
def gen_target(lm, sys_prompt: str, user_msg: str) -> str:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg}]
    raw = batch_generate_local(lm, [msgs], MAX_GEN_TOK, TEMP, no_think=True)[0]
    parsed = parse_message(_make_local_response(raw))
    return parsed["content"] or raw


def gen_jail(lm_jail, jail_sys: str, prefill: str, user_msg: str) -> str:
    """Mirror batch_generate_contrastive_local's jail-side prefix construction."""
    j_msgs = [{"role": "system", "content": jail_sys},
              {"role": "user",   "content": user_msg}]
    j_prompt = lm_jail.tokenizer.apply_chat_template(
        j_msgs, tokenize=False, add_generation_prompt=True,
    )
    j_prompt += NO_THINK_SUFFIX
    if prefill:
        j_prompt += prefill
    # Use the worker's raw generation against this hand-built prompt.
    j_prefix_ids = lm_jail.tokenizer.encode(j_prompt, add_special_tokens=False)
    out_ids = lm_jail.worker.generate_n_tokens(
        [j_prefix_ids],
        dict(max_tokens=MAX_GEN_TOK, temperature=TEMP, top_p=1.0,
             skip_special_tokens=False),
    )[0][0]
    text = lm_jail.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    # If prefill was used, drop it from the front so we score only the sampled portion.
    if prefill and text.startswith(prefill.strip()):
        text = text[len(prefill.strip()):].lstrip()
    return text


# ── Step 3: score all three under target ──────────────────────────────────
def score_under_target(lm_target, sys_prompt: str, user_msg: str, output: str) -> float | None:
    """Per-token avg logP(output | sys, user) under target with no-think chat template."""
    ctx_msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_msg}]
    ctx_str = lm_target.tokenizer.apply_chat_template(
        ctx_msgs, tokenize=False, add_generation_prompt=True,
    ) + NO_THINK_SUFFIX
    ctx_ids    = lm_target.tokenizer.encode(ctx_str, add_special_tokens=False)
    target_ids = lm_target.tokenizer.encode(output,  add_special_tokens=False)
    if not target_ids:
        return None
    full_ids = ctx_ids + target_ids
    res = lm_target.worker.compute_target_logprobs(
        [(full_ids, len(ctx_ids), len(target_ids))],
    )
    return res[0]


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    inputs = load_inputs(N_INPUTS)
    print(f"Loaded {len(inputs)} inputs from {SRC_RUN.name}", flush=True)

    prompts = yaml.safe_load(open(SCRIPT_DIR / "prompts.yaml", encoding="utf-8"))
    jail_sys = prompts.get("jailbroken_output_system_prompt", "")
    if not jail_sys:
        raise RuntimeError("prompts.yaml missing 'jailbroken_output_system_prompt'")
    jail_prefill = "As a racist:\n"  # matches bloom_beast_logitdiff cfg default

    print(f"\n[{time.time()-t0:.0f}s] Loading target on GPU {TARGET_GPU}: {TARGET_MODEL}", flush=True)
    lm_target = _get_local_model(TARGET_MODEL, gpu_id=TARGET_GPU,
                                 gpu_memory_utilization=TARGET_UTIL)
    print(f"[{time.time()-t0:.0f}s] Loading jail on GPU {JAIL_GPU}: {JAIL_MODEL}", flush=True)
    lm_jail = _get_local_model(JAIL_MODEL, gpu_id=JAIL_GPU,
                               gpu_memory_utilization=JAIL_UTIL)

    rows: List[Dict] = []
    for i, inp in enumerate(inputs):
        print(f"\n[{time.time()-t0:.0f}s] === scenario {i+1}/{len(inputs)} ({inp['scenario_file']}) ===", flush=True)
        target_out = gen_target(lm_target, inp["sys_prompt"], inp["user_msg"])
        jail_out   = gen_jail(lm_jail, jail_sys, jail_prefill, inp["user_msg"])
        eval_out   = inp["eval_trs"]

        lp_target = score_under_target(lm_target, inp["sys_prompt"], inp["user_msg"], target_out)
        lp_jail   = score_under_target(lm_target, inp["sys_prompt"], inp["user_msg"], jail_out)
        lp_eval   = score_under_target(lm_target, inp["sys_prompt"], inp["user_msg"], eval_out)
        print(f"  target avg_lp = {lp_target:.4f}   ({len(target_out)} chars)", flush=True)
        print(f"  jail   avg_lp = {lp_jail:.4f}     ({len(jail_out)} chars)", flush=True)
        print(f"  eval   avg_lp = {lp_eval:.4f}     ({len(eval_out)} chars)", flush=True)

        rows.append({
            "scenario_file": inp["scenario_file"],
            "user_msg_preview": inp["user_msg"][:120],
            "target_out": target_out,
            "jail_out":   jail_out,
            "eval_out":   eval_out,
            "lp_target":  lp_target,
            "lp_jail":    lp_jail,
            "lp_eval":    lp_eval,
        })

    # Aggregate.
    def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
    summary = {
        "n":              len(rows),
        "mean_lp_target": mean([r["lp_target"] for r in rows]),
        "mean_lp_jail":   mean([r["lp_jail"]   for r in rows]),
        "mean_lp_eval":   mean([r["lp_eval"]   for r in rows]),
    }
    print(f"\n=== SUMMARY (n={summary['n']}) ===")
    print(f"  mean lp_target = {summary['mean_lp_target']:.4f}")
    print(f"  mean lp_jail   = {summary['mean_lp_jail']:.4f}")
    print(f"  mean lp_eval   = {summary['mean_lp_eval']:.4f}")

    json.dump({"summary": summary, "rows": rows}, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved → {OUT_PATH}")

    import os
    os._exit(0)  # bypass vLLM hang


if __name__ == "__main__":
    main()
