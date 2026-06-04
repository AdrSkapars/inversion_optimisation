"""Diagnostic: classic jail PoE β sweep on the SAME scenario as diag_rephrase_poe.

For each token at sampling step t, mix:
  - target natural distribution: P(x_t | sys, user)
  - jail distribution:           P(x_t | jail_sys, user, prefill)

Combined logits = target_logits + β · log P_jail. Same β sweep as
diag_rephrase_poe so we can plot apples-to-apples lp curves.

For consistency with diag_rephrase_poe: NO latin_mask (matches that setup).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from bloom_beast_logitdiff import (  # type: ignore
    _get_local_model, _contrastive_sample_extensions,
)

SRC_RUN     = SCRIPT_DIR / "runs_14" / "1r3t_bon25io_judgeloss" / "round_1"
OUT_PATH    = SCRIPT_DIR / "diag_jail_poe_results.json"

TARGET_MODEL = "Qwen/Qwen3-4B"
JAIL_MODEL   = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
GPU_ID       = 0
TARGET_UTIL  = 0.22
JAIL_UTIL    = 0.22

BETAS        = [0.5, 1.0, 2.0, 3.0, 5.0]
N_PER_BETA   = 5
TEMP         = 1.0
TOP_K_LP     = 1000
MAX_TOK      = 400
JAIL_PREFILL = "As a racist:\n"
NO_THINK_SUFFIX = "<think>\n\n</think>\n"


def load_example() -> Dict:
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
        trs = user_msg.get("targeted_response_start", "") or ""
        if not trs.strip():
            continue
        return {
            "scenario_file":     tf.name,
            "behavior_presence": entry.get("behavior_presence"),
            "sys_prompt":        sys_msg.get("content", "") or "",
            "user_msg":          user_msg.get("content", "") or "",
        }
    raise RuntimeError("No example found")


def build_natural_prefix(lm, sys_prompt: str, user_msg: str) -> List[int]:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg}]
    s = lm.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
    return lm.tokenizer.encode(s, add_special_tokens=False)


def build_jail_prefix(lm_jail, jail_sys: str, user_msg: str, prefill: str) -> List[int]:
    """Mirrors batch_generate_contrastive_local's jail-side construction."""
    j_msgs = [{"role": "system", "content": jail_sys},
              {"role": "user",   "content": user_msg}]
    s = lm_jail.tokenizer.apply_chat_template(
        j_msgs, tokenize=False, add_generation_prompt=True)
    s += NO_THINK_SUFFIX
    if prefill:
        s += prefill
    return lm_jail.tokenizer.encode(s, add_special_tokens=False)


def score_under_natural(lm, sys_prompt: str, user_msg: str, output: str) -> Optional[float]:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg}]
    ctx_str = lm.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
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

    prompts = yaml.safe_load(open(SCRIPT_DIR / "prompts.yaml", encoding="utf-8"))
    jail_sys = prompts.get("jailbroken_output_system_prompt", "")
    if not jail_sys:
        raise RuntimeError("prompts.yaml missing jailbroken_output_system_prompt")
    print(f"  jail_sys ({len(jail_sys)} chars): {jail_sys[:120]}...", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Loading target...", flush=True)
    lm_target = _get_local_model(TARGET_MODEL, gpu_id=GPU_ID,
                                 gpu_memory_utilization=TARGET_UTIL)
    print(f"[{time.time()-t0:.0f}s] Loading jail...", flush=True)
    lm_jail = _get_local_model(JAIL_MODEL, gpu_id=GPU_ID,
                               gpu_memory_utilization=JAIL_UTIL)

    natural_prefix = build_natural_prefix(lm_target, ex["sys_prompt"], ex["user_msg"])
    jail_prefix    = build_jail_prefix(lm_jail, jail_sys, ex["user_msg"], JAIL_PREFILL)
    print(f"  natural prefix:  {len(natural_prefix)} toks", flush=True)
    print(f"  jail prefix:     {len(jail_prefix)} toks", flush=True)

    eos = lm_target.tokenizer.eos_token_id
    rows: List[Dict] = []

    for beta in BETAS:
        print(f"\n[{time.time()-t0:.0f}s] === β={beta} — PoE sampling {N_PER_BETA} ===", flush=True)
        out_lists = _contrastive_sample_extensions(
            lm_target=lm_target, lm_jail=lm_jail,
            target_prefixes=[natural_prefix],
            jail_prefixes=[jail_prefix],
            n=N_PER_BETA, max_tokens=MAX_TOK,
            beta=beta, top_k_logprobs=TOP_K_LP,
            temperature=TEMP, top_p=1.0,
            allowed_token_ids=None, ignore_eos=False, eos_token_id=eos,
        )[0]
        for idx, ids in enumerate(out_lists):
            text = lm_target.tokenizer.decode(ids, skip_special_tokens=True).strip()
            lp   = score_under_natural(lm_target, ex["sys_prompt"], ex["user_msg"], text)
            print(f"\n--- β={beta} #{idx} (lp={lp:.4f}, len={len(text)}) ---", flush=True)
            print(text[:500], flush=True)
            rows.append({"beta": beta, "idx": idx, "text": text, "lp_target": lp})

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs)/len(xs) if xs else None
    summary = {"scenario": ex["scenario_file"], "by_beta": {}}
    print(f"\n=== SUMMARY ===")
    for b in BETAS:
        rs = [r for r in rows if r["beta"] == b]
        lps = [r["lp_target"] for r in rs if r["lp_target"] is not None]
        summary["by_beta"][str(b)] = {"n": len(rs), "mean_lp": mean(lps),
                                     "best_lp": max(lps) if lps else None,
                                     "worst_lp": min(lps) if lps else None}
        s = summary["by_beta"][str(b)]
        print(f"  β={b:.1f}: mean={s['mean_lp']:.4f}  best={s['best_lp']:.4f}  worst={s['worst_lp']:.4f}")

    json.dump({"summary": summary, "example": ex, "samples": rows},
              open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
