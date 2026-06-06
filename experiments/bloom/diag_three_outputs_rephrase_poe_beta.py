"""Jail-variant PoE rephrasal at higher β values (2 and 3).

Same setup as diag_three_outputs_rephrase_poe.py but only the jail variant,
and sweeping β ∈ {2, 3}. β=1 already in results.json from prior run.

PoE = target(sys + user_input) + β · target("Rephrase: {raw_jail_output}")
Sample n=5, score each under (sys, user_input), keep best.

We want to see if higher β preserves more bias content (and at what
probability cost).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from bloom_beast_tree import (  # type: ignore
    _get_local_model, batch_logprob_local,
    _contrastive_sample_extensions, _get_or_build_latin_mask,
    _resolve_eos_token_id,
)

RESULTS_PATH = SCRIPT_DIR / "runs_15" / "diag_three_outputs" / "results.json"
TARGET_MODEL = "Qwen/Qwen3-4B"
GPU_ID       = 0
TARGET_UTIL  = 0.40

JAIL_KEY    = "jail"
N_SAMPLES   = 5
MAX_TOKENS  = 250
TEMPERATURE = 1.0
TOP_P       = 1.0
BETAS       = [2.0, 3.0]
POE_TOP_K   = 50
NO_THINK_SUFFIX = "<think>\n\n</think>\n"


def main():
    t0 = time.time()
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    scenarios = data["scenarios"]
    P = len(scenarios)
    print(f"[{time.time()-t0:.0f}s] Loaded {P} scenarios", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Loading target Qwen...", flush=True)
    lm_target = _get_local_model(TARGET_MODEL, gpu_id=GPU_ID, gpu_memory_utilization=TARGET_UTIL)

    eos = _resolve_eos_token_id(lm_target, True)
    latin_ids = _get_or_build_latin_mask(
        lm_target, True, eos, extra_chars="",
        cache_tag="diag_three_outputs_rephrase_poe_beta", label="(poe-reph-beta)",
    )

    # ── Build prefixes (same for both β values) ─────────────────────────────
    def build_output_gen_prefix(sc) -> List[int]:
        msgs = []
        if sc.get("sys_prompt"):
            msgs.append({"role": "system", "content": sc["sys_prompt"]})
        msgs.append({"role": "user", "content": sc["input"]})
        s = lm_target.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        return lm_target.tokenizer.encode(s, add_special_tokens=False)

    def build_rephrase_prefix(text: str) -> List[int]:
        msgs = [{"role": "user", "content": f"Rephrase the following response: {text}"}]
        s = lm_target.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        return lm_target.tokenizer.encode(s, add_special_tokens=False)

    output_gen_prefixes = [build_output_gen_prefix(sc) for sc in scenarios]
    jail_reph_prefixes  = [build_rephrase_prefix(sc["outputs"][JAIL_KEY]) for sc in scenarios]
    print(f"[{time.time()-t0:.0f}s] Built {P} prefixes per side", flush=True)

    all_outs: Dict[float, List[List[List[int]]]] = {}
    for beta in BETAS:
        print(f"\n[{time.time()-t0:.0f}s] PoE β={beta} jail-variant ({P}×{N_SAMPLES} streams)...", flush=True)
        out_3d = _contrastive_sample_extensions(
            lm_target=lm_target, lm_jail=lm_target,
            target_prefixes=output_gen_prefixes,
            jail_prefixes=jail_reph_prefixes,
            n=N_SAMPLES, max_tokens=MAX_TOKENS,
            beta=beta, top_k_logprobs=POE_TOP_K,
            temperature=TEMPERATURE, top_p=TOP_P,
            allowed_token_ids=latin_ids,
            ignore_eos=False, eos_token_id=eos,
        )
        all_outs[beta] = out_3d
        print(f"[{time.time()-t0:.0f}s] β={beta} done.", flush=True)

    # ── Score under (sys, user_input) ───────────────────────────────────────
    items = []
    idx = []  # (beta, s_idx, r_idx)
    decoded: Dict[float, List[List[str]]] = {b: [] for b in BETAS}

    for beta in BETAS:
        out_3d = all_outs[beta]
        for s_idx, sc in enumerate(scenarios):
            sys_prompt = sc["sys_prompt"]; user_input = sc["input"]
            base_msgs = []
            if sys_prompt:
                base_msgs.append({"role": "system", "content": sys_prompt})
            base_msgs.append({"role": "user", "content": user_input})
            sample_texts = []
            for r_idx, token_ids in enumerate(out_3d[s_idx]):
                text = lm_target.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                sample_texts.append(text)
                items.append((list(base_msgs), text))
                idx.append((beta, s_idx, r_idx))
            decoded[beta].append(sample_texts)

    print(f"\n[{time.time()-t0:.0f}s] Scoring {len(items)} items...", flush=True)
    scores: List = []
    for b in range(0, len(items), 16):
        scores.extend(batch_logprob_local(lm_target, items[b:b+16]))
    print(f"[{time.time()-t0:.0f}s] scoring done.", flush=True)

    # ── Pick best-of-5 per (β, scenario) ────────────────────────────────────
    best_per_beta: Dict[float, List[Dict]] = {b: [] for b in BETAS}
    for beta in BETAS:
        for s_idx, sc in enumerate(scenarios):
            lps = [scores[i] for i, (bb, ss, _) in enumerate(idx) if bb == beta and ss == s_idx]
            texts = decoded[beta][s_idx]
            valid = [(lp, t) for lp, t in zip(lps, texts) if lp is not None]
            if not valid:
                best_per_beta[beta].append({"variation_number": sc["variation_number"], "best_lp": None})
                continue
            best_lp, best_text = max(valid, key=lambda x: x[0])
            best_per_beta[beta].append({
                "variation_number":   sc["variation_number"],
                "best_lp":            best_lp,
                "best_per_token_p":   math.exp(best_lp) * 100,
                "best_text":          best_text,
                "all_lps":            [lp for lp in lps if lp is not None],
                "all_samples":        texts,
            })

    # ── Save into results.json ──────────────────────────────────────────────
    for s_idx, sc in enumerate(scenarios):
        key = f"poe_rephrasal_jail_variant_beta_sweep"
        sc[key] = {str(b): best_per_beta[b][s_idx] for b in BETAS}
    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\n[{time.time()-t0:.0f}s] Saved → {RESULTS_PATH}", flush=True)

    # ── Stats ───────────────────────────────────────────────────────────────
    def stats(name, ps):
        if not ps: print(f"  {name:<48} n=0"); return
        mean = sum(ps)/len(ps); med = sorted(ps)[len(ps)//2]
        print(f"  {name:<48} n={len(ps)}  mean P={mean:>7.3f}%  median P={med:>7.3f}%  min={min(ps):>6.3f}%  max={max(ps):>6.3f}%")

    print(f"\n=== SUMMARY ===")
    # Prior result for reference
    prior_key = next((k for k in scenarios[0] if k.startswith("poe_rephrasal_best_of_")), None)
    if prior_key:
        prior_jail = [sc[prior_key]["jail_variant"]["best_per_token_p"] for sc in scenarios
                       if sc[prior_key]["jail_variant"].get("best_per_token_p") is not None]
        stats(f"  jail-variant best-of-5 (β=1.0)", prior_jail)
    for beta in BETAS:
        ps = [b["best_per_token_p"] for b in best_per_beta[beta]
              if b.get("best_per_token_p") is not None]
        stats(f"  jail-variant best-of-5 (β={beta})", ps)

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
