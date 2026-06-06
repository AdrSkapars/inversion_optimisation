"""PoE between output-generation and rephrasal distributions, both from target.

For each scenario, build two PoE distributions:
  - jail-variant: target(sys + user_input) + β · target("Rephrase: {raw_jail_output}")
  - poe-variant:  target(sys + user_input) + β · target("Rephrase: {raw_poe_output}")

Both sides of the PoE are the same Qwen target — the "rephrasal" side acts as
the guidance distribution biasing the natural output-generation distribution
toward producing content similar to the existing (jail or poe) output.

Sample n=5 candidates from each PoE distribution per scenario. Score each
under target with (sys, user_input) context. Take max per scenario. Compare
mean across 15 scenarios for jail-variant vs poe-variant.

Reuse the existing diag_three_outputs results.json (it has the raw outputs).
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
TARGET_UTIL  = 0.40       # target is the only model — give it more headroom

JAIL_KEY    = "jail"
POE_KEY     = "poe_beta2.0"
N_SAMPLES   = 5
MAX_TOKENS  = 250
TEMPERATURE = 1.0
TOP_P       = 1.0
POE_BETA    = 1.0
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
        cache_tag="diag_three_outputs_rephrase_poe", label="(poe-reph)",
    )

    # ── Build prefixes ──────────────────────────────────────────────────────
    # Target-output-gen prefix (one per scenario, same for both variants).
    def build_output_gen_prefix(sc) -> List[int]:
        msgs = []
        if sc.get("sys_prompt"):
            msgs.append({"role": "system", "content": sc["sys_prompt"]})
        msgs.append({"role": "user", "content": sc["input"]})
        s = lm_target.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        return lm_target.tokenizer.encode(s, add_special_tokens=False)

    # Rephrase-task prefix for a given output text.
    def build_rephrase_prefix(text: str) -> List[int]:
        msgs = [{"role": "user", "content": f"Rephrase the following response: {text}"}]
        s = lm_target.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        return lm_target.tokenizer.encode(s, add_special_tokens=False)

    output_gen_prefixes = [build_output_gen_prefix(sc) for sc in scenarios]
    jail_reph_prefixes  = [build_rephrase_prefix(sc["outputs"][JAIL_KEY]) for sc in scenarios]
    poe_reph_prefixes   = [build_rephrase_prefix(sc["outputs"][POE_KEY])  for sc in scenarios]
    print(f"[{time.time()-t0:.0f}s] Built {P} output-gen, {P} jail-reph, {P} poe-reph prefixes", flush=True)

    # ── PoE generation: 2 calls ─────────────────────────────────────────────
    print(f"\n[{time.time()-t0:.0f}s] PoE β={POE_BETA} jail-variant ({P}×{N_SAMPLES} streams)...", flush=True)
    jail_out_3d = _contrastive_sample_extensions(
        lm_target=lm_target, lm_jail=lm_target,
        target_prefixes=output_gen_prefixes,
        jail_prefixes=jail_reph_prefixes,
        n=N_SAMPLES, max_tokens=MAX_TOKENS,
        beta=POE_BETA, top_k_logprobs=POE_TOP_K,
        temperature=TEMPERATURE, top_p=TOP_P,
        allowed_token_ids=latin_ids,
        ignore_eos=False, eos_token_id=eos,
    )
    print(f"[{time.time()-t0:.0f}s] jail-variant done.", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] PoE β={POE_BETA} poe-variant ({P}×{N_SAMPLES} streams)...", flush=True)
    poe_out_3d = _contrastive_sample_extensions(
        lm_target=lm_target, lm_jail=lm_target,
        target_prefixes=output_gen_prefixes,
        jail_prefixes=poe_reph_prefixes,
        n=N_SAMPLES, max_tokens=MAX_TOKENS,
        beta=POE_BETA, top_k_logprobs=POE_TOP_K,
        temperature=TEMPERATURE, top_p=TOP_P,
        allowed_token_ids=latin_ids,
        ignore_eos=False, eos_token_id=eos,
    )
    print(f"[{time.time()-t0:.0f}s] poe-variant done.", flush=True)

    # ── Decode + score under (sys, user_input) ──────────────────────────────
    items = []
    idx = []  # (variant, scenario_idx, sample_idx)
    decoded: Dict[str, List[List[str]]] = {"jail": [], "poe": []}

    for variant, out_3d in [("jail", jail_out_3d), ("poe", poe_out_3d)]:
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
                idx.append((variant, s_idx, r_idx))
            decoded[variant].append(sample_texts)

    print(f"\n[{time.time()-t0:.0f}s] Scoring {len(items)} items under target...", flush=True)
    scores: List = []
    for b in range(0, len(items), 16):
        scores.extend(batch_logprob_local(lm_target, items[b:b+16]))
    print(f"[{time.time()-t0:.0f}s] scoring done.", flush=True)

    # ── Pick best-of-5 per variant per scenario ─────────────────────────────
    best_per_variant: Dict[str, List[Dict]] = {"jail": [], "poe": []}
    for variant in ("jail", "poe"):
        for s_idx, sc in enumerate(scenarios):
            lps = [scores[i] for i, (vv, ss, _) in enumerate(idx) if vv == variant and ss == s_idx]
            texts = decoded[variant][s_idx]
            valid = [(lp, t) for lp, t in zip(lps, texts) if lp is not None]
            if not valid:
                best_per_variant[variant].append({"variation_number": sc["variation_number"], "best_lp": None})
                continue
            best_lp, best_text = max(valid, key=lambda x: x[0])
            best_per_variant[variant].append({
                "variation_number":   sc["variation_number"],
                "best_lp":            best_lp,
                "best_per_token_p":   math.exp(best_lp) * 100,
                "best_text":          best_text,
                "all_lps":            [lp for lp in lps if lp is not None],
                "all_samples":        texts,
            })

    # ── Save into results.json ──────────────────────────────────────────────
    for s_idx, sc in enumerate(scenarios):
        sc[f"poe_rephrasal_best_of_{N_SAMPLES}_b{POE_BETA}"] = {
            "jail_variant": best_per_variant["jail"][s_idx],
            "poe_variant":  best_per_variant["poe"][s_idx],
        }
    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\n[{time.time()-t0:.0f}s] Saved → {RESULTS_PATH}", flush=True)

    # ── Stats ───────────────────────────────────────────────────────────────
    jail_ps = [b["best_per_token_p"] for b in best_per_variant["jail"]
               if b.get("best_per_token_p") is not None]
    poe_ps = [b["best_per_token_p"] for b in best_per_variant["poe"]
              if b.get("best_per_token_p") is not None]

    def stats(name, ps):
        if not ps: print(f"  {name:<40} n=0"); return
        mean = sum(ps)/len(ps); med = sorted(ps)[len(ps)//2]
        print(f"  {name:<40} n={len(ps)}  mean P={mean:>7.3f}%  median P={med:>7.3f}%  min={min(ps):>6.3f}%  max={max(ps):>6.3f}%")

    print(f"\n=== SUMMARY (per-token P under target | sys, user_input) ===")
    print(f"For comparison from prior rounds:")
    stats("  jail (raw output)",            [sc["scores"]["jail"]["per_token_p_pct"] for sc in scenarios])
    stats("  jail best-of-5 rephrasal",     [sc[f"rephrasals_best_of_5"]["jail"]["best_per_token_p"] for sc in scenarios])
    stats("  poe (raw output)",             [sc["scores"][POE_KEY]["per_token_p_pct"] for sc in scenarios])
    stats("  poe best-of-5 rephrasal",      [sc[f"rephrasals_best_of_5"][POE_KEY]["best_per_token_p"] for sc in scenarios])
    print(f"\nNew (PoE between output-gen and rephrasal-task distributions, β={POE_BETA}):")
    stats(f"  jail-variant best-of-{N_SAMPLES}", jail_ps)
    stats(f"  poe-variant  best-of-{N_SAMPLES}", poe_ps)

    print(f"\n=== PER-SCENARIO ===")
    print(f"{'v':>3}  {'jail-var %':>11}  {'poe-var %':>10}  {'diff':>7}")
    for s_idx, sc in enumerate(scenarios):
        v = sc["variation_number"]
        j = best_per_variant["jail"][s_idx].get("best_per_token_p")
        p = best_per_variant["poe"][s_idx].get("best_per_token_p")
        diff = (p - j) if (j is not None and p is not None) else None
        def fmt(x): return f"{x:>10.3f}%" if x is not None else "       ---"
        print(f"{v:>3}  {fmt(j):>11}  {fmt(p):>10}  {diff:>+6.2f}pp" if diff is not None else f"{v:>3}  {fmt(j)}  {fmt(p)}  ---")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
