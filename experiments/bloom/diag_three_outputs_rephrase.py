"""Rephrase the poe outputs 5 times each, keep highest-prob rephrasal under target.

For each of the 15 scenarios in diag_three_outputs/results.json:
  - Take the poe output text
  - Build rephrase prompt: user-msg only, no system, no original input:
        "Rephrase the following response: {poe_output}"
  - Sample 5 rephrasals from target Qwen
  - Score each as P(rephrasal | sys_prompt, user_input) under target  (the
    "could target have plausibly said this in response to the original input"
    metric)
  - Keep highest-prob rephrasal

Compare mean per-token P across 15 scenarios:
  - jail raw      (already in results.json)
  - poe-best-rephrasal-of-5  (new)

Fast: 15 × 5 = 75 generations + 75 scoring forward passes, all batched.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from bloom_beast_tree import (  # type: ignore
    _get_local_model, batch_generate_local, batch_logprob_local,
)

RESULTS_PATH = SCRIPT_DIR / "runs_15" / "diag_three_outputs" / "results.json"
TARGET_MODEL = "Qwen/Qwen3-4B"
GPU_ID       = 0
TARGET_UTIL  = 0.30

POE_KEY        = "poe_beta2.0"
N_REPHRASE     = 5
MAX_TOKENS     = 250
TEMPERATURE    = 1.0
NO_THINK = True


def main():
    t0 = time.time()
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    scenarios = data["scenarios"]
    print(f"[{time.time()-t0:.0f}s] Loaded {len(scenarios)} scenarios", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Loading target Qwen...", flush=True)
    lm_target = _get_local_model(TARGET_MODEL, gpu_id=GPU_ID, gpu_memory_utilization=TARGET_UTIL)

    # ── 1) Build rephrase prompts (one per scenario) ────────────────────────
    rephrase_msgs_list: List[List[Dict]] = []
    for sc in scenarios:
        poe_out = sc["outputs"][POE_KEY]
        rephrase_msgs_list.append([
            {"role": "user", "content": f"Rephrase the following response: {poe_out}"},
        ])

    # ── 2) Batched generation: one call returning n=5 candidates per scenario ─
    # batch_generate_local doesn't support n>1, so use worker.generate_n_tokens
    # which takes token-id prefixes and returns n candidates per prefix.
    prefixes = []
    for msgs in rephrase_msgs_list:
        s = lm_target.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        if NO_THINK:
            s += "<think>\n\n</think>\n"
        prefixes.append(lm_target.tokenizer.encode(s, add_special_tokens=False))

    print(f"\n[{time.time()-t0:.0f}s] Sampling {N_REPHRASE} rephrasals × {len(prefixes)} scenarios in one batched call...", flush=True)
    out_lists = lm_target.worker.generate_n_tokens(
        prefixes,
        dict(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=1.0,
             skip_special_tokens=False, n=N_REPHRASE),
    )
    # Shape: [P scenarios] × [N rephrasals] × [token-id list]
    rephrasals: List[List[str]] = []
    for i, scenario_outs in enumerate(out_lists):
        decoded = [lm_target.tokenizer.decode(ids, skip_special_tokens=True).strip()
                   for ids in scenario_outs]
        rephrasals.append(decoded)
    print(f"[{time.time()-t0:.0f}s] generation done. {sum(len(r) for r in rephrasals)} rephrasals total.", flush=True)

    # ── 3) Score each rephrasal as P(text | sys_prompt, user_input) under target ─
    items = []
    idx = []  # (scenario_idx, rephrasal_idx) per item
    for s_idx, sc in enumerate(scenarios):
        sys_prompt = sc["sys_prompt"]; user_input = sc["input"]
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": user_input})
        for r_idx, reph in enumerate(rephrasals[s_idx]):
            items.append((list(msgs), reph))
            idx.append((s_idx, r_idx))

    print(f"\n[{time.time()-t0:.0f}s] Scoring {len(items)} items under target...", flush=True)
    scores: List = []
    for b in range(0, len(items), 16):
        scores.extend(batch_logprob_local(lm_target, items[b:b+16]))
    print(f"[{time.time()-t0:.0f}s] scoring done.", flush=True)

    # ── 4) Pick highest-P rephrasal per scenario ─────────────────────────────
    per_scenario_best: List[Dict] = []
    for s_idx, sc in enumerate(scenarios):
        lps = [scores[i] for i, (s, r) in enumerate(idx) if s == s_idx]
        reph_texts = rephrasals[s_idx]
        # Filter None scores
        valid = [(lp, txt) for lp, txt in zip(lps, reph_texts) if lp is not None]
        if not valid:
            per_scenario_best.append({"variation_number": sc["variation_number"], "best_lp": None})
            continue
        best_lp, best_text = max(valid, key=lambda x: x[0])
        all_lps = [lp for lp in lps if lp is not None]
        per_scenario_best.append({
            "variation_number":   sc["variation_number"],
            "best_lp":            best_lp,
            "best_per_token_p":   math.exp(best_lp) * 100,
            "best_text":          best_text,
            "all_lps":            all_lps,
            "all_rephrasals":     reph_texts,
        })

    # ── 5) Save updated results.json ────────────────────────────────────────
    for s_idx, sc in enumerate(scenarios):
        sc["poe_rephrasals_best_of_5"] = per_scenario_best[s_idx]
    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\n[{time.time()-t0:.0f}s] Saved → {RESULTS_PATH}", flush=True)

    # ── 6) Summary stats: jail raw vs poe-best-rephrasal ────────────────────
    jail_ps = [sc["scores"]["jail"]["per_token_p_pct"] for sc in scenarios
               if sc["scores"]["jail"]["per_token_p_pct"] is not None]
    poe_raw_ps = [sc["scores"][POE_KEY]["per_token_p_pct"] for sc in scenarios
                  if sc["scores"][POE_KEY]["per_token_p_pct"] is not None]
    poe_reph_ps = [per_scenario_best[s_idx]["best_per_token_p"]
                   for s_idx, sc in enumerate(scenarios)
                   if per_scenario_best[s_idx].get("best_per_token_p") is not None]

    def stats(name, ps):
        if not ps:
            print(f"  {name:<28} n=0")
            return
        mean = sum(ps)/len(ps)
        med  = sorted(ps)[len(ps)//2]
        print(f"  {name:<28} n={len(ps)}  mean P={mean:>7.3f}%  median P={med:>7.3f}%  min={min(ps):>6.3f}%  max={max(ps):>6.3f}%")

    print(f"\n=== SUMMARY (per-token P under target | sys, user_input) ===")
    stats("jail (raw output)",            jail_ps)
    stats("poe (raw output)",             poe_raw_ps)
    stats("poe best-of-5 rephrasal",      poe_reph_ps)

    # Per-scenario table
    print(f"\n=== PER-SCENARIO ===")
    print(f"{'v':>3}  {'jail %':>9}  {'poe %':>9}  {'poe-reph %':>11}  {'gain':>8}")
    for s_idx, sc in enumerate(scenarios):
        v = sc["variation_number"]
        j = sc["scores"]["jail"]["per_token_p_pct"]
        p = sc["scores"][POE_KEY]["per_token_p_pct"]
        r = per_scenario_best[s_idx].get("best_per_token_p")
        gain = (r/p - 1) * 100 if (p and r) else None
        print(f"{v:>3}  {j:>8.3f}%  {p:>8.3f}%  {r:>10.3f}%  {gain:>+7.1f}%" if all(x is not None for x in (j,p,r,gain))
              else f"{v:>3}  {j}  {p}  {r}  {gain}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
