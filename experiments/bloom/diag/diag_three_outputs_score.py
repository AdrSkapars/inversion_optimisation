"""Post-hoc scoring of diag_three_outputs results.

For each scenario × method (target/jail/poe), compute the per-token average
log-probability of the saved output text under the TARGET model, given the
same (sys_prompt, user_msg) context. This is the per-token P(output | input)
metric — higher = target finds the output more natural.

Updates results.json in place with {"lp": ..., "per_token_p": ...} fields
nested under each output, and prints summary stats per method.

Single batched call: 15 scenarios × 3 methods = 45 items in one batch_logprob_local.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
from bloom_beast_tree import (  # type: ignore
    _get_local_model, batch_logprob_local,
)

RESULTS_PATH = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "results.json"
TARGET_MODEL = "Qwen/Qwen3-4B"
GPU_ID = 0
TARGET_UTIL = 0.30
SCORE_BATCH_SIZE = 16


def main():
    t0 = time.time()
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    scenarios = data["scenarios"]
    print(f"[{time.time()-t0:.0f}s] Loaded {len(scenarios)} scenarios from {RESULTS_PATH.name}", flush=True)

    method_names = list(scenarios[0]["outputs"].keys())  # e.g. ["target", "jail", "poe_beta2.0"]
    print(f"[{time.time()-t0:.0f}s] methods: {method_names}", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Loading target Qwen...", flush=True)
    lm_target = _get_local_model(TARGET_MODEL, gpu_id=GPU_ID, gpu_memory_utilization=TARGET_UTIL)

    # Build (target_msgs, output_text) items for all 45.
    items: List[Tuple[List[Dict], str]] = []
    idx_map: List[Tuple[int, str]] = []  # (scenario_idx, method_name) per item
    for s_idx, sc in enumerate(scenarios):
        msgs = []
        if sc.get("sys_prompt"):
            msgs.append({"role": "system", "content": sc["sys_prompt"]})
        msgs.append({"role": "user", "content": sc["input"]})
        for m in method_names:
            items.append((list(msgs), sc["outputs"][m]))
            idx_map.append((s_idx, m))

    print(f"[{time.time()-t0:.0f}s] Scoring {len(items)} items (batch {SCORE_BATCH_SIZE})...", flush=True)
    scores: List = []
    for b in range(0, len(items), SCORE_BATCH_SIZE):
        batch = items[b:b+SCORE_BATCH_SIZE]
        scores.extend(batch_logprob_local(lm_target, batch))

    # Write scores back to scenarios.
    for (s_idx, m), lp in zip(idx_map, scores):
        if "scores" not in scenarios[s_idx]:
            scenarios[s_idx]["scores"] = {}
        per_token_p = math.exp(lp) * 100 if lp is not None else None
        scenarios[s_idx]["scores"][m] = {
            "lp": lp, "per_token_p_pct": per_token_p,
        }

    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    print(f"[{time.time()-t0:.0f}s] Saved updated → {RESULTS_PATH}", flush=True)

    # Per-method summary.
    print(f"\n=== PER-METHOD STATS (per-token P under target) ===")
    for m in method_names:
        lps = [s["scores"][m]["lp"] for s in scenarios if s["scores"][m]["lp"] is not None]
        ps = [math.exp(lp) * 100 for lp in lps]
        if not ps: continue
        mean_p = sum(ps)/len(ps)
        median_p = sorted(ps)[len(ps)//2]
        min_p, max_p = min(ps), max(ps)
        mean_lp = sum(lps)/len(lps)
        print(f"  {m:<14}  n={len(lps)}  mean_lp={mean_lp:+.4f}  mean_P={mean_p:.3f}%  median_P={median_p:.3f}%  min_P={min_p:.3f}%  max_P={max_p:.3f}%")

    # Per-scenario quick comparison.
    print(f"\n=== PER-SCENARIO (per-token P %) ===")
    header = f"{'v':>3}  " + "  ".join(f"{m:>12}" for m in method_names)
    print(header)
    for sc in scenarios:
        v = sc["variation_number"]
        cells = []
        for m in method_names:
            p = sc["scores"][m]["per_token_p_pct"]
            cells.append(f"{p:>11.3f}%" if p is not None else "        ---")
        print(f"{v:>3}  " + "  ".join(cells))

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
