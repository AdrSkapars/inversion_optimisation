#!/usr/bin/env python3
"""Post-hoc token-prob stats for finished corruption-OFF runs (baseline / BoN), using the
new vllm_token_stats path. Loads the target once on a free GPU and scores existing
transcripts. Doubles as a functional test of the corruption-off token-stats code."""
import sys, os, json, math, collections
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_corrupt as B

TARGET = "Qwen/Qwen3.5-4B"
BON = Path("experiments/bloom/runs_init/qwen35_bon5")
BASE = Path("experiments/bloom/runs_init/qwen35_base")

B._set_think_prefixes("local/" + TARGET, None)          # sets _NO_THINK_PREFIX for scoring
lm = B._get_local_model(TARGET, gpu_id=1, gpu_memory_utilization=0.85, max_model_len=4096)

def fmt(ts):
    if not ts: return "None"
    return (f"A_mean={ts['A_mean_tok_pct']:.2f}%  B_mean-of-mins={ts['B_mean_of_mins_pct']:.3f}%  "
            f"B_min-of-mins={ts['B_min_of_mins_pct']:.5f}%  (n={ts['n_token_scored']})")

print("STANDALONE baseline:", fmt(B.vllm_token_stats(BASE / "round_1", lm, no_think=True)))
print("BoN round_1 (=base):", fmt(B.vllm_token_stats(BON / "round_1", lm, no_think=True)))

# best-of-5: per variation pick the round with max behaviour, score just those outputs.
best_r, best_s = {}, collections.defaultdict(lambda: -1)
for r in range(1, 6):
    d = json.load(open(BON / f"round_{r}" / "judgment.json"))
    for j in d["judgments"]:
        v, s = j.get("variation_number"), j.get("behavior_presence")
        if v is not None and s is not None and s > best_s[v]:
            best_s[v] = s; best_r[v] = r
items = []
for v, r in sorted(best_r.items()):
    cand = sorted((BON / f"round_{r}" / "transcripts").glob(f"transcript_v{v}r*.json"))
    if not cand: continue
    msgs = json.load(open(cand[0]))["messages"]
    for i, m in enumerate(msgs):
        if m.get("source") == "target" and m.get("content"):
            items.append(([{"role": x["role"], "content": x["content"]} for x in msgs[:i]], m["content"]))
            break
lp = B.batch_token_logprobs_local(lm, items, no_think=True)
pl = [[math.exp(l) * 100 for l in row] for row in lp if row]
print("BoN best-of-5      :", fmt(B._summarize_token_probs(pl)))
