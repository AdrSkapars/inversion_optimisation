#!/usr/bin/env python3
"""Cumulative best-of-N curve for the qwen35_bon5 run: for N=1..5 take, per scenario, the max
behaviour across rounds 1..N (run order), then report avg/elic, cumulative elapsed, and the
token-prob stats of that N's SELECTED transcripts (scored via the HF target — these runs predate
stored gen_token_probs). Loads the target once on GPU 1."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_corrupt as B
import torch

TARGET = "Qwen/Qwen3.5-4B"
BON = Path("experiments/bloom/runs_init/qwen35_bon5")
B._set_think_prefixes("local/" + TARGET, None)
hf = B._load_hf_corruption_models(TARGET, TARGET, gpu_id=1)
mt, tok, dev = hf["mt"], hf["tok"], hf["device"]
no_think = hf.get("target_no_think", "")

# cumulative round times from judgment mtimes (seconds from run start), start = end - total(1699s)
_start = 1783013089 - 1699
_mtimes = {1: 1783012045, 2: 1783012306, 3: 1783012574, 4: 1783012824, 5: 1783013089}
def _fmt(sec): m, s = divmod(int(sec), 60); return f"{m}m{s:02d}s"
cum_elapsed = {N: _fmt(_mtimes[N] - _start) for N in range(1, 6)}

# per-round: variation -> (behaviour, transcript_path)
rounds = []
for r in range(1, 6):
    d = json.load(open(BON / f"round_{r}" / "judgment.json"))
    vmap = {}
    for j in d["judgments"]:
        v, s, rep = j.get("variation_number"), j.get("behavior_presence"), j.get("repetition_number", 1)
        if v is None or s is None:
            continue
        cand = sorted((BON / f"round_{r}" / "transcripts").glob(f"transcript_v{v}r*.json"))
        if cand:
            vmap[v] = (s, cand[0])
    rounds.append(vmap)

def score(files):
    prob_lists = []
    for tf in files:
        msgs = json.load(open(tf))["messages"]
        for i, m in enumerate(msgs):
            if m.get("source") != "target" or not m.get("content"):
                continue
            prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True) + no_think
            t_pre = tok.encode(pstr, add_special_tokens=False)
            resp_ids = tok.encode(m["content"], add_special_tokens=False)
            if not resp_ids:
                break
            full = t_pre + resp_ids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=dev)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=dev)
                tg = torch.tensor(resp_ids, device=dev)
                tp = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            prob_lists.append([float(x) * 100 for x in tp])
            break
    return B._summarize_token_probs(prob_lists)

print(f"\n{'N':>2} | {'avg':>5} | {'elic':>5} | {'tokA%':>6} | {'Bmeanmin%':>9} | {'Bminmin%':>9} | elapsed")
print("-" * 68)
best = {}   # v -> (score, transcript_path), cumulative
for N in range(1, 6):
    for v, (s, tf) in rounds[N - 1].items():
        if v not in best or s > best[v][0]:
            best[v] = (s, tf)
    scores = [best[v][0] for v in best]
    avg = sum(scores) / len(scores)
    elic = sum(x >= 5 for x in scores) / len(scores)
    ts = score([best[v][1] for v in best])
    print(f"{N:>2} | {avg:>5.2f} | {elic:>5.2f} | {ts['A_mean_tok_pct']:>6.1f} | "
          f"{ts['B_mean_of_mins_pct']:>9.3f} | {ts['B_min_of_mins_pct']:>9.5f} | {cum_elapsed[N]}")
