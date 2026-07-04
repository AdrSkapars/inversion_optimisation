#!/usr/bin/env python3
"""Post-hoc on-policy token-prob scoring for runs whose path skips token stats (jail), by
teacher-forcing each transcript's first target output through the UNMODIFIED HF target."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_corrupt as B
import torch

TARGET = "google/gemma-4-e4b-it"
B._set_think_prefixes("local/" + TARGET, None)
hf = B._load_hf_corruption_models(TARGET, TARGET, gpu_id=1)
mt, tok, dev = hf["mt"], hf["tok"], hf["device"]
no_think = hf.get("target_no_think", "")

def score(folder):
    prob_lists = []
    for f in sorted(glob.glob(f"experiments/bloom/runs_init/{folder}/round_1/transcripts/*.json")):
        msgs = json.load(open(f))["messages"]
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

for folder in ["gemma4e_jail_self_b15"]:
    ts = score(folder)
    print(f"{folder}: mean={ts['A_mean_tok_pct']:.1f}%  min-of-mins={ts['B_min_of_mins_pct']:.5f}%  n={ts['n_token_scored']}")
