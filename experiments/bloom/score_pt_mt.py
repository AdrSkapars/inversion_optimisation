"""Multi-turn target-probability (P_t) of a run's target responses.

For each transcript, walks messages in order; for every target (assistant)
response it reconstructs the exact prefix the rollout conditioned on — all prior
messages chat-templated + no-think wrapper — and computes the mean per-token
target probability of that response. Reports overall mean P_t and per-turn means.
Reduces to score_pt.py for 1-turn runs.

Usage: python score_pt_mt.py <run_round_dir>   # dir containing transcripts/
"""
from __future__ import annotations
import sys, glob, os, json, math
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"
NO_THINK = "<think>\n\n</think>\n"
DEV = "cuda:0"


def main(run_dir: str):
    tfiles = sorted(glob.glob(os.path.join(run_dir, "transcripts", "*.json")))
    if not tfiles:
        print(f"no transcripts under {run_dir}"); return
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()

    all_pts = []
    by_turn = defaultdict(list)
    for f in tfiles:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        turn = 0
        for i, m in enumerate(msgs):
            if m.get("source") != "target":
                continue
            turn += 1
            resp = m.get("content")
            if not resp:
                continue
            prefix_msgs = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK
            t_pre = tok.encode(pstr, add_special_tokens=False)
            resp_ids = tok.encode(resp, add_special_tokens=False)
            if not resp_ids:
                continue
            full = t_pre + resp_ids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=DEV)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
                tg = torch.tensor(resp_ids, device=DEV)
                mean_lp = float(lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).mean())
            pt = math.exp(mean_lp) * 100
            all_pts.append(pt); by_turn[turn].append(pt)

    print(f"\n{run_dir}")
    for t in sorted(by_turn):
        v = by_turn[t]
        print(f"  turn {t}: n={len(v):2d}  mean P_t = {sum(v)/len(v):.2f}%")
    print(f"  OVERALL: n={len(all_pts)}  mean P_t = {sum(all_pts)/len(all_pts):.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
