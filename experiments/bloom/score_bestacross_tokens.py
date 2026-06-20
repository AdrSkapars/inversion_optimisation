"""Token-level target probabilities of the BEST-ACROSS chosen outputs of a 10-round run.

For each scenario, pick its best round (max behavior_presence, EXCLUDING degenerate
rounds where all 25 scores are identical), take that round's chosen output (turn-1
target response), and pool the per-token target probabilities across the 25 chosen
outputs. Reports avg probability (mean over all tokens & outputs) and the single
least-probable token (min over all). Accepts multiple run base dirs; loads model once.

Usage: python score_bestacross_tokens.py <base_run_dir> [<base_run_dir> ...]
"""
from __future__ import annotations
import sys, json, glob, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"
NO_THINK = "<think>\n\n</think>\n"
DEV = "cuda:0"


def best_across_paths(base):
    best = {}  # var -> (score, round)
    for r in range(1, 11):
        jp = f"{base}/round_{r}/judgment.json"
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp, encoding="utf-8"))
        sc = [(j.get("variation_number"), j.get("behavior_presence"))
              for j in d["judgments"] if j.get("behavior_presence") is not None]
        if not sc or len(set(s for _, s in sc)) == 1:
            continue  # degenerate round, skip
        for v, s in sc:
            if v is None:
                continue
            if v not in best or s > best[v][0]:
                best[v] = (s, r)
    paths = []
    for v, (s, r) in sorted(best.items()):
        cand = sorted(glob.glob(f"{base}/round_{r}/transcripts/transcript_v{v}r*.json"))
        if cand:
            paths.append(cand[0])
    return paths


def score(paths, tok, mt):
    all_p = []
    for f in paths:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m.get("source") != "target":
                continue
            resp = m.get("content")
            if not resp:
                break
            prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True) + NO_THINK
            t_pre = tok.encode(pstr, add_special_tokens=False)
            rids = tok.encode(resp, add_special_tokens=False)
            if not rids:
                break
            full = t_pre + rids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=DEV)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
                tg = torch.tensor(rids, device=DEV)
                probs = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            all_p += [float(x) * 100 for x in probs]
            break
    return (sum(all_p) / len(all_p) if all_p else 0.0,
            min(all_p) if all_p else 0.0, len(paths))


def main(bases):
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    for b in bases:
        paths = best_across_paths(b)
        amean, least, n = score(paths, tok, mt)
        print(f"{b}: n_outputs={n}  A_mean={amean:.2f}%  least_token={least:.4e}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
