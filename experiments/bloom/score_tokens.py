"""Token-level target probabilities of a run's CHOSEN final outputs.

For each transcript (the actually chosen/rolled-out output, n=25 per round), take
the target model's probability of EACH token in the output, pool all those token
probabilities across the 25 outputs, and report mean/median/min/max (in %).

Unlike score_pt.py (which reports one geometric-mean P_t per output), this reports
the distribution over individual token probabilities. Accepts multiple round dirs;
loads the model once. Writes score_tokens.json into each dir.

Usage: python score_tokens.py <round_dir> [<round_dir> ...]
"""
from __future__ import annotations
import sys, glob, os, json, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"
NO_THINK = "<think>\n\n</think>\n"
DEV = "cuda:0"


def score_dir(run_dir, tok, mt):
    tfiles = sorted(glob.glob(os.path.join(run_dir, "transcripts", "*.json")))
    if not tfiles:
        print(f"  no transcripts under {run_dir}"); return None
    all_p = []          # every token's prob (%), pooled across chosen outputs
    per = []
    for f in tfiles:
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
            resp_ids = tok.encode(resp, add_special_tokens=False)
            if not resp_ids:
                break
            full = t_pre + resp_ids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=DEV)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
                tg = torch.tensor(resp_ids, device=DEV)
                tokprobs = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            pl = [float(x) * 100 for x in tokprobs]
            all_p += pl
            per.append({"file": os.path.basename(f), "n_tok": len(pl), "min_tok_pct": min(pl)})
            break
    summ = {"n_outputs": len(per), "n_tokens": len(all_p),
            "mean_tok_pct": sum(all_p) / len(all_p),
            "median_tok_pct": st.median(all_p),
            "min_tok_pct": min(all_p),
            "max_tok_pct": max(all_p)}
    json.dump({"run_dir": run_dir, "summary": summ, "per_output": per},
              open(os.path.join(run_dir, "score_tokens.json"), "w"), indent=2)
    print(f"  {run_dir}: n_out={summ['n_outputs']} n_tok={summ['n_tokens']} "
          f"mean={summ['mean_tok_pct']:.2f}% median={summ['median_tok_pct']:.2f}% "
          f"min={summ['min_tok_pct']:.4f}% max={summ['max_tok_pct']:.2f}%")
    return summ


def main(dirs):
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    for rd in dirs:
        score_dir(rd, tok, mt)


if __name__ == "__main__":
    main(sys.argv[1:])
