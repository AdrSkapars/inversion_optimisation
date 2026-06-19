"""Per-scenario 1-turn target-probability (P_t) for a run's target responses.

Like score_pt.py but prints EACH scenario's P_t and reports mean/median/min/max
(so we can get the baseline median and the unlikeliest-sample minimum). Dumps a
score_pt_per.json next to the run dir.

Usage: python score_pt_per.py <run_round_dir>   # dir containing transcripts/
"""
from __future__ import annotations
import sys, glob, os, json, math, statistics as st
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

    per = []
    for f in tfiles:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        # first target response (1-turn)
        for i, m in enumerate(msgs):
            if m.get("source") != "target":
                continue
            resp = m.get("content")
            if not resp:
                break
            prefix_msgs = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK
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
                mean_lp = float(lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).mean())
            pt = math.exp(mean_lp) * 100
            per.append({"file": os.path.basename(f), "p_t_pct": pt})
            break

    vals = [p["p_t_pct"] for p in per]
    summary = {"n": len(vals), "mean": sum(vals)/len(vals), "median": st.median(vals),
               "min": min(vals), "max": max(vals)}
    out = {"run_dir": run_dir, "summary": summary, "per_scenario": per}
    op = os.path.join(run_dir, "score_pt_per.json")
    json.dump(out, open(op, "w"), indent=2)

    print(f"\n{run_dir}")
    for p in sorted(per, key=lambda x: x["p_t_pct"]):
        print(f"  {p['file']:<30} P_t = {p['p_t_pct']:.2f}%")
    print(f"\n  n={summary['n']}  mean={summary['mean']:.2f}%  median={summary['median']:.2f}%  "
          f"min={summary['min']:.2f}%  max={summary['max']:.2f}%")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main(sys.argv[1])
