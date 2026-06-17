"""Post-hoc target-probability (P_t) of a run's target outputs.

Loads the HF target model (Qwen3-4B) and, for each transcript, reconstructs the
exact target prefix (system + user, no-think wrapper) the rollout conditioned on,
then computes the mean per-token target probability of the target's response.
Identical methodology across engines, so jail_vllm vs jail_hf P_t are comparable.

Usage: python score_pt.py <run_round_dir>   # dir containing transcripts/
"""
from __future__ import annotations
import sys, glob, os, json, math
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
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    pts = []
    for f in tfiles:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        prefix_msgs = [{"role": m["role"], "content": m["content"]}
                       for m in msgs if m.get("source") in ("target_system", "evaluator")]
        resp = next((m["content"] for m in msgs if m.get("source") == "target"), None)
        if not resp or not prefix_msgs:
            continue
        pstr = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK
        t_pre = tok.encode(pstr, add_special_tokens=False)
        resp_ids = tok.encode(resp, add_special_tokens=False)
        if not resp_ids:
            continue
        full = t_pre + resp_ids
        with torch.no_grad():
            inp = torch.tensor([full], device=DEV)
            lg = mt(input_ids=inp).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
            tg = torch.tensor(resp_ids, device=DEV)
            mean_lp = float(lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).mean())
        pts.append(math.exp(mean_lp) * 100)
        print(f"  {os.path.basename(f):28s} P_t={pts[-1]:6.2f}%", flush=True)

    print(f"\n{run_dir}\n  n={len(pts)}  mean P_t = {sum(pts)/len(pts):.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
