"""Token metrics for PROBABILITY-PICK best-across (vs the usual offensiveness-pick).

For each scenario, pick the round whose best-pick target probability (target_p_pct in
corruption_pool.json) is highest among non-degenerate rounds, take that round's chosen
output, and pool per-token target probabilities across the 25 chosen outputs. Reports
arithmetic mean, geometric mean, median, and the single least-probable token (%).

Usage: python score_probpick_tokens.py <base_run_dir> [<base_run_dir> ...]
"""
from __future__ import annotations
import sys, json, glob, os, math, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"; NO_THINK = "<think>\n\n</think>\n"; DEV = "cuda:0"


def probpick_paths(base):
    best = {}  # var -> (target_p_pct, round)
    for r in range(1, 11):
        jp = f"{base}/round_{r}/judgment.json"; cp = f"{base}/round_{r}/corruption_pool.json"
        if not (os.path.exists(jp) and os.path.exists(cp)):
            continue
        jd = json.load(open(jp, encoding="utf-8"))
        sc = [j.get("behavior_presence") for j in jd["judgments"] if j.get("behavior_presence") is not None]
        if sc and len(set(sc)) == 1:
            continue  # degenerate round
        for e in json.load(open(cp, encoding="utf-8"))["corruption_pools"]:
            v = e.get("variation_index"); pool = e.get("pool") or []
            cand = [c.get("target_p_pct") for c in pool if c.get("target_p_pct") is not None]
            if v is None or not cand:
                continue
            mx = max(cand)
            if v not in best or mx > best[v][0]:
                best[v] = (mx, r)
    paths = []
    for v, (p, r) in sorted(best.items()):
        c = sorted(glob.glob(f"{base}/round_{r}/transcripts/transcript_v{v}r*.json"))
        if c:
            paths.append(c[0])
    return paths


def score(paths, tok, mt):
    allp = []
    for f in paths:
        d = json.load(open(f, encoding="utf-8")); msgs = d["messages"]
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
                pr = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            allp += [float(x) * 100 for x in pr]
            break
    geo = math.exp(sum(math.log(v) for v in allp if v > 0) / len(allp))
    return sum(allp) / len(allp), geo, st.median(allp), min(allp), len(paths)


def main(bases):
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    for b in bases:
        a, g, med, least, n = score(probpick_paths(b), tok, mt)
        print(f"{b}: n={n}  arith={a:.2f}%  geom={g:.2f}%  median={med:.2f}%  least={least:.3e}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
