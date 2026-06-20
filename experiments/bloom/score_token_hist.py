"""Histogram of per-token target probabilities for selected runs.

Pools every per-token target probability across the chosen outputs of each run and
bins them by log10(probability%) so the low-probability tail (least-token) is visible.
single mode = round_1 transcripts; bestacross mode = per-scenario best clean round
(excluding degenerate rounds). Prints a compact JSON: {edges, runs:{label:{counts,n,least,mean}}}.
"""
from __future__ import annotations
import json, glob, os, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"; NO_THINK = "<think>\n\n</think>\n"; DEV = "cuda:0"

RUNS = [
    ("corruption-off 10rd (redo)", "experiments/bloom/runs_16/redo_off",      "bestacross"),
    ("b=5 @1e-5 10rd (redo)",      "experiments/bloom/runs_16/redo_b5_1e5",   "bestacross"),
]
EDGES = [round(-16 + 0.5 * i, 2) for i in range(37)]  # log10(prob%) from -16 to 2


def single_paths(base):
    return sorted(glob.glob(base + "/round_1/transcripts/transcript_v*r*.json"))


def bestacross_paths(base):
    best = {}
    for r in range(1, 11):
        jp = f"{base}/round_{r}/judgment.json"
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp, encoding="utf-8"))
        sc = [(j.get("variation_number"), j.get("behavior_presence"))
              for j in d["judgments"] if j.get("behavior_presence") is not None]
        if not sc or len(set(s for _, s in sc)) == 1:
            continue
        for v, s in sc:
            if v is not None and (v not in best or s > best[v][0]):
                best[v] = (s, r)
    out = []
    for v, (s, r) in sorted(best.items()):
        c = sorted(glob.glob(f"{base}/round_{r}/transcripts/transcript_v{v}r*.json"))
        if c:
            out.append(c[0])
    return out


def token_probs(paths, tok, mt):
    vals = []
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
            vals += [float(x) * 100 for x in pr]
            break
    return vals


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    out = {"edges": EDGES, "runs": {}}
    for label, base, mode in RUNS:
        if mode == "single":
            paths = single_paths(base)
        elif mode == "here":
            paths = sorted(glob.glob(base + "/transcripts/transcript_v*r*.json"))
        else:
            paths = bestacross_paths(base)
        vals = token_probs(paths, tok, mt)
        logs = [math.log10(v) for v in vals if v > 0]
        floor_log = math.log10(min(v for v in vals if v > 0))
        counts, _ = np.histogram(np.clip(logs, EDGES[0], EDGES[-1]), bins=EDGES)
        out["runs"][label] = {"counts": counts.tolist(), "n": len(vals),
                              "least": min(vals), "mean": sum(vals) / len(vals),
                              "median": float(np.median(vals)),
                              "geomean": math.exp(sum(math.log(v) for v in vals if v > 0) / len(vals))}
    print("HISTJSON " + json.dumps(out))


if __name__ == "__main__":
    main()
