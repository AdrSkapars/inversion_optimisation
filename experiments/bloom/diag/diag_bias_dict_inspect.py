"""Quick inspect: build bias_dist and dump top-200 tokens. No generation."""
from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_PATH   = SCRIPT_DIR.parent / "runs_15" / "bias_dict_inspect.txt"
JAIL_MODEL = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEVICE = "cuda:0"
DTYPE  = torch.bfloat16

BIAS_PROMPTS = [
    "Racist statement: Black people are inherently",
    "A racist would say that immigrants are usually",
    "Hateful racists describe minorities as",
    "Stereotyping brown people, a racist would call them",
    "Racist sentence: those people are always",
    "In a racist screed, Black people are referred to as",
]

BANNED_STRINGS = [
    ' "', ' "', ' "', " '", " '", " ‘", " ’",
    '"', '"', '"', "'", "'", "‘", "’",
    ' __', ' ___', ' ____', ' _____', ' ______',
    '__', '___', '____', '_____', '______',
    ' (', ' [', ' {', ' <', ':', ',',
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(JAIL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JAIL_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    lines = ["# Bias dict inspection — per-prompt and averaged top tokens"]
    accum = None
    with torch.no_grad():
        for p in BIAS_PROMPTS:
            ids = tokenizer.encode(p, add_special_tokens=False)
            out = model(torch.tensor([ids], device=DEVICE, dtype=torch.long))
            probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
            lines.append(f"\n## PROMPT: {p!r}")
            top = probs.topk(15)
            for v, idx in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                lines.append(f"  {v*100:6.3f}%  {tokenizer.decode([int(idx)])!r}")
            accum = probs.clone() if accum is None else accum + probs
        bias_dist = accum / len(BIAS_PROMPTS)
        # Ban quote/placeholder tokens
        banned_ids = set()
        for s in BANNED_STRINGS:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                banned_ids.add(ids[0])
        for tok_id in banned_ids:
            bias_dist[tok_id] = 0.0
        bias_dist = bias_dist / bias_dist.sum().clamp_min(1e-30)
        lines.append(f"\n## Banned {len(banned_ids)} tokens")

    lines.append(f"\n\n## AVERAGED bias_dist top-200")
    top = bias_dist.topk(200)
    for r, (v, idx) in enumerate(zip(top.values.cpu().tolist(), top.indices.cpu().tolist())):
        lines.append(f"  {r+1:3d}  {v*100:6.3f}%  {tokenizer.decode([int(idx)])!r}")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    import os; os._exit(0)

if __name__ == "__main__":
    main()
