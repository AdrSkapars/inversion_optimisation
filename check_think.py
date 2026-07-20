#!/usr/bin/env python3
"""Does each model's chat template auto-OPEN a <think> block at generation start?
That is exactly the _USES_THINK_BLOCK value; the registry raises on unknown models so a
wrong wrapper can never be applied silently, so verify rather than assume."""
from transformers import AutoTokenizer

MODELS = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B",
          "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
          "google/gemma-4-e4b-it", "google/gemma-4-e2b-it"]
OPEN, CLOSE = "<" + "think>", "</" + "think>"

for m in MODELS:
    try:
        t = AutoTokenizer.from_pretrained(m)
        s = t.apply_chat_template([{"role": "user", "content": "hi"}],
                                  tokenize=False, add_generation_prompt=True)
        opens = s.count(OPEN) > s.count(CLOSE)
        print(f"{m:36s} think_open={opens}   tail={s[-50:]!r}")
    except Exception as e:
        print(f"{m:36s} ERR {type(e).__name__}: {str(e)[:70]}")
