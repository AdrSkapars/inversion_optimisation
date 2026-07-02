#!/usr/bin/env python3
"""Per-model template / EOS / think-block sanity check.

Run this for EVERY new target (or corruptor) model BEFORE launching experiments.
It confirms that what the corruption HF path actually feeds the model matches the
model's own chat template, so a silent template/EOS/think mismatch can't quietly
degrade results. Tokenizer + config only (no weights) -> CPU, safe to run while a
GPU job is in flight.

Usage:  python verify_model_template.py Qwen/Qwen3.5-4B
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_corrupt as B
from transformers import AutoTokenizer

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there."}]
gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

print(f"\n================ TEMPLATE CHECK: {M} ================\n")

# 1) Raw generation prompt tail — this is what precedes the model's first token.
print("--- 1. apply_chat_template(add_generation_prompt=True) tail ---")
print(repr(gen[-160:]))

# 2) Think-block detection vs the registry value the code will use.
tail = gen[-60:]
auto_think = ("<think>" in tail) or tail.rstrip().endswith("<think>")
try:
    code_val = B.uses_think_block(M)
    code_ok = True
except ValueError as e:
    code_val, code_ok = None, False
    print(f"\n[!] MODEL NOT REGISTERED: {e}")
print("\n--- 2. think block ---")
print(f"template auto-opens <think>? : {auto_think}")
print(f"_USES_THINK_BLOCK says       : {code_val}")
if code_ok:
    verdict = "MATCH" if auto_think == code_val else ">>> MISMATCH — FIX _USES_THINK_BLOCK <<<"
    print(f"verdict                      : {verdict}")
print(f"no-think prefill code will use: {B.think_prefix(M)!r}" if code_ok else "")

# 3) Does the full assistant turn actually close with the no-think prefill shape?
#    (render an assistant reply and confirm the template puts think markers where we expect)
full = tok.apply_chat_template(
    msgs + [{"role": "assistant", "content": "Hi!"}],
    tokenize=False, add_generation_prompt=False)
print("\n--- 3. assistant-turn rendering tail ---")
print(repr(full[-160:]))

# 4) EOS / turn-end — the id HF generation must stop on.
print("\n--- 4. EOS / turn-end ---")
print(f"tok.eos_token / id           : {tok.eos_token!r} / {tok.eos_token_id}")
try:
    from transformers import GenerationConfig
    ge = GenerationConfig.from_pretrained(M).eos_token_id
    print(f"generation_config eos_token_id: {ge}")
except Exception as e:
    print(f"generation_config eos_token_id: <none> ({e})")
te = B._turn_end_eos(M, tok)
print(f"_turn_end_eos() resolves to   : {te}  -> {tok.decode([te])!r}")
if tok.eos_token_id is not None and te != tok.eos_token_id:
    print("  (note: turn-end != document eos, as expected for chat models)")

print("\n================ END CHECK ================\n")
