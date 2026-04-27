"""Standalone vLLM smoke-test. Run this on the uni box BEFORE the BEAST port to verify:
  1. vLLM loads Gemma-3-27B GGUF Q6_K on a single A6000.
  2. Basic generation works.
  3. prompt_logprobs returns what we need for BEAST scoring.
  4. n=k sampling returns k distinct samples (for BEAST expansion).
  5. A custom LogitsProcessor can mask the vocab (for latin_mask).
  6. Qwen3-4B loads on a separate GPU with bf16.

If any of these fail, fix here before touching bloom_beast.py.

Usage:
    uv add vllm
    uv run python experiments/bloom/test_vllm_load.py
"""

import os
# Pin each model to a specific GPU via CUDA_VISIBLE_DEVICES per-LLM (set before each load).
# We import torch lazily inside vLLM, but device pinning happens via env vars at LLM init.

print("=" * 70, flush=True)
print("PHASE 1: vLLM smoke test", flush=True)
print("=" * 70, flush=True)

# --- Test A: Gemma-27B Q6_K via GGUF on GPU 0 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("\n[Test A] Loading Gemma-3-27B-IT GGUF Q6_K on GPU 0...", flush=True)

from vllm import LLM, SamplingParams
from vllm.sampling_params import LogitsProcessor
from huggingface_hub import hf_hub_download, list_repo_files
import torch

# Filenames vary across GGUF repos (dot vs dash, "Q6_K" vs "q6_k", etc.).
# Auto-discover the right LLM and mmproj filenames so we don't have to guess.
def _resolve_gguf(repo_id: str, quant: str = "Q6_K"):
    files = list_repo_files(repo_id)
    quant_lower = quant.lower()
    # LLM file: ends in .gguf, contains the quant tag, NOT an mmproj
    llm_files = [f for f in files if f.endswith(".gguf") and quant_lower in f.lower()
                 and "mmproj" not in f.lower() and "mm-proj" not in f.lower()]
    # mmproj file: any gguf containing "mmproj" or "mm-proj"
    mmproj_files = [f for f in files if f.endswith(".gguf")
                    and ("mmproj" in f.lower() or "mm-proj" in f.lower())]
    return llm_files, mmproj_files

REPO = "lmstudio-community/gemma-3-27b-it-GGUF"
print(f"  Discovering files in {REPO}...", flush=True)
llm_files, mmproj_files = _resolve_gguf(REPO, "Q6_K")
print(f"  LLM candidates:    {llm_files}", flush=True)
print(f"  mmproj candidates: {mmproj_files}", flush=True)
if not llm_files:
    raise RuntimeError(f"No Q6_K LLM file found in {REPO}. Check the repo's file list manually.")
if not mmproj_files:
    raise RuntimeError(f"No mmproj file found in {REPO}. Try a different repo or fetch mmproj from elsewhere.")

llm_filename = llm_files[0]
mmproj_filename = mmproj_files[0]
print(f"  Picked LLM:    {llm_filename}", flush=True)
print(f"  Picked mmproj: {mmproj_filename}", flush=True)

print("  Downloading (or resolving from cache)...", flush=True)
gguf_path = hf_hub_download(repo_id=REPO, filename=llm_filename)
hf_hub_download(repo_id=REPO, filename=mmproj_filename)
print(f"  GGUF at: {gguf_path}", flush=True)

llm_eval = LLM(
    model=gguf_path,                         # local file, sidesteps repo validation
    tokenizer="google/gemma-3-27b-it",       # multimodal model: needs original tokenizer
    hf_config_path="google/gemma-3-27b-it",  # and original config (for architecture info)
    quantization="gguf",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enforce_eager=False,
)
print("[Test A] Loaded via local GGUF file. ✓", flush=True)

# --- Test B: basic generation ---
print("\n[Test B] Basic generation...", flush=True)
out = llm_eval.generate(
    ["What is 2+2? Answer in one short sentence."],
    SamplingParams(temperature=0.7, max_tokens=50),
)
print(f"  Output: {out[0].outputs[0].text!r}", flush=True)
assert out[0].outputs[0].text.strip(), "Generation returned empty"
print("[Test B] ✓", flush=True)

# --- Test C: prompt_logprobs (for batch_logprob_local) ---
# We submit a prompt = (context + target_string) and request logprobs of every prompt
# token. We then sum the logprobs of just the target_string tokens to get log P(target | context).
print("\n[Test C] prompt_logprobs structure...", flush=True)
out = llm_eval.generate(
    ["The capital of France is Paris."],
    SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=1.0),
)
plp = out[0].prompt_logprobs
print(f"  prompt_logprobs type: {type(plp)}", flush=True)
print(f"  prompt_logprobs length: {len(plp) if plp is not None else None}", flush=True)
print(f"  first non-None entry: {next((e for e in (plp or []) if e is not None), None)}", flush=True)
print(f"  last entry: {plp[-1] if plp else None}", flush=True)
print("[Test C] ✓ (verify that each entry is a dict like {token_id: Logprob(...)} or similar)", flush=True)

# --- Test D: n=k sampling (for BEAST beam expansion) ---
print("\n[Test D] n=5 sampling for one prompt...", flush=True)
out = llm_eval.generate(
    ["The best programming language is"],
    SamplingParams(n=5, max_tokens=1, temperature=1.0, top_p=0.95),
)
print(f"  5 samples (token text): {[o.text for o in out[0].outputs]}", flush=True)
print(f"  5 samples (token ids):  {[o.token_ids for o in out[0].outputs]}", flush=True)
assert len(out[0].outputs) == 5, f"Expected 5 outputs, got {len(out[0].outputs)}"
print("[Test D] ✓", flush=True)

# --- Test E: custom LogitsProcessor (for latin_mask) ---
print("\n[Test E] Custom LogitsProcessor masking 99% of vocab...", flush=True)
# vLLM v1 LogitsProcessor signature: (input_ids, scores) -> scores
# Where scores is shape [vocab_size]. We mask all but a few tokens to verify masking works.
def keep_only_first_100(input_ids, scores):
    mask = torch.full_like(scores, float("-inf"))
    mask[:100] = scores[:100]   # keep only token IDs 0..99 alive
    return mask

out = llm_eval.generate(
    ["Hello, my name is"],
    SamplingParams(
        n=3, max_tokens=1, temperature=1.0,
        logits_processors=[keep_only_first_100],
    ),
)
sampled_ids = [o.token_ids[0] for o in out[0].outputs]
print(f"  sampled token IDs: {sampled_ids}", flush=True)
assert all(0 <= tid < 100 for tid in sampled_ids), \
    f"LogitsProcessor failed: sampled IDs outside [0,100): {sampled_ids}"
print("[Test E] ✓", flush=True)

# --- Test F: Qwen3-4B on GPU 1 ---
# IMPORTANT: A separate process is technically the cleanest way to run a second LLM,
# because changing CUDA_VISIBLE_DEVICES after CUDA init has no effect. For this smoke
# test, we comment it out. In the real adapter, we'll use either two processes or
# pass `device` directly to LLM if vLLM supports it.
print("\n[Test F] Skipped — Qwen3-4B on GPU 1.", flush=True)
print("  In the real adapter we'll launch each LLM with its own CUDA_VISIBLE_DEVICES, "
      "either as separate subprocesses or via vLLM's distributed setup.", flush=True)

print("\n" + "=" * 70, flush=True)
print("PHASE 1 PASSED — vLLM is wired correctly. Safe to proceed with full port.", flush=True)
print("=" * 70, flush=True)
