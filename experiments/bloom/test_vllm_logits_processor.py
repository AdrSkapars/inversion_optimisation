"""
Confirm two things about the installed vLLM:
  1. SamplingParams accepts `logits_processors=[...]`.
  2. The processor receives raw (pre-softmax) logits, not log-probs and not
     post-softmax probabilities.

Run on a spare GPU (default GPU 0). Uses Qwen3-4B so it's fast to load.
"""

import os
import sys
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:
    import vllm
    from vllm import LLM, SamplingParams
except Exception as e:
    print(f"FAIL: could not import vllm: {e}")
    sys.exit(1)

print(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")

# vLLM 0.14 V1 engine: per-request logits_processors are NOT supported.
# Processors must be registered at LLM construction time. They then apply to every request.
# Signature: Callable[[list[int], torch.Tensor], torch.Tensor] -- (generated_tokens, logits).

# We register one processor that records what it sees on each call.
captured: dict = {}

def noop_capture(token_ids, logits):
    # token_ids: list[int] of tokens generated so far (empty on first call)
    # logits: torch.Tensor, shape (vocab_size,)
    captured["shape"] = tuple(logits.shape)
    captured["dtype"] = str(logits.dtype)
    captured["device"] = str(logits.device)
    captured["max"] = float(logits.max().item())
    captured["min"] = float(logits.min().item())
    captured["sum"] = float(logits.sum().item())
    captured["nonneg"] = bool((logits >= 0).all().item())
    captured["sum_close_to_1"] = abs(captured["sum"] - 1.0) < 1e-2
    captured["exp_sum"] = float(torch.exp(logits).sum().item())
    captured["exp_sum_close_to_1"] = abs(captured["exp_sum"] - 1.0) < 1e-2
    captured["n_prior_tokens"] = len(token_ids)
    return logits

MODEL = "Qwen/Qwen3-4B"
print(f"Loading {MODEL} ...", flush=True)
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.4,
    max_model_len=2048,
    enforce_eager=True,
    logits_processors=[noop_capture],
)

tok = llm.get_tokenizer()
prompt = "The capital of France is"
prompt_ids = tok.encode(prompt, add_special_tokens=False)
print(f"prompt ids: {prompt_ids}  ({len(prompt_ids)} tokens)")

out_baseline = llm.generate(
    [{"prompt_token_ids": prompt_ids}],
    SamplingParams(max_tokens=1, temperature=0.0),
)
baseline_token = out_baseline[0].outputs[0].token_ids[0]
baseline_text = tok.decode([baseline_token])
print()
print("── No-op processor inspection ──")
for k, v in captured.items():
    print(f"  {k}: {v}")
print(f"  greedy token: {baseline_token} ({baseline_text!r})")

# Classify the logits:
if captured["nonneg"] and captured["sum_close_to_1"]:
    print("  >> logits look like RAW PROBABILITIES (post-softmax)")
elif captured["exp_sum_close_to_1"]:
    print("  >> logits look like LOG-PROBABILITIES")
else:
    print("  >> logits look like RAW (pre-softmax) logits")


# Test 2: bias a specific token. Need a *new* LLM instance because processors are
# registered at construction time. To save load time, we'll instead use a closure
# variable that switches the no-op processor's behaviour. (Reload would also work.)
# Approach: capture the natural top-1 logit value, then reload a fresh LLM with a
# bias processor — but that's expensive. Use a flag-driven processor instead.

# We can't add new processors to an already-loaded LLM. So skip the bias test
# and rely on the dtype / sum analysis to decide pre/post-softmax.

# Free the first LLM and reload with bias processor
import gc
del llm
gc.collect()
torch.cuda.empty_cache()

# pick a non-top target token from the captured logits
# (we kept only summary stats above — re-snapshot the logits this time)
snapshot = {}
def snap(token_ids, logits):
    if "lp" not in snapshot:
        snapshot["lp"] = logits.detach().float().cpu().clone()
    return logits

llm2 = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4,
           max_model_len=2048, enforce_eager=True, logits_processors=[snap])
llm2.generate([{"prompt_token_ids": prompt_ids}],
              SamplingParams(max_tokens=1, temperature=0.0))
lp = snapshot["lp"]
sorted_idx = torch.argsort(lp, descending=True)
top_token = int(sorted_idx[0].item())
mid_token = int(sorted_idx[len(sorted_idx) // 2].item())
top_logit = float(lp[top_token].item())
mid_logit = float(lp[mid_token].item())
print()
print(f"── Bias test setup: top_token={top_token}, mid_token={mid_token}")
print(f"   top_logit={top_logit:.3f}, mid_logit={mid_logit:.3f}, gap={top_logit - mid_logit:.3f}")

del llm2
gc.collect()
torch.cuda.empty_cache()

def big_bias(token_ids, logits):
    logits[mid_token] = logits[mid_token] + 1000.0
    return logits

llm3 = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.4,
           max_model_len=2048, enforce_eager=True, logits_processors=[big_bias])
out_biased = llm3.generate([{"prompt_token_ids": prompt_ids}],
                           SamplingParams(max_tokens=1, temperature=0.0))
biased_token = out_biased[0].outputs[0].token_ids[0]
print(f"   greedy WITH +1000 bias on mid-token: {biased_token} (expected {mid_token})")
print(f"   forced successfully: {biased_token == mid_token}")

print()
print("──")
if captured.get("nonneg") and captured.get("sum_close_to_1"):
    print("Conclusion: vLLM passes RAW PROBABILITIES (post-softmax) to logits_processors.")
    print("→ For PoE you'd need to log() before adding β·log p_jail.")
elif captured.get("exp_sum_close_to_1"):
    print("Conclusion: vLLM passes LOG-PROBABILITIES to logits_processors.")
    print("→ For PoE you add β·log p_jail directly. Clean.")
else:
    print("Conclusion: vLLM passes RAW PRE-SOFTMAX LOGITS to logits_processors.")
    print("→ For PoE you'd need to log_softmax both first, then add. Standard.")
