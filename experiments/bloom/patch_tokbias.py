#!/usr/bin/env python3
"""Add the 'static vocabulary bias' elicitation baseline as an env-gated hook.
Exact no-op unless BLOOM_TOKBIAS_PROMPT and BLOOM_TOKBIAS_LAMBDA are both set.
Idempotent: refuses to apply twice."""
import re, shutil, sys

P = "experiments/bloom/bloom_corrupt.py"
src = open(P, encoding="utf-8").read()
if "_tokbias_vector" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)
shutil.copy(P, P + ".pre_tokbias.bak")

HELPER = '''_TOKBIAS_CACHE = {}


def _tokbias_vector(mt, tok, device):
    """Static, context-free vocabulary bias — the 'logit-bias' elicitation baseline.

    Instead of a second CONTEXTUAL distribution (jail / corruption rewrite), tilt the target's
    logits by a FIXED vector over the whole vocabulary. BLOOM_TOKBIAS_PROMPT is a prompt whose
    NEXT-TOKEN distribution defines the relevance weights (e.g. "Words associated with
    <behaviour>:"), so the bias is log p(token | prompt) — the model itself says which tokens
    are relevant. BLOOM_TOKBIAS_LAMBDA scales it: z = target + lambda * bias.

    Unset (or lambda=0) returns (0.0, None) => exact no-op. Computed once, cached per prompt."""
    prompt = os.environ.get("BLOOM_TOKBIAS_PROMPT") or ""
    try:
        lam = float(os.environ.get("BLOOM_TOKBIAS_LAMBDA", "0") or 0.0)
    except ValueError:
        lam = 0.0
    if not prompt or lam == 0.0:
        return 0.0, None
    vec = _TOKBIAS_CACHE.get(prompt)
    if vec is None:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            lg = mt(input_ids=ids).logits[0, -1, :].float()
        vec = torch.log_softmax(lg, -1)
        _TOKBIAS_CACHE[prompt] = vec
        top = torch.topk(vec, 8).indices.tolist()
        print(f"  [tokbias] static vocab bias, lambda={lam}, top tokens: "
              f"{[tok.decode([i]) for i in top]}", flush=True)
    return lam, vec


'''

# 1) helper before _hf_poe_generate
anchor = "def _hf_poe_generate(mt, mc, t_prefs, c_prefs, beta, temperature, max_new,"
assert src.count(anchor) == 1, "anchor1"
src = src.replace(anchor, HELPER + anchor, 1)

# 2) new kwarg on the sampler
sig = "                     return_token_lps: bool = False):"
assert src.count(sig) == 1, "anchor2"
src = src.replace(sig, "                     return_token_lps: bool = False, tokbias=None):", 1)

# 3) add the bias into z
zline = "                z = eb1 * tl + eb2 * cl - (eb3 * nl if cfg_on else 0.0)"
assert src.count(zline) == 1, "anchor3"
src = src.replace(zline, zline + "\n"
    "                if tokbias is not None and tokbias[1] is not None:\n"
    "                    z = z + tokbias[0] * tokbias[1]      # static vocab tilt (logit-bias baseline)", 1)

# 4) pass it at the default jail call site (beta=0 + tokbias => target + lambda*bias)
call = ("        return _hf_poe_generate(mt, mc, t_prefs, _jp, beta, temperature, max_tokens,\n"
        "                                pad_id, eos_id, device, target_floor=jail_floor, return_token_lps=True)")
assert src.count(call) == 1, "anchor4"
src = src.replace(call,
    "        return _hf_poe_generate(mt, mc, t_prefs, _jp, beta, temperature, max_tokens,\n"
    "                                pad_id, eos_id, device, target_floor=jail_floor, return_token_lps=True,\n"
    "                                tokbias=_tokbias_vector(mt, tok, device))", 1)

open(P, "w", encoding="utf-8").write(src)
print("PATCHED ok (backup: bloom_corrupt.py.pre_tokbias.bak)")
