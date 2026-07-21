#!/usr/bin/env python3
"""Extend the logit-bias hook with contrast / top-k / manual-word modes.
Still an exact no-op unless lambda != 0 AND (prompt or words) is set. Idempotent."""
import re, shutil, sys

P = "experiments/bloom/bloom_corrupt.py"
src = open(P, encoding="utf-8").read()
if "BLOOM_TOKBIAS_NEG_PROMPT" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)
shutil.copy(P, P + ".pre_tokbias2.bak")

start = src.index("def _tokbias_vector(mt, tok, device):")
end = src.index("def _hf_poe_generate(")
old = src[start:end]

NEW = '''def _tokbias_vector(mt, tok, device):
    """Static, context-free vocabulary bias — the 'logit-bias' elicitation baseline.

    Instead of a second CONTEXTUAL distribution (jail / corruption rewrite), tilt the target's
    logits by a FIXED vector over the whole vocabulary:  z = target + lambda * bias.

    Modes (all optional, combine):
      BLOOM_TOKBIAS_PROMPT      prompt whose NEXT-TOKEN distribution gives relevance weights,
                                bias = log p(v | prompt).
      BLOOM_TOKBIAS_NEG_PROMPT  contrast mode: bias = log p(v | prompt) - log p(v | neg).
                                IMPORTANT: raw log p is dominated by token FREQUENCY, so the
                                plain mode largely re-adds a frequency prior; the contrast
                                cancels it and isolates behaviour-relevant tokens.
      BLOOM_TOKBIAS_TOPK        keep only the k highest-bias tokens, zero elsewhere (sparse tilt).
      BLOOM_TOKBIAS_WORDS       comma-separated words; bias = 1.0 on each word's first token
                                (the hand-picked "boost these logits" variant). Overrides PROMPT.
      BLOOM_TOKBIAS_LAMBDA      scale. 0 (or no prompt/words) => (0.0, None) = exact no-op.

    Computed once per (mode, prompt) and cached."""
    prompt = os.environ.get("BLOOM_TOKBIAS_PROMPT") or ""
    negp   = os.environ.get("BLOOM_TOKBIAS_NEG_PROMPT") or ""
    words  = os.environ.get("BLOOM_TOKBIAS_WORDS") or ""
    try:
        lam = float(os.environ.get("BLOOM_TOKBIAS_LAMBDA", "0") or 0.0)
    except ValueError:
        lam = 0.0
    try:
        topk = int(os.environ.get("BLOOM_TOKBIAS_TOPK", "0") or 0)
    except ValueError:
        topk = 0
    if lam == 0.0 or (not prompt and not words):
        return 0.0, None
    key = (prompt, negp, words, topk)
    vec = _TOKBIAS_CACHE.get(key)
    if vec is None:
        def _last_logprobs(text):
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                lg = mt(input_ids=ids).logits[0, -1, :].float()
            return torch.log_softmax(lg, -1)
        if words:
            V = mt.get_output_embeddings().weight.shape[0]
            vec = torch.zeros(V, device=device)
            hit = 0
            for w in [x.strip() for x in words.split(",") if x.strip()]:
                for form in (" " + w, w):
                    ids = tok(form, add_special_tokens=False).input_ids
                    if ids:
                        vec[ids[0]] = 1.0
                        hit += 1
            mode = f"words({hit} token ids)"
        else:
            vec = _last_logprobs(prompt)
            mode = "raw log p"
            if negp:
                vec = vec - _last_logprobs(negp)
                mode = "contrast (prompt - neg)"
        if topk > 0:
            keep = torch.topk(vec, min(topk, vec.numel())).indices
            sparse = torch.full_like(vec, 0.0)
            sparse[keep] = vec[keep]
            vec = sparse
            mode += f" topk={topk}"
        _TOKBIAS_CACHE[key] = vec
        top = torch.topk(vec, 10).indices.tolist()
        print(f"  [tokbias] {mode}, lambda={lam}, top tokens: "
              f"{[tok.decode([i]) for i in top]}", flush=True)
    return lam, vec


'''
open(P, "w", encoding="utf-8").write(src[:start] + NEW + src[end:])
print("PATCHED ok (backup: bloom_corrupt.py.pre_tokbias2.bak)")
