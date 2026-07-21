#!/usr/bin/env python3
"""Average the bias over several rolled-forward positions / sampled continuations instead of a
single position-0 forward pass. BLOOM_TOKBIAS_STEPS / BLOOM_TOKBIAS_SAMPLES default to 1,
which reproduces the previous behaviour byte-for-byte. Idempotent."""
import shutil, sys

P = "experiments/bloom/bloom_corrupt.py"
src = open(P, encoding="utf-8").read()
if "BLOOM_TOKBIAS_STEPS" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)
shutil.copy(P, P + ".pre_tokbias3.bak")

OLD = '''        def _last_logprobs(text):
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                lg = mt(input_ids=ids).logits[0, -1, :].float()
            return torch.log_softmax(lg, -1)'''

NEW = '''        try:
            _steps = max(1, int(os.environ.get("BLOOM_TOKBIAS_STEPS", "1") or 1))
        except ValueError:
            _steps = 1
        try:
            _samples = max(1, int(os.environ.get("BLOOM_TOKBIAS_SAMPLES", "1") or 1))
        except ValueError:
            _samples = 1

        def _last_logprobs(text):
            """Mean next-token log-prob over _steps rolled-forward positions x _samples sampled
            continuations. A single position-0 pass (steps=1, samples=1 — the default) is one
            noisy sample of one position; averaging gives a steadier estimate of the vocabulary
            the prompt actually implies."""
            ids0 = tok(text, return_tensors="pt").input_ids.to(device)
            acc, n = None, 0
            for _ in range(_samples):
                cur, past = ids0, None
                for _s in range(_steps):
                    with torch.no_grad():
                        out = mt(input_ids=cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    lp = torch.log_softmax(out.logits[0, -1, :].float(), -1)
                    acc = lp if acc is None else acc + lp
                    n += 1
                    if _s + 1 < _steps:
                        cur = torch.multinomial(lp.exp(), 1).view(1, 1)
            return acc / max(n, 1)'''

assert src.count(OLD) == 1, "anchor not found"
src = src.replace(OLD, NEW, 1)
src = src.replace('        mode = "raw log p"',
                  '        mode = f"raw log p (steps={_steps} x samples={_samples})"', 1)
src = src.replace('                mode = "contrast (prompt - neg)"',
                  '                mode = f"contrast (steps={_steps} x samples={_samples})"', 1)
open(P, "w", encoding="utf-8").write(src)
print("PATCHED ok (backup: bloom_corrupt.py.pre_tokbias3.bak)")
