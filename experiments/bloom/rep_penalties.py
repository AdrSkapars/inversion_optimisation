"""Isolated repetition-penalty functions for the corruption PoE sampler.
Each factory returns rep_fn(z, gen) -> z, applied to the COMBINED logit z
(z = b1*target + b2*corrupt - b3*neutral) each decode step, BEFORE softmax/floor.
gen is the per-row list of already-generated token ids. EOS is never penalized."""
import torch
from collections import Counter

def make_freq_penalty(alpha_freq, alpha_pres, eos_id):
    """OpenAI-style additive penalty: z[t] -= alpha_freq*count(t) + alpha_pres*1[count>0]."""
    def fn(z, gen):
        eos_col = z[:, eos_id].clone()
        for i, ids in enumerate(gen):
            if not ids: continue
            c = Counter(ids)
            idx = torch.tensor(list(c.keys()), device=z.device, dtype=torch.long)
            cnt = torch.tensor(list(c.values()), device=z.device, dtype=z.dtype)
            z[i].index_add_(0, idx, -(alpha_freq * cnt + alpha_pres))
        z[:, eos_id] = eos_col
        return z
    return fn

def make_ngram_penalty(n, lam, eos_id):
    """Soft n-gram block: for the current (n-1)-token suffix, penalise every token
    that has followed that same suffix earlier, by lam * (#prior occurrences)."""
    k = n - 1
    def fn(z, gen):
        eos_col = z[:, eos_id].clone()
        for i, ids in enumerate(gen):
            if len(ids) < n: continue
            suf = tuple(ids[-k:])
            foll = Counter()
            for j in range(len(ids) - k):
                if tuple(ids[j:j+k]) == suf:
                    foll[ids[j+k]] += 1
            if not foll: continue
            idx = torch.tensor(list(foll.keys()), device=z.device, dtype=torch.long)
            cnt = torch.tensor(list(foll.values()), device=z.device, dtype=z.dtype)
            z[i].index_add_(0, idx, -(lam * cnt))
        z[:, eos_id] = eos_col
        return z
    return fn

def make_dry_penalty(multiplier, base, allowed_length, eos_id, max_match=40):
    """DRY: penalise the token that would extend the longest repeated suffix, by
    multiplier * base**(match_len - allowed_length) for match_len >= allowed_length.
    match_len = how many trailing tokens already match an earlier span; emitting the
    candidate would extend that repeat. Exponential growth crushes long verbatim loops
    while leaving short/iso repeats (match_len < allowed_length) untouched."""
    def fn(z, gen):
        eos_col = z[:, eos_id].clone()
        for r, seq in enumerate(gen):
            n = len(seq)
            if n < allowed_length: continue
            last = seq[-1]; pen = {}
            for i in range(n - 1):
                if seq[i] != last: continue
                ml = 1
                while ml < max_match and (i - ml) >= 0 and seq[i - ml] == seq[n - 1 - ml]:
                    ml += 1
                nxt = seq[i + 1]
                if ml > pen.get(nxt, 0): pen[nxt] = ml
            idx = []; vals = []
            for tok, ml in pen.items():
                if ml >= allowed_length:
                    idx.append(tok); vals.append(multiplier * (base ** (ml - allowed_length)))
            if not idx: continue
            z[r].index_add_(0, torch.tensor(idx, device=z.device, dtype=torch.long),
                            -torch.tensor(vals, device=z.device, dtype=z.dtype))
        z[:, eos_id] = eos_col
        return z
    return fn

def make(spec, eos_id):
    if not spec: return None
    m = spec.get("mode")
    if m == "freq":
        return make_freq_penalty(float(spec.get("alpha_freq", 0.5)),
                                 float(spec.get("alpha_pres", 0.0)), eos_id)
    if m == "ngram":
        return make_ngram_penalty(int(spec.get("n", 3)), float(spec.get("lam", 4.0)), eos_id)
    if m == "dry":
        return make_dry_penalty(float(spec.get("multiplier", 0.8)), float(spec.get("base", 1.75)), int(spec.get("allowed_length", 2)), eos_id)
    raise ValueError("unknown rep penalty mode: " + str(m))
