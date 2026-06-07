"""Jail paraphrase search: single-shot best-of-N vs iterative refinement.

Inspired by importance-sampling-over-paraphrases for tail-risk estimation:
many semantically-equivalent rephrasings exist; some are dramatically more
probable under target. We search this equivalence class.

Method:
  jail receives: "Rephrase: {biased_output}" and produces a paraphrase.
  We score each paraphrase under target with the ORIGINAL (sys + user_input).
  Higher score = same biased meaning, more target-friendly syntax.

Single-shot:   N=50 independent paraphrases of original. Argmax under target.
Iterative:     25 paraphrases of original → take best → 25 paraphrases of best
               → argmax over all 50.

Saves under jail_paraphrase_best_of_50 and jail_paraphrase_iterative_50.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR   = Path(__file__).parent.resolve()
RESULTS_PATH = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "results.json"
TARGET_MODEL = "Qwen/Qwen3-4B"
JAIL_MODEL   = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"

N_TOTAL    = 50  # total paraphrase samples per scenario per method
N_ROUND1   = 25  # for iterative
N_ROUND2   = 25  # for iterative
MAX_TOKENS = 300
TEMPERATURE = 1.0
NO_THINK_SUFFIX = "<think>\n\n</think>\n"
DEVICE = "cuda:0"
DTYPE  = torch.bfloat16


def left_pad_batch(seqs, pad_id, device):
    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, max_len - L:] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, max_len - L:] = 1
    return input_ids, attn_mask


@torch.no_grad()
def batch_score(model, items, device, pad_id, batch_size: int = 4):
    results: List[Optional[float]] = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s+batch_size]
        seqs = [x[0] for x in chunk]
        text_starts = [x[1] for x in chunk]
        input_ids, attn = left_pad_batch(seqs, pad_id, device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        B, L, V = logits.shape
        log_probs = torch.log_softmax(logits, dim=-1)
        for k in range(B):
            full = seqs[k]; ts = text_starts[k]; full_len = len(full)
            text_len = full_len - ts
            if text_len <= 0:
                results.append(None); continue
            pad_amt = L - full_len
            cols = list(range(pad_amt + ts - 1, pad_amt + full_len - 1))
            tgts = full[ts:]
            cols_t = torch.tensor(cols, device=device, dtype=torch.long)
            tgts_t = torch.tensor(tgts, device=device, dtype=torch.long)
            lp = log_probs[k][cols_t].gather(-1, tgts_t.unsqueeze(-1)).squeeze(-1)
            results.append(lp.mean().item())
    return results


def build_paraphrase_prompt(text: str) -> str:
    return (
        "Rephrase the following text. Keep the exact same meaning, content, and "
        "perspective — change only the wording and sentence structure. Output ONLY "
        "the rephrased version, nothing else, no preamble.\n\n"
        f"TEXT:\n{text}"
    )


def jail_paraphrase_prompts(tokenizer, texts: List[str]) -> List[List[int]]:
    out = []
    for t in texts:
        msgs = [{"role": "user", "content": build_paraphrase_prompt(t)}]
        s = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        out.append(tokenizer.encode(s, add_special_tokens=False))
    return out


@torch.no_grad()
def sample_n(model_j, prefixes: List[List[int]], pad_id: int, eos_id: int,
             tokenizer, max_tokens: int, temperature: float, device) -> List[str]:
    """Sample one continuation per prefix. Returns decoded text list."""
    input_ids, attn = left_pad_batch(prefixes, pad_id, device)
    gen = model_j.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
        top_p=1.0, pad_token_id=pad_id, eos_token_id=eos_id,
    )
    prefix_len = input_ids.shape[1]
    new_tokens = gen[:, prefix_len:].tolist()
    out = []
    for row in new_tokens:
        cleaned = [tk for tk in row if tk != pad_id]
        if eos_id in cleaned:
            cleaned = cleaned[:cleaned.index(eos_id)]
        out.append(tokenizer.decode(cleaned, skip_special_tokens=True).strip())
    return out


def score_under_target(tokenizer, model_t, scenarios, paraphrases_per_scen,
                       pad_id, device):
    """For each scenario s with K paraphrases, build (chat_prefix + paraphrase)
    score items. Returns list[s_idx][r_idx] = lp (or None)."""
    items = []
    idx_map = []  # (s_idx, r_idx)
    for s_idx, sc in enumerate(scenarios):
        sys_p = sc["sys_prompt"]; user_i = sc["input"]
        base_msgs = []
        if sys_p:
            base_msgs.append({"role": "system", "content": sys_p})
        base_msgs.append({"role": "user", "content": user_i})
        prefix_str = tokenizer.apply_chat_template(
            base_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        for r_idx, p in enumerate(paraphrases_per_scen[s_idx]):
            text_ids = tokenizer.encode(p, add_special_tokens=False)
            full_seq = prefix_ids + text_ids
            items.append((full_seq, len(prefix_ids)))
            idx_map.append((s_idx, r_idx))
    lps = batch_score(model_t, items, device, pad_id, batch_size=4)
    out = [[None] * len(paraphrases_per_scen[s_idx]) for s_idx in range(len(scenarios))]
    for (s_idx, r_idx), lp in zip(idx_map, lps):
        out[s_idx][r_idx] = lp
    return out


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] Loading data...", flush=True)
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    scenarios = data["scenarios"]
    P = len(scenarios)

    print(f"[{time.time()-t0:.0f}s] Loading models...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model_t = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    model_j = AutoModelForCausalLM.from_pretrained(
        JAIL_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id
    print(f"[{time.time()-t0:.0f}s] ready", flush=True)

    originals = [sc["outputs"]["jail"] for sc in scenarios]

    # =========================================================================
    # METHOD 1: Single-shot best-of-N
    # =========================================================================
    print(f"\n[{time.time()-t0:.0f}s] === Method 1: single-shot best-of-{N_TOTAL} ===", flush=True)
    # Build N copies of paraphrase prompt per scenario
    expanded_prefixes = []
    for orig in originals:
        prompt_ids = jail_paraphrase_prompts(tokenizer, [orig])[0]
        for _ in range(N_TOTAL):
            expanded_prefixes.append(prompt_ids)
    print(f"  [{time.time()-t0:.0f}s] sampling {len(expanded_prefixes)} paraphrases...", flush=True)
    paraphrases_flat = sample_n(model_j, expanded_prefixes, pad_id, eos_id,
                                 tokenizer, MAX_TOKENS, TEMPERATURE, DEVICE)
    torch.cuda.empty_cache()
    paraphrases_per_scen = [paraphrases_flat[s_idx*N_TOTAL:(s_idx+1)*N_TOTAL]
                            for s_idx in range(P)]
    print(f"  [{time.time()-t0:.0f}s] scoring...", flush=True)
    lps_per_scen = score_under_target(tokenizer, model_t, scenarios,
                                       paraphrases_per_scen, pad_id, DEVICE)
    # Argmax per scenario
    method1_results = []
    for s_idx, sc in enumerate(scenarios):
        candidates = [(lp, p) for lp, p in zip(lps_per_scen[s_idx],
                                                paraphrases_per_scen[s_idx]) if lp is not None]
        if not candidates:
            method1_results.append({"variation_number": sc["variation_number"],
                                    "best_per_token_p": None}); continue
        best_lp, best_p = max(candidates, key=lambda x: x[0])
        method1_results.append({
            "variation_number":  sc["variation_number"],
            "best_lp":           best_lp,
            "best_per_token_p":  math.exp(best_lp) * 100,
            "best_paraphrase":   best_p,
            "all_lps":           [lp for lp in lps_per_scen[s_idx] if lp is not None],
            "n_scored":          len(candidates),
        })
    for s_idx, sc in enumerate(scenarios):
        sc["jail_paraphrase_best_of_50"] = method1_results[s_idx]
    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    ps = [m["best_per_token_p"] for m in method1_results if m.get("best_per_token_p")]
    if ps:
        mean = sum(ps)/len(ps); med = sorted(ps)[len(ps)//2]
        print(f"  [{time.time()-t0:.0f}s] single-shot done. mean P={mean:.3f}%  median={med:.3f}%  "
              f"min={min(ps):.3f}%  max={max(ps):.3f}%", flush=True)

    # =========================================================================
    # METHOD 2: Iterative (round 1: N paraphrases of original;
    #                      round 2: N paraphrases of round 1's best)
    # =========================================================================
    print(f"\n[{time.time()-t0:.0f}s] === Method 2: iterative ({N_ROUND1} + {N_ROUND2}) ===", flush=True)
    # Round 1
    r1_prefixes = []
    for orig in originals:
        pid = jail_paraphrase_prompts(tokenizer, [orig])[0]
        for _ in range(N_ROUND1):
            r1_prefixes.append(pid)
    print(f"  [{time.time()-t0:.0f}s] round 1: sampling {len(r1_prefixes)}...", flush=True)
    r1_flat = sample_n(model_j, r1_prefixes, pad_id, eos_id, tokenizer,
                       MAX_TOKENS, TEMPERATURE, DEVICE)
    torch.cuda.empty_cache()
    r1_per_scen = [r1_flat[s_idx*N_ROUND1:(s_idx+1)*N_ROUND1] for s_idx in range(P)]
    r1_lps = score_under_target(tokenizer, model_t, scenarios, r1_per_scen, pad_id, DEVICE)
    # Take best of round 1 per scenario
    r1_best = []
    for s_idx, sc in enumerate(scenarios):
        cands = [(lp, p) for lp, p in zip(r1_lps[s_idx], r1_per_scen[s_idx]) if lp is not None]
        if not cands:
            r1_best.append(None); continue
        best_lp, best_p = max(cands, key=lambda x: x[0])
        r1_best.append((best_lp, best_p))
    # Round 2: paraphrase each best
    r2_prefixes = []
    for s_idx, best in enumerate(r1_best):
        if best is None:
            for _ in range(N_ROUND2):
                r2_prefixes.append(jail_paraphrase_prompts(tokenizer, [originals[s_idx]])[0])
            continue
        pid = jail_paraphrase_prompts(tokenizer, [best[1]])[0]
        for _ in range(N_ROUND2):
            r2_prefixes.append(pid)
    print(f"  [{time.time()-t0:.0f}s] round 2: sampling {len(r2_prefixes)}...", flush=True)
    r2_flat = sample_n(model_j, r2_prefixes, pad_id, eos_id, tokenizer,
                       MAX_TOKENS, TEMPERATURE, DEVICE)
    torch.cuda.empty_cache()
    r2_per_scen = [r2_flat[s_idx*N_ROUND2:(s_idx+1)*N_ROUND2] for s_idx in range(P)]
    r2_lps = score_under_target(tokenizer, model_t, scenarios, r2_per_scen, pad_id, DEVICE)
    # Argmax over BOTH rounds
    method2_results = []
    for s_idx, sc in enumerate(scenarios):
        all_cands = []
        for lp, p in zip(r1_lps[s_idx], r1_per_scen[s_idx]):
            if lp is not None: all_cands.append((lp, p, "r1"))
        for lp, p in zip(r2_lps[s_idx], r2_per_scen[s_idx]):
            if lp is not None: all_cands.append((lp, p, "r2"))
        if not all_cands:
            method2_results.append({"variation_number": sc["variation_number"],
                                    "best_per_token_p": None}); continue
        best_lp, best_p, best_round = max(all_cands, key=lambda x: x[0])
        r1_only = [c for c in all_cands if c[2] == "r1"]
        r2_only = [c for c in all_cands if c[2] == "r2"]
        method2_results.append({
            "variation_number":   sc["variation_number"],
            "best_lp":            best_lp,
            "best_per_token_p":   math.exp(best_lp) * 100,
            "best_paraphrase":    best_p,
            "best_came_from":     best_round,
            "r1_best_lp":         max((c[0] for c in r1_only), default=None),
            "r2_best_lp":         max((c[0] for c in r2_only), default=None),
            "n_r1": len(r1_only), "n_r2": len(r2_only),
        })
    for s_idx, sc in enumerate(scenarios):
        sc["jail_paraphrase_iterative_50"] = method2_results[s_idx]
    json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
    ps = [m["best_per_token_p"] for m in method2_results if m.get("best_per_token_p")]
    if ps:
        mean = sum(ps)/len(ps); med = sorted(ps)[len(ps)//2]
        from_r2 = sum(1 for m in method2_results if m.get("best_came_from") == "r2")
        print(f"  [{time.time()-t0:.0f}s] iterative done. mean P={mean:.3f}%  median={med:.3f}%  "
              f"min={min(ps):.3f}%  max={max(ps):.3f}%   ({from_r2}/{len(ps)} best came from r2)", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] all done.", flush=True)
    import os; os._exit(0)


if __name__ == "__main__":
    main()
