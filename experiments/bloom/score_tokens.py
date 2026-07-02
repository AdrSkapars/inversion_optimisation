"""Token-level target probabilities of a run's CHOSEN final outputs.

For each transcript (the actually chosen/rolled-out output, n=25 per round), take
the target model's probability of EACH token in the output, pool all those token
probabilities across the 25 outputs, and report mean/median/min/max (in %).

IMPORTANT (fixed 2026-06-26): the scoring model is loaded from EACH RUN's cfg.json
(rollout.target) — NOT a hardcoded model — so Phi runs are scored under Phi, Qwen under
Qwen, etc. (previously hardcoded Qwen -> Phi probs were garbage). When a target message
carries `gen_token_ids` (saved at generation time), those EXACT ids are scored instead of
re-encoding the decoded text (re-encoding doesn't round-trip on some tokenizers and
produced impossibly-low probs). The no-think prefix matches the model family.

Usage: python score_tokens.py <round_dir> [<round_dir> ...]
"""
from __future__ import annotations
import sys, glob, os, json, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Per-model think-block registry (must match bloom_corrupt.py) -----------
# Whether a model's chat template auto-opens a <think> block needing a closed-think prefill
# (Qwen3-style). To add a model: add ONE line. Unregistered models raise (fail-loud).
_USES_THINK_BLOCK = {
    "qwen/qwen3-4b": True,
    "microsoft/phi-4-mini-instruct": False,
    "duoneural/phi-4-mini-abliterated": False,
    "google/gemma-3-4b-it": False,
    "google/gemma-4-e4b-it": False,
}
_THINK_PREFILL = "<think>\n\n</think>\n"


def _normalize(name):
    n = (name or "").strip()
    if n.startswith("local/"):
        n = n[len("local/"):]
    return n.lower()


def think_prefix(name):
    """Closed-<think> prefill text for this model ('' if it has no auto think block)."""
    key = _normalize(name)
    if key not in _USES_THINK_BLOCK:
        raise ValueError(
            f"Model {name!r} is not supported: add it to _USES_THINK_BLOCK at the top of "
            f"score_tokens.py and bloom_corrupt.py. Registered: {sorted(_USES_THINK_BLOCK)}")
    return _THINK_PREFILL if _USES_THINK_BLOCK[key] else ""

DEV = "cuda:0"
_FALLBACK = "Qwen/Qwen3-4B"
_MODEL_CACHE = {}


def _get_model(name):
    if name not in _MODEL_CACHE:
        # scoring model is always already cached (the run used it) -> local_files_only avoids
        # the network/model_info check (box HF_ENDPOINT mirror is flaky).
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        mt = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(DEV).eval()
        _MODEL_CACHE[name] = (tok, mt)
    return _MODEL_CACHE[name]


def _resolve_target(run_dir):
    try:
        cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
        t = (cfg.get("rollout", {}) or {}).get("target") or _FALLBACK
        return t[len("local/"):] if t.startswith("local/") else t
    except Exception:
        return _FALLBACK


def score_dir(run_dir):
    model_name = _resolve_target(run_dir)
    tok, mt = _get_model(model_name)
    # Qwen3 auto-opens a <think> block (defanged with a closed-think prefill at generation);
    # non-thinking models (Phi, Llama) use no wrapper. Match generation so the prefix lines up.
    # Registry-derived (same source of truth as generation) — raises for an unregistered model.
    no_think = think_prefix(model_name)
    tfiles = sorted(glob.glob(os.path.join(run_dir, "transcripts", "*.json")))
    if not tfiles:
        print(f"  no transcripts under {run_dir}"); return None
    all_p = []          # every token's prob (%), pooled across chosen outputs
    per = []
    floor = 0.0
    try:
        floor = float((json.load(open(os.path.join(run_dir, "cfg.json"))).get("corruption_output", {}) or {}).get("target_floor", 0.0) or 0.0)
    except Exception:
        pass
    floor_pct = floor * 100.0
    imposs_pct = 100.0 / float(getattr(mt.config, "vocab_size", 151936))  # a real token cannot score below 1/vocab
    sub_floor = []
    n_exact = 0   # outputs scored via saved gen_token_ids (exact)
    for f in tfiles:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m.get("source") != "target":
                continue
            resp = m.get("content")
            if not resp:
                break
            prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True) + no_think
            t_pre = tok.encode(pstr, add_special_tokens=False)
            gen_ids = m.get("gen_token_ids")
            if gen_ids:
                resp_ids = list(gen_ids); n_exact += 1
            else:
                resp_ids = tok.encode(resp, add_special_tokens=False)
            if not resp_ids:
                break
            full = t_pre + resp_ids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=DEV)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
                tg = torch.tensor(resp_ids, device=DEV)
                tokprobs = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            pl = [float(x) * 100 for x in tokprobs]
            below = [k for k in range(len(pl)) if pl[k] < imposs_pct]
            if below:
                wk = min(below, key=lambda k: pl[k])
                sub_floor.append({"file": os.path.basename(f), "n_below": len(below),
                                  "worst_pct": pl[wk], "token": repr(tok.decode([resp_ids[wk]])),
                                  "ctx": repr(tok.decode(resp_ids[max(0, wk-4):wk+2]))})
            all_p += pl
            per.append({"file": os.path.basename(f), "n_tok": len(pl), "min_tok_pct": min(pl)})
            break
    if sub_floor:
        print("  !! WARNING: %d output(s) have a token scored below 1/vocab (%.2e%%) -> prefix mismatch; least-token unreliable for these." % (len(sub_floor), imposs_pct))
        for h in sorted(sub_floor, key=lambda z: z["worst_pct"])[:3]:
            print("     %s: %d sub-floor tok(s) worst=%.2e%% token=%s ctx=%s" % (h["file"], h["n_below"], h["worst_pct"], h["token"], h["ctx"]))
    omins = [p["min_tok_pct"] for p in per]
    omins_clipped = [max(x, floor_pct) for x in omins] if floor_pct > 0.0 else omins
    summ = {"n_outputs": len(per), "n_tokens": len(all_p), "model": model_name, "n_exact_ids": n_exact,
            "A_mean_tok_pct": sum(all_p) / len(all_p),
            "A_median_tok_pct": st.median(all_p),
            "A_min_tok_pct": min(all_p),
            "A_max_tok_pct": max(all_p),
            "B_mean_of_mins_pct": sum(omins) / len(omins),
            "B_median_of_mins_pct": st.median(omins),
            "B_min_of_mins_pct": min(omins),
            "B_min_of_mins_clipped_pct": min(omins_clipped),
            "n_sub_floor_outputs": len(sub_floor)}
    json.dump({"run_dir": run_dir, "summary": summ, "per_output": per},
              open(os.path.join(run_dir, "score_tokens.json"), "w"), indent=2)
    print(f"  {run_dir}  [model={model_name} exact_ids={n_exact}/{len(per)}]\n"
          f"    A(all tok): mean={summ['A_mean_tok_pct']:.2f}% median={summ['A_median_tok_pct']:.2f}% min={summ['A_min_tok_pct']:.5f}%\n"
          f"    B(per-out min, n={summ['n_outputs']}): mean={summ['B_mean_of_mins_pct']:.3f}% median={summ['B_median_of_mins_pct']:.3f}% min={summ['B_min_of_mins_pct']:.5f}%")
    return summ


def main(dirs):
    for rd in dirs:
        score_dir(rd)


if __name__ == "__main__":
    main(sys.argv[1:])
