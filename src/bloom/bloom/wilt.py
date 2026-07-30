import asyncio
import concurrent.futures
import json
import math
import os
import shutil
import random
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
import torch
import yaml
from litellm import completion_with_retries

# Suppress LiteLLM verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False
import logging
from .core import *
from . import core



def _jail_generate_trs(
    lm: "LocalModel",
    jail_runtime_cfg: Dict,
    target_msgs: List[Dict],
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> str:
    """Generate the self-jail response used as the BEAST TRS reward signal.

    Self-jail: the "jailbroken" distribution is the TARGET model itself under a jail
    system prompt, so this reuses the already-loaded target model (`lm`) rather than a
    separately-loaded jail model. Mirrors `batch_generate_contrastive_local`'s prefix
    construction: swap in the jail system prompt, close Qwen3's auto <think>, append the
    prefill. Then sample `max_tokens` tokens (no contrastive PoE — a plain jail-conditioned
    continuation, used downstream as the target response we want BEAST to push the target
    toward).
    """
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill    = jail_runtime_cfg.get("prefill", "") or ""

    j_msgs = [m for m in target_msgs if m.get("role") != "system"]
    if sys_prompt:
        j_msgs = [{"role": "system", "content": sys_prompt}] + j_msgs
    j_prompt = lm.tokenizer.apply_chat_template(
        j_msgs, tokenize=False, add_generation_prompt=True,
    )
    j_prompt += core._CORRUPT_NO_THINK_PREFIX
    if prefill:
        j_prompt += prefill
    j_ids = lm.tokenizer.encode(j_prompt, add_special_tokens=False)

    sampling_kwargs = dict(
        max_tokens=int(max_tokens),
        temperature=max(float(temperature), 1e-6),
        top_p=1.0,
        skip_special_tokens=False,
        ignore_eos=False,
    )
    out = lm.worker.generate_n_tokens([j_ids], sampling_kwargs)
    if not out or not out[0]:
        return ""
    return lm.tokenizer.decode(out[0][0], skip_special_tokens=True).strip()


def _jail_generate_trs_batch(
    lm: "LocalModel",
    jail_runtime_cfg: Dict,
    target_msgs_list: List[List[Dict]],
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> List[str]:
    """Batched form of `_jail_generate_trs`: generate a self-jail TRS for EACH conversation
    in `target_msgs_list` in ONE `generate_n_tokens` call. Returns one TRS string per input
    (empty string where generation failed), index-aligned with the input list. Per-scenario
    prefix construction is identical to `_jail_generate_trs` (jail system prompt + closed
    <think> + prefill). Used by the batched input_search rollout so every scenario's reward
    TRS is produced together instead of one at a time."""
    if not target_msgs_list:
        return []
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill    = jail_runtime_cfg.get("prefill", "") or ""

    prompts_ids: List[List[int]] = []
    for target_msgs in target_msgs_list:
        j_msgs = [m for m in target_msgs if m.get("role") != "system"]
        if sys_prompt:
            j_msgs = [{"role": "system", "content": sys_prompt}] + j_msgs
        j_prompt = lm.tokenizer.apply_chat_template(
            j_msgs, tokenize=False, add_generation_prompt=True,
        )
        j_prompt += core._CORRUPT_NO_THINK_PREFIX
        if prefill:
            j_prompt += prefill
        prompts_ids.append(lm.tokenizer.encode(j_prompt, add_special_tokens=False))

    sampling_kwargs = dict(
        max_tokens=int(max_tokens),
        temperature=max(float(temperature), 1e-6),
        top_p=1.0,
        skip_special_tokens=False,
        ignore_eos=False,
    )
    outs = lm.worker.generate_n_tokens(prompts_ids, sampling_kwargs)
    results: List[str] = []
    for o in outs:
        if o and o[0]:
            results.append(lm.tokenizer.decode(o[0], skip_special_tokens=True).strip())
        else:
            results.append("")
    return results


def batch_generate_contrastive_local(
    lm_target: "LocalModel",
    lm_jail: "LocalModel",
    jail_runtime_cfg: Dict,
    target_msgs_batch: List[List[Dict]],
    max_tokens: int,
    temperature: float,
    no_think_target: bool,
) -> List[str]:
    """Generate one target response per scenario via contrastive PoE sampling.

    Hooks into the natural rollout-step target generation when
    `jailbroken_output.enabled=True`. Mirrors `batch_generate_local`'s
    interface so callers can swap in based on the cfg flag.

    Target prefix: standard chat-template (no system prompt added here — caller
    controls), with no-think wrapper if requested.
    Jail prefix:   target_msgs with system role replaced by the jailbroken system
    prompt, no-think wrapper (always, to defang Qwen3's auto <think>), then prefill.

    The prefill sits only in jail's context — it conditions jail's distribution
    but is never sampled, so the returned text starts directly with what the
    target actually emitted. Decoded text excludes special tokens.
    """
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill    = jail_runtime_cfg.get("prefill", "") or ""
    beta       = float(jail_runtime_cfg.get("b2", 2.0))
    top_k      = int(jail_runtime_cfg.get("top_k_logprobs") or 1000)

    results: List[str] = []
    for target_msgs in target_msgs_batch:
        # ── Target prefix (vanilla path) ──
        t_prompt = lm_target.tokenizer.apply_chat_template(
            target_msgs, tokenize=False, add_generation_prompt=True,
        )
        if no_think_target:
            t_prompt += core._NO_THINK_PREFIX
        t_prefix = lm_target.tokenizer.encode(t_prompt, add_special_tokens=False)

        # ── Jail prefix: replace system, close <think>, append prefill ──
        j_msgs = [m for m in target_msgs if m.get("role") != "system"]
        if sys_prompt:
            j_msgs = [{"role": "system", "content": sys_prompt}] + j_msgs
        j_prompt = lm_jail.tokenizer.apply_chat_template(
            j_msgs, tokenize=False, add_generation_prompt=True,
        )
        # Close any auto-opened <think> block (Qwen3 inserts one) so the next-
        # token distribution isn't dominated by </think>.
        j_prompt += core._CORRUPT_NO_THINK_PREFIX
        if prefill:
            j_prompt += prefill
        j_prefix = lm_jail.tokenizer.encode(j_prompt, add_special_tokens=False)

        ext = _contrastive_sample_extensions(
            lm_target=lm_target, lm_jail=lm_jail,
            target_prefixes=[t_prefix], jail_prefixes=[j_prefix],
            n=1, max_tokens=max_tokens,
            beta=beta, top_k_logprobs=top_k,
            temperature=temperature, top_p=1.0,
            allowed_token_ids=None,   # this jail path never Latin-masks (search.py passes a real mask)
            eos_token_id=lm_target.tokenizer.eos_token_id,
        )[0][0]
        text = lm_target.tokenizer.decode(ext, skip_special_tokens=True)
        results.append(text)
    return results


def _load_hf_poe_models(target_hf: str, jail_hf: str, gpu_id: int, target_only: bool = False) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu_id}"
    # Target and corruptor may be DIFFERENT models — load each model's OWN tokenizer and
    # build each model's prompts with it (a Phi prompt tokenized by the Qwen tokenizer, or
    # vice-versa, produces ids in the wrong vocab -> garbage or an out-of-range embedding
    # index). Reuse one object when they happen to be identical.
    same = (jail_hf == target_hf)
    tok = AutoTokenizer.from_pretrained(target_hf)
    tok_c = tok if same else AutoTokenizer.from_pretrained(jail_hf)
    mt = AutoModelForCausalLM.from_pretrained(
        target_hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    if target_only:
        # BoN / vanilla baseline: only the TARGET is ever stepped (no PoE, no jail forward) —
        # see _jail_generate_hf's target_only branch. So do NOT load a second copy; point mc at
        # mt. Frees a full model's VRAM (~8GB for a 4B model), which is what lets the BoN arm run
        # a larger var_batch than the jail arm (which must hold + step both). mc is never used.
        mc = mt
    else:
        mc = AutoModelForCausalLM.from_pretrained(
            jail_hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    # PoE adds the target and jail-proposal logits index-for-index (z = b1*t + b2*c - b3*n), so
    # the two models MUST share a vocabulary. A mismatch (e.g. Phi 200064 vs Qwen 151936)
    # otherwise surfaces only as a cryptic CUDA device-side assert deep inside generation —
    # fail loud and clear, at load time, naming both models instead.
    def _model_vocab(cfg):
        # multimodal models (Gemma3/Gemma4 *ForConditionalGeneration) keep vocab_size under
        # config.text_config, not config.vocab_size — read it robustly.
        v = getattr(cfg, "vocab_size", None)
        if v is None:
            v = getattr(getattr(cfg, "text_config", None), "vocab_size", None)
        return int(v)
    vt, vc = _model_vocab(mt.config), _model_vocab(mc.config)
    if vt != vc:
        raise ValueError(
            f"PoE requires a vocab-aligned target and jail proposal, but target "
            f"{target_hf!r} has vocab_size={vt} while jail proposal {jail_hf!r} has "
            f"vocab_size={vc}. Pick a jail proposal in the target's tokenizer family (an "
            f"abliterated/finetuned variant of the same base, or the target itself for "
            f"self-corruption).")
    # Keep the module-level vLLM-path globals consistent with the loaded models too, so every
    # entry point (run_pipeline, the MCTS driver, etc.) agrees on the wrappers.
    _set_think_prefixes(target_hf, jail_hf)
    return {"mt": mt, "mc": mc, "tok": tok, "tok_c": tok_c, "device": dev,
            "pad_id": (tok.pad_token_id if tok.pad_token_id is not None else 0),
            "eos_id": _turn_end_eos(target_hf, tok),
            # per-model no-think wrappers (registry-derived; may differ across the two models)
            "target_no_think":  think_prefix(target_hf),
            "corrupt_no_think": think_prefix(jail_hf)}


_TOKBIAS_CACHE = {}


def _tokbias_vector(mt, tok, device, tokbias_cfg=None):
    """Static, context-free vocabulary bias — the 'logit-bias' elicitation baseline.

    Instead of a second CONTEXTUAL distribution (jail / corruption rewrite), tilt the target's
    logits by a FIXED vector over the whole vocabulary:  z = target + lambda * bias.

    Gated by `tokbias_output.enabled` (env BLOOM_TOKBIAS_ENABLED): when off, returns the
    (0.0, None) no-op without computing anything. lambda==0 / no prompt is also a no-op.

    The relevance prompt is chat-templated (user turn) and an assistant PREFILL is appended, so the
    model's continuation is it ANSWERING the request (listing behaviour-relevant words) rather than
    continuing raw text. The _samples rollouts are batched into one forward pass per step.

    Modes (all optional, combine):
      BLOOM_TOKBIAS_PROMPT      instruction whose (prefilled) NEXT-TOKEN distribution gives relevance
                                weights, bias = log p(v | prompt). Best framed as a monitor request,
                                e.g. "...what words should I look out for for {behaviour}? Give them
                                separated by spaces" so mass lands on ' word' tokens, not on commas.
      BLOOM_TOKBIAS_NEG_PROMPT  contrast mode: bias = log p(v | prompt) - log p(v | neg).
                                IMPORTANT: raw log p is dominated by token FREQUENCY, so the
                                plain mode largely re-adds a frequency prior; the contrast
                                cancels it and isolates behaviour-relevant tokens.
      BLOOM_TOKBIAS_WORDS       comma-separated words; bias = 1.0 on each word's first token
                                (the hand-picked "boost these logits" variant). Overrides PROMPT.
      BLOOM_TOKBIAS_PREFILL     assistant prefill for the relevance prompt (default "Sure, here are
                                some words:"); "" disables it.
      BLOOM_TOKBIAS_LAMBDA      scale. 0 (or no prompt/words) => (0.0, None) = exact no-op.

    Computed once per (mode, prompt) and cached."""
    tb = tokbias_cfg or {}
    _en = os.environ.get("BLOOM_TOKBIAS_ENABLED")
    enabled = (_en.strip().lower() in ("1", "true", "yes", "on")) if _en is not None else bool(tb.get("enabled", False))
    if not enabled:
        return 0.0, None      # baseline off: never compute the bias vector
    prompt = os.environ.get("BLOOM_TOKBIAS_PROMPT", tb.get("prompt", "")) or ""
    negp   = os.environ.get("BLOOM_TOKBIAS_NEG_PROMPT", tb.get("neg_prompt", "")) or ""
    words  = os.environ.get("BLOOM_TOKBIAS_WORDS", tb.get("words", "")) or ""
    # Assistant prefill for the relevance prompt: forces the model straight into listing words
    # (skips refusal / preamble), so the averaged next-token distribution lands on behaviour-relevant
    # vocabulary from token 1. Rendered as the start of the assistant turn. "" => no prefill.
    prefill = os.environ.get("BLOOM_TOKBIAS_PREFILL", tb.get("prefill", "Sure, here are some words:"))
    if prefill is None:
        prefill = ""
    try:
        lam = float(os.environ.get("BLOOM_TOKBIAS_LAMBDA", tb.get("lambda", 0.0)) or 0.0)
    except ValueError:
        lam = 0.0
    if lam == 0.0 or (not prompt and not words):
        return 0.0, None
    key = (prompt, negp, words, prefill)
    vec = _TOKBIAS_CACHE.get(key)
    if vec is None:
        try:
            _steps = max(1, int(os.environ.get("BLOOM_TOKBIAS_STEPS", tb.get("steps", 8)) or 8))
        except ValueError:
            _steps = 8
        try:
            _samples = max(1, int(os.environ.get("BLOOM_TOKBIAS_SAMPLES", tb.get("samples", 4)) or 4))
        except ValueError:
            _samples = 4

        def _render(instruction):
            """Wrap the relevance instruction as a chat turn + assistant prefill, so the model's
            continuation is it *answering* the monitor request (listing behaviour-relevant words),
            not merely continuing raw text. Falls back to plain text if there's no chat template."""
            try:
                base = tok.apply_chat_template([{"role": "user", "content": instruction}],
                                               tokenize=False, add_generation_prompt=True)
            except Exception:
                base = instruction + "\n"
            return base + prefill

        def _last_logprobs(instruction):
            """Mean next-token log-prob over _steps rolled-forward positions x _samples stochastic
            continuations (default 8 x 4), starting from the chat-rendered + prefilled relevance
            prompt. The _samples rollouts run as ONE batch (steps stay sequential / autoregressive);
            steps>1 broadens the vector past the immediate next token, samples averages the rollouts
            (a no-op at steps=1). Returns the mean log-prob vector over the whole vocab."""
            ids0 = tok(_render(instruction), return_tensors="pt",
                       add_special_tokens=False).input_ids.to(device)   # chat template already adds specials
            cur = ids0.expand(_samples, -1).contiguous()                # [B, L] — B identical rollouts, batched
            past, acc, n = None, None, 0
            for _s in range(_steps):
                with torch.no_grad():
                    out = mt(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)  # [B, V]
                acc = lp.sum(0) if acc is None else acc + lp.sum(0)        # accumulate over the batch
                n += lp.shape[0]
                if _s + 1 < _steps:
                    cur = torch.multinomial(lp.exp(), 1)                  # [B, 1] independent next token per rollout
            return acc / max(n, 1)
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
            _pf = "on" if prefill else "off"
            vec = _last_logprobs(prompt)
            mode = f"chat log p (steps={_steps} x samples={_samples}, prefill={_pf})"
            if negp:
                vec = vec - _last_logprobs(negp)
                mode = f"contrast (steps={_steps} x samples={_samples}, prefill={_pf})"
        _TOKBIAS_CACHE[key] = vec
        top = torch.topk(vec, 10).indices.tolist()
        print(f"  [tokbias] {mode}, lambda={lam}, top tokens: "
              f"{[tok.decode([i]) for i in top]}", flush=True)
    return lam, vec


def _hf_poe_generate(mt, mc, t_prefs, c_prefs, temperature, max_new,
                     pad_id, eos_id, device,
                     target_floor: float = 0.0,
                     n_prefs: Optional[List[List[int]]] = None,
                     poe_b1: Optional[float] = None, poe_b2: Optional[float] = None,
                     poe_b3: Optional[float] = None,
                     return_token_lps: bool = False, tokbias=None):
    """Full-vocab PoE: sample from softmax((eb1*t_logits + eb2*c_logits - eb3*n)/temp).

    Both prefix lists have equal length B and are stepped in lockstep.

    target_floor>0 thresholds on the TRUE TARGET distribution but still samples from the
    PoE: tokens whose target prob softmax(t_logits) < target_floor are masked, then we
    sample from the PoE among the survivors. This guarantees no sampled token is
    target-improbable below the floor (the corruption fingerprint), while the choice
    among allowed tokens is still corruption-weighted. If every token is below the floor
    (degenerate), fall back to the most target-likely token (argmax of softmax(t))."""
    with torch.no_grad():
        B = len(t_prefs)
        ti, ta = _hf_left_pad(t_prefs, pad_id, device)
        ci, ca = _hf_left_pad(c_prefs, pad_id, device)
        to = mt(input_ids=ti, attention_mask=ta, use_cache=True, logits_to_keep=1)
        co = mc(input_ids=ci, attention_mask=ca, use_cache=True, logits_to_keep=1)
        tp, cp = to.past_key_values, co.past_key_values
        tl = to.logits[:, -1, :].float(); cl = co.logits[:, -1, :].float()
        taf, caf = ta, ca
        # Sampling logit is a linear combination of the three distributions:
        #     z = eb1*t + eb2*c - eb3*n
        # eb1 = target weight (default 1), eb2 = jail/corruption weight (c = corruptor on the
        # offensive prompt), eb3 = negative-steering weight (n = corruptor on the NEUTRAL prompt;
        # 0 = off). The neutral pass runs only when eb3!=0; the floor masks on softmax(t).
        eb1 = float(poe_b1 if poe_b1 is not None else 1.0)
        eb2 = float(poe_b2 if poe_b2 is not None else 0.0)
        eb3 = float(poe_b3 if poe_b3 is not None else 0.0)
        cfg_on = (eb3 != 0.0) and (n_prefs is not None)
        ncp = naf = nl = None
        if cfg_on:
            ni, na = _hf_left_pad(n_prefs, pad_id, device)
            no = mc(input_ids=ni, attention_mask=na, use_cache=True, logits_to_keep=1)
            ncp = no.past_key_values; nl = no.logits[:, -1, :].float(); naf = na
        gen: List[List[int]] = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        tlps: List[List[float]] = [[] for _ in range(B)]   # per-token true-target logprob (return_token_lps)
        for _ in range(max_new):
            z = eb1 * tl + eb2 * cl - (eb3 * nl if cfg_on else 0.0)
            if tokbias is not None and tokbias[1] is not None:
                z = z + tokbias[0] * tokbias[1]      # static vocab tilt (logit-bias baseline)
            probs = torch.softmax(z / max(temperature, 1e-6), -1)
            if target_floor > 0.0:
                # threshold on the TRUE target dist, but sample from the PoE among survivors
                tprobs = torch.softmax(tl, dim=-1)
                keep = tprobs >= target_floor
                masked = torch.where(keep, probs, torch.zeros_like(probs))
                s = masked.sum(dim=-1, keepdim=True)
                dead = (s.squeeze(-1) <= 0)
                if bool(dead.any()):
                    am = tprobs.argmax(dim=-1)   # most target-likely token
                    onehot = torch.zeros_like(masked)
                    onehot[torch.arange(B, device=device), am] = 1.0
                    masked = torch.where(dead.unsqueeze(-1), onehot, masked)
                    s = masked.sum(dim=-1, keepdim=True)
                probs = masked / s.clamp_min(1e-12)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            if return_token_lps:
                # True-target logprob of the sampled token; tl was already computed this step
                # so this is ~free vs a separate scoring forward. logsumexp avoids the full
                # [B, vocab] log_softmax tensor (the OOM-prone bit of the old _hf_score). This is
                # the UNMODIFIED-target (temp=1) logprob — the on-policy plausibility of the token
                # the PoE/jail actually sampled — NOT the steered z-logprob it was drawn from.
                tlp = tl.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(tl, dim=-1)
                live = (~done) & (nxt != eos_id)
                for i in range(B):
                    if bool(live[i]):
                        tlps[i].append(float(tlp[i]))
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            for i in range(B):
                if not done[i]: gen[i].append(int(nxt[i]))
            done = done | (nxt == eos_id)
            if bool(done.all()): break
            ones = torch.ones(B, 1, dtype=torch.long, device=device)
            taf = torch.cat([taf, ones], -1); caf = torch.cat([caf, ones], -1)
            to = mt(input_ids=nxt.unsqueeze(-1), attention_mask=taf, past_key_values=tp, use_cache=True)
            co = mc(input_ids=nxt.unsqueeze(-1), attention_mask=caf, past_key_values=cp, use_cache=True)
            tp, cp = to.past_key_values, co.past_key_values
            tl = to.logits[:, -1, :].float(); cl = co.logits[:, -1, :].float()
            if cfg_on:
                naf = torch.cat([naf, ones], -1)
                no = mc(input_ids=nxt.unsqueeze(-1), attention_mask=naf, past_key_values=ncp, use_cache=True)
                ncp = no.past_key_values; nl = no.logits[:, -1, :].float()
        if return_token_lps:
            return gen, tlps
        return gen


def _jail_generate_hf(hf: Dict, jail_runtime_cfg: Dict,
                      target_msgs_batch: List[List[Dict]], max_tokens: int,
                      temperature: float, no_think_target: bool) -> List[Dict]:
    """[jail engine=hf_full] n=1 full-vocab PoE jail sampling (HF analogue of
    batch_generate_contrastive_local; target+jail both HF here). Returns one
    {"best_text","best_ids","best_token_probs"} per scenario. GOTCHA: SAMPLES from
    z=target+beta*jail, but best_token_probs are the on-policy UNMODIFIED-target (temp=1)
    probs captured during the decode (return_token_lps), no separate scoring pass — that is
    the plausibility metric we report."""
    mt, mc, tok, tok_c, device = hf["mt"], hf["mc"], hf["tok"], hf["tok_c"], hf["device"]
    pad_id, eos_id = hf["pad_id"], hf["eos_id"]
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill    = jail_runtime_cfg.get("prefill", "") or ""
    beta       = float(jail_runtime_cfg.get("b2", 2.0))
    jail_b1    = jail_runtime_cfg.get("b1")                          # None => legacy z=target+beta*jail; set (e.g. 0) => z=b1*target+beta*jail, floor-masked (floor-only jail = b1=0, beta=1, floor>0)
    jail_floor = float(jail_runtime_cfg.get("target_floor", 0.0) or 0.0)
    # ---- BoN / vanilla baseline (beta=0 WILT): plain target sampling, NO jail model, NO floor. ----
    # The vanilla reference is the unmodified target sampled once; on-policy token probs are captured
    # DURING the decode (return_token_lps) so token stats are free. No PoE / second distribution runs,
    # so this can decode at a larger batch than a jail run.
    if bool(jail_runtime_cfg.get("target_only")):
        _NT = hf.get("target_no_think", core._NO_THINK_PREFIX)
        t_pres: List[List[int]] = []
        for tm in target_msgs_batch:
            ts = tok.apply_chat_template(tm, tokenize=False, add_generation_prompt=True)
            if no_think_target:
                ts += _NT
            t_pres.append(tok.encode(ts, add_special_tokens=False))
        # tokbias baseline: add the static vocab tilt to the plain BoN sampling (no-op when disabled).
        baselines, base_tlps = _hf_generate(mt, t_pres, max_tokens, temperature,
                                            pad_id, eos_id, device, return_token_lps=True,
                                            tokbias=_tokbias_vector(mt, tok, device, jail_runtime_cfg.get("tokbias", {})))
        out: List[Dict] = []
        for i, base_ids in enumerate(baselines):
            ids = [x for x in base_ids if x != eos_id]
            txt = tok.decode(ids, skip_special_tokens=True).strip()
            tprobs = [math.exp(l) * 100 for l in base_tlps[i]]
            out.append({"best_text": txt, "best_ids": ids, "best_token_probs": tprobs})
        return out
    # ---- NEGATIVE STEERING for jail (input-conditioned CFG, NOT a rewrite) ----
    # z = target + beta*jail - b3*neg, where `neg` is the SAME jail expert (mc) prompted with a
    # NEGATIVE system prompt and the SAME conversation/input (a continuation), NOT a rewrite of the
    # output. This is the jail-space analogue of corruption's b3 term: subtracting a refusal-heavy
    # persona ('rc') cancels the refusal direction; subtracting a bland assistant ('neutral') is
    # plain CFG sharpening. b3=0 (default) => legacy jail (no negative term).
    jail_b3      = float(jail_runtime_cfg.get("b3", 0.0) or 0.0)
    jail_neg_sys = jail_runtime_cfg.get("neg_system_prompt", "") or ""
    jail_neg_user = jail_runtime_cfg.get("neg_user_prompt", "") or ""   # replaces the conversation on the neg branch
    jail_neg_pf  = jail_runtime_cfg.get("neg_prefill", "") or ""    # default: no prefill on the negative branch
    NO_THINK = hf.get("target_no_think", core._NO_THINK_PREFIX)       # target wrapper
    NO_THINK_C = hf.get("corrupt_no_think", core._CORRUPT_NO_THINK_PREFIX)  # jail/corruptor wrapper
    neg_on = (jail_b3 != 0.0) and (bool(jail_neg_sys) or bool(jail_neg_user))
    t_prefs: List[List[int]] = []
    j_prefs: List[List[int]] = []
    n_prefs: List[List[int]] = []
    for tm in target_msgs_batch:
        ts = tok.apply_chat_template(tm, tokenize=False, add_generation_prompt=True)
        if no_think_target:
            ts += NO_THINK
        t_prefs.append(tok.encode(ts, add_special_tokens=False))
        conv = [m for m in tm if m.get("role") != "system"]
        j_msgs = ([{"role": "system", "content": sys_prompt}] + conv) if sys_prompt else conv
        # jail prompt is generated by the CORRUPTOR -> corruptor tokenizer + wrapper
        js = tok_c.apply_chat_template(j_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_C
        if prefill:
            js += prefill
        j_prefs.append(tok_c.encode(js, add_special_tokens=False))
        if neg_on:
            # NEGATIVE branch: identical construction (same conversation/input, same corruptor
            # tokenizer + wrapper), only the SYSTEM PROMPT swapped to the negative persona. It is
            # a CONTINUATION of the input, NOT a rewrite of the target output.
            if jail_neg_user:
                # ELICITED refusal: the conversation is REPLACED by a harmful request, so the
                # subtracted direction is the refusal the model actually produces for it,
                # continued along the tokens generated so far.
                n_msgs = ([{"role": "system", "content": jail_neg_sys}] if jail_neg_sys else []) \
                         + [{"role": "user", "content": jail_neg_user}]
            else:
                n_msgs = [{"role": "system", "content": jail_neg_sys}] + conv
            ns = tok_c.apply_chat_template(n_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_C
            if jail_neg_pf:
                ns += jail_neg_pf
            n_prefs.append(tok_c.encode(ns, add_special_tokens=False))
    # One jail sample per scenario: a single PoE decode over the batch. On-policy
    # (unmodified-target, temp=1) token probs are captured inline during the decode.
    # z = b1*target + b2*jail - b3*neg (b1=None => target weight 1). The negative-steering pass
    # (n_prefs) runs only when neg_on (b3 != 0); tokbias adds a static vocab tilt when configured.
    _b1 = float(jail_b1) if jail_b1 is not None else 1.0
    gen, tlps = _hf_poe_generate(mt, mc, t_prefs, j_prefs, temperature, max_tokens,
                                 pad_id, eos_id, device, target_floor=jail_floor,
                                 n_prefs=(n_prefs if neg_on else None),
                                 poe_b1=_b1, poe_b2=beta, poe_b3=(jail_b3 if neg_on else 0.0),
                                 return_token_lps=True,
                                 tokbias=_tokbias_vector(mt, tok, device, jail_runtime_cfg.get("tokbias", {})))
    out: List[Dict] = []
    for g, lps in zip(gen, tlps):
        ids = [x for x in g if x != eos_id]
        txt = tok.decode(ids, skip_special_tokens=True).strip()
        tprobs = [math.exp(l) * 100 for l in lps]           # on-policy token probs (%)
        out.append({"best_text": txt, "best_ids": ids, "best_token_probs": tprobs})
    return out


def _contrastive_sample_extensions(
    lm_target: "LocalModel",
    lm_jail: "LocalModel",
    target_prefixes: List[List[int]],
    jail_prefixes: List[List[int]],
    n: int,
    max_tokens: int,
    beta: float,
    top_k_logprobs: int,
    temperature: float,
    top_p: float,
    allowed_token_ids: Optional[List[int]] = None,
    ignore_eos: bool = False,
    eos_token_id: Optional[int] = None,
) -> List[List[List[int]]]:
    """Sample `n` extensions of `max_tokens` tokens via weighted PoE:
    softmax(log p_target + beta · log p_jailbroken).

    Implementation: at each step, jail proposes its top-K next-token logprobs;
    those are passed as logit_bias to target's sampler. Target samples from
    target_logits + beta * log p_jail (within jail's top-K; tokens outside get
    bias=0). The result is the exact PoE distribution on jail's top-K and the
    unbiased target distribution elsewhere.

    Returns same shape as `_vllm_sample_extensions`:
      per prompt → list of n candidates → list of token IDs (length max_tokens).

    The caller passes parallel target/jail prefix lists of the same length P
    (one entry per scenario/prompt). Each prefix is expanded to `n` parallel
    sampling slots, and we step all P*n slots in lockstep across both models.
    Final results are reorganised back into the P x n layout.
    """
    P = len(target_prefixes)
    if P != len(jail_prefixes):
        raise ValueError(f"target_prefixes ({P}) and jail_prefixes ({len(jail_prefixes)}) must align")
    if P == 0 or n < 1 or max_tokens < 1:
        return [[[] for _ in range(n)] for _ in range(P)]

    # Materialise n*P parallel slots.
    t_states: List[List[int]] = []
    j_states: List[List[int]] = []
    for i in range(P):
        for _ in range(n):
            t_states.append(list(target_prefixes[i]))
            j_states.append(list(jail_prefixes[i]))
    sampled_tokens: List[List[int]] = [[] for _ in t_states]
    done: List[bool] = [False] * len(t_states)

    # Allowed-token mask intersection for vLLM (Latin mask, etc.). We let
    # max_tokens=1 sampling enforce it on the target side. Jail's proposal
    # logprobs we read directly — no need to restrict its top-K to the mask
    # since target's allowed_token_ids will only sample from the intersection.
    jail_step_kwargs = dict(
        max_tokens=1,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        logprobs=top_k_logprobs,
        skip_special_tokens=False,
        ignore_eos=ignore_eos,
    )

    for step in range(max_tokens):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break

        active_t_prompts = [t_states[i] for i in active]
        active_j_prompts = [j_states[i] for i in active]

        # 1) Jail step → top-K logprobs per active slot.
        jail_out = lm_jail.worker.step_with_logprobs(active_j_prompts, jail_step_kwargs)

        # 2) Build per-prompt logit_bias and issue one batched target call.
        # vLLM accepts list[SamplingParams] in the same generate() call, one per
        # prompt — so we get per-prompt biases without IPC overhead per slot.
        per_prompt_kwargs: List[Dict] = []
        for (_tok, _slp, jail_topk) in jail_out:
            kwargs = dict(
                max_tokens=1,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                logprobs=1,
                skip_special_tokens=False,
                ignore_eos=ignore_eos,
                logit_bias={tid: beta * lp for tid, lp in jail_topk.items()},
            )
            if allowed_token_ids is not None:
                kwargs["allowed_token_ids"] = allowed_token_ids
            per_prompt_kwargs.append(kwargs)
        target_outs = lm_target.worker.step_with_logprobs(active_t_prompts, per_prompt_kwargs)

        # 3) Append sampled tokens to both prefix sides and mark EOS hits.
        for idx_in_active, slot_idx in enumerate(active):
            tok_id, _lp, _topk = target_outs[idx_in_active]
            if tok_id < 0:
                done[slot_idx] = True
                continue
            sampled_tokens[slot_idx].append(tok_id)
            t_states[slot_idx].append(tok_id)
            j_states[slot_idx].append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id and not ignore_eos:
                done[slot_idx] = True

    # Reorganise back to P x n shape.
    out: List[List[List[int]]] = []
    for i in range(P):
        out.append([sampled_tokens[i * n + j] for j in range(n)])
    return out

__all__ = ['_jail_generate_trs', '_jail_generate_trs_batch', 'batch_generate_contrastive_local', '_load_hf_poe_models', '_TOKBIAS_CACHE', '_tokbias_vector', '_hf_poe_generate', '_jail_generate_hf', '_contrastive_sample_extensions']
