"""FLRT-style input-side search (`flrt_search_input`).

A black-box, gradient-free port of "Fluent student-teacher redteaming" (Thompson &
Sklar, Confirm Labs, arXiv:2407.17447) adapted to BLOOM's auditor/target setup.

Differences from the two search methods already in `search.py`:
  • Search loop is a MUTATION BUFFER (append/insert/delete/swap), not append-only
    beam top-K → it needs its own loop (`_flrt_single_trial`), NOT `_beast_single_trial_local`.
  • Scoring is the FULL-VOCAB distillation loss L_D (a whole-vocab expectation), so it
    runs on HuggingFace logits (the `jail_runtime_cfg["hf"]` handles that `jailbroken_output`
    engine=hf_full already loads), NOT vLLM top-K.

Roles:
  • auditor  (lm_eval, vLLM)      — proposes candidate mutation tokens for the attack message.
  • target   (hf["mt"], HF)       — the victim; L_D pulls its per-token distribution toward the teacher.
  • self-jail teacher (hf["mc"], HF, jail system prompt + prefill) — supplies the continuation
    and the soft target distribution p_jail. For self-jail hf["mc"] IS hf["mt"] (same weights,
    different prompt).

This module touches NO existing method — it composes existing helpers (`_hf_left_pad`,
`_hf_generate`, `_extract_message_tags`, `build_latin_token_ids`, `_vllm_sample_extensions`).

Defaults follow the ORIGINAL PAPER except the agreed adaptations (self-jail teacher; HF
engine; teacher task-only by default; append split out as the highest-probability op).
"""

import math
import random
import re
from typing import Dict, List, Optional, Tuple

import torch

from . import core
from .core import (
    _hf_left_pad,
    _hf_generate,
    _extract_message_tags,
    _strip_eos_tail,
    _vllm_sample_extensions,
    _resolve_eos_token_id,
)
from .search import build_latin_token_ids


# ── Mutation proposal (candidate tokens from the AUDITOR's distribution) ──────

def _sample_at(lm_eval, prompts_tids: List[List[int]], n: int,
               allowed_token_ids: Optional[List[int]], temperature: float) -> List[List[int]]:
    """Sample `n` single tokens per prompt from the auditor (vLLM). Returns, per prompt,
    a flat list of n sampled token ids. Reuses `_vllm_sample_extensions` (max_tokens=1)."""
    if not prompts_tids:
        return []
    ext = _vllm_sample_extensions(
        lm_eval, prompts_tids, n=n, max_tokens=1, temperature=temperature, top_p=1.0,
        allowed_token_ids=allowed_token_ids, ignore_eos=True,
    )
    # ext: per prompt → n candidates → [tok]; flatten each candidate's single token.
    return [[c[0] for c in per_prompt if c] for per_prompt in ext]


def _mutate(lm_eval, best_seq: List[int], prefix_length: int, mutation_type: str,
            k1: int, allowed_token_ids: Optional[List[int]], temperature: float,
            min_tokens: int, max_tokens, n_mut: int = 1) -> List[List[int]]:
    """Produce up to k1 mutated candidates of `best_seq` (mutating only the SUFFIX region
    [prefix_length:]). One mutation TYPE per call (chosen by the caller); the chosen operator
    is applied `n_mut` times to each candidate. Because every candidate shares the operator
    AND the count, they all take the SAME token-length delta (append/insert +n_mut, delete −m,
    swap ±0), so the returned batch stays rectangular for lockstep scoring.

    • append : sample an n_mut-token continuation per candidate → append it (BEAST-style end-insert).
    • swap   : apply a single-position swap n_mut times (sampled replacement each time).
    • insert : apply a single-position insert n_mut times (sampled token each time).
    • delete : drop n_mut distinct random suffix positions (no model call). GUARDED so the
               suffix never falls below `min_tokens` (m = min(n_mut, suffix_len − min_tokens)).
    """
    n_mut = max(1, int(n_mut))
    suffix_len = len(best_seq) - prefix_length
    if mutation_type == "append":
        # one auditor call: k1 continuations of length n_mut (autoregressive = appending n_mut tokens).
        ext = _vllm_sample_extensions(
            lm_eval, [best_seq], n=k1, max_tokens=n_mut, temperature=temperature, top_p=1.0,
            allowed_token_ids=allowed_token_ids, ignore_eos=True,
        )
        cands = ext[0] if ext else []
        return [list(best_seq) + list(c) for c in cands if len(c) == n_mut]

    if mutation_type == "delete":
        # GUARD: never shrink the suffix below min_tokens.
        m = min(n_mut, suffix_len - int(min_tokens))
        if m <= 0:
            return []
        out: List[List[int]] = []
        for _ in range(k1):
            positions = sorted(random.sample(range(prefix_length, len(best_seq)), m), reverse=True)
            seq = list(best_seq)
            for pos in positions:
                del seq[pos]
            out.append(seq)
        return out

    # swap / insert: apply the single-position operator n_mut times to each of k1 candidates.
    if suffix_len <= 0 and mutation_type == "swap":
        return []
    seqs = [list(best_seq) for _ in range(k1)]
    for _ in range(n_mut):
        positions = [prefix_length + random.randrange(max(len(s) - prefix_length, 1)) for s in seqs]
        prompts = [s[:p] for s, p in zip(seqs, positions)]     # auditor sees the seq up to the mutation site
        per_prompt_tok = _sample_at(lm_eval, prompts, n=1, allowed_token_ids=allowed_token_ids,
                                    temperature=temperature)
        new_seqs = []
        for s, p, toks in zip(seqs, positions, per_prompt_tok):
            if not toks:
                new_seqs.append(s)                              # sample failed → leave as-is (filtered below)
                continue
            t = toks[0]
            if mutation_type == "swap":
                ns = list(s); ns[p] = t
            else:  # insert
                ns = s[:p] + [t] + s[p:]
            new_seqs.append(ns)
        seqs = new_seqs
    # keep only fully-mutated candidates so the batch is rectangular.
    want = len(best_seq) + (n_mut if mutation_type == "insert" else 0)
    return [s for s in seqs if len(s) == want]


def _pick_mutation(cfg, suffix_len: int) -> Optional[str]:
    """Pick a mutation type by the configured probabilities, disabling ops that would violate
    the min/max suffix-length bounds (paper behaviour: no delete when too short; no insert/append
    when too long)."""
    under_max = (cfg.max_tokens is None) or (suffix_len < cfg.max_tokens)   # None = unbounded suffix
    p_del = cfg.p_delete if suffix_len > cfg.min_tokens else 0.0
    p_ins = cfg.p_insert if under_max else 0.0
    p_app = cfg.p_append if under_max else 0.0
    p_swp = cfg.p_swap if suffix_len > 0 else 0.0
    total = p_del + p_ins + p_app + p_swp
    if total <= 1e-12:
        return None
    r = random.random() * total
    if r < p_del:
        return "delete"
    if r < p_del + p_ins:
        return "insert"
    if r < p_del + p_ins + p_app:
        return "append"
    return "swap"


# ── Self-jail teacher: prefix, continuation, and p_jail ──────────────────────

def _jail_teacher_prefix(hf: Dict, jail_runtime_cfg: Dict, target_msgs: List[Dict],
                         no_think_target: bool, task_msg: str) -> List[int]:
    """Build the self-jail teacher's prompt token ids (in the jail model's tokenizer):
    jail system prompt + conversation-so-far + task user-turn + generation prompt + no-think
    closer + prefill. Mirrors output_search_target_response's j_prompt construction.

    `task_msg` is the teacher's user turn — the Phase-1 baseline message (the base intent), NOT
    the searched attack. The teacher is always task-only (see flrt_search_input_one)."""
    tok_c = hf["tok_c"]
    sys_prompt = jail_runtime_cfg.get("system_prompt", "") or ""
    prefill = jail_runtime_cfg.get("prefill", "") or ""
    j_msgs = [m for m in target_msgs if m.get("role") != "system"]
    j_msgs = j_msgs + [{"role": "user", "content": task_msg}]
    if sys_prompt:
        j_msgs = [{"role": "system", "content": sys_prompt}] + j_msgs
    j_prompt = tok_c.apply_chat_template(j_msgs, tokenize=False, add_generation_prompt=True)
    if no_think_target:
        j_prompt += hf.get("corrupt_no_think", core._CORRUPT_NO_THINK_PREFIX)
    if prefill:
        j_prompt += prefill
    return tok_c.encode(j_prompt, add_special_tokens=False)


@torch.no_grad()
def _teacher_continuation_and_pjail(
    hf: Dict, jail_prefix_ids: List[int], reward_len: int, temperature: float,
) -> Tuple[List[int], Optional[torch.Tensor]]:
    """Sample the teacher's continuation from the jail model, then teacher-force to read its
    full per-position distribution p_jail over that continuation.

    Returns (continuation_ids, p_jail) where p_jail is [T, V] (softmax, T = #continuation tokens),
    or ([], None) if the teacher produced nothing."""
    mc, device = hf["mc"], hf["device"]
    pad_id, eos_id = hf["pad_id"], hf["eos_id"]
    gen = _hf_generate(mc, [jail_prefix_ids], max_new=reward_len, temperature=temperature,
                       pad_id=pad_id, eos_id=eos_id, device=device)
    cont = [t for t in gen[0] if t != eos_id]
    if not cont:
        return [], None
    T = len(cont)
    full = torch.tensor([jail_prefix_ids + cont], dtype=torch.long, device=device)
    out = mc(input_ids=full, use_cache=False, logits_to_keep=T + 1)
    logits = out.logits[0, :T, :].float()            # position i predicts continuation token i
    p_jail = torch.softmax(logits, dim=-1)           # [T, V]
    return cont, p_jail


# ── The distillation scorer (full-vocab L_D, higher = better) ────────────────

@torch.no_grad()
def _score_distillation(
    hf: Dict, victim_prefixes: List[List[int]], continuation_ids: List[int],
    p_jail: torch.Tensor, p_threshold: float, chunk: Optional[int],
    w_distillation: float = 1.0, w_force: float = 0.0,
) -> List[float]:
    """Two per-candidate terms, computed over the SAME reward positions (first token weighted x1.5):
      • distillation L_D = mean_i pos_w[i] * min( sum_x p_jail(x) log q_base(x),  log p_threshold )
      • teacher forcing  = mean_i pos_w[i] * min( log q_base(forced_token_i),     log p_threshold )
    q_base is the TARGET (victim) reading [candidate message + continuation]; p_jail the fixed
    teacher distribution; forced_token_i = continuation_ids[i] (the teacher's actually-sampled
    token). The forcing term is free — it's a single gather from the log_q we already build.

    w_force <= 0  → returns the raw distillation score (exact legacy behaviour, no z-norm).
    w_force >  0  → both terms are z-normed across the scored candidate batch (so the weights are
                    on the same scale and actually control the trade-off) and combined as
                    w_distillation * z(L_D) + w_force * z(forcing). Higher = better.
    """
    mt, device, pad_id = hf["mt"], hf["device"], hf["pad_id"]
    T = len(continuation_ids)
    cap = math.log(max(p_threshold, 1e-6))
    pos_w = torch.ones(T, device=device)
    if T > 0:
        pos_w[0] = 1.5
    w_sum = float(pos_w.sum())
    do_force = w_force is not None and w_force > 0.0

    full_seqs = [vp + continuation_ids for vp in victim_prefixes]
    distill_scores: List[float] = []
    force_scores: List[float] = []
    step = chunk or len(full_seqs)
    for b in range(0, len(full_seqs), step):
        batch = full_seqs[b:b + step]
        inp, attn = _hf_left_pad(batch, pad_id, device)
        out = mt(input_ids=inp, attention_mask=attn, use_cache=False, logits_to_keep=T + 1)
        logits = out.logits[:, :T, :].float()                       # [B, T, V]
        Bn = logits.shape[0]
        d_acc = torch.zeros(Bn, device=device)
        f_acc = torch.zeros(Bn, device=device)
        for i in range(T):
            log_q = torch.log_softmax(logits[:, i, :], dim=-1)      # [B, V]
            d = (p_jail[i].unsqueeze(0) * log_q).sum(dim=-1)        # [B]  E_pjail[log q_base]
            d_acc = d_acc + pos_w[i] * torch.clamp(d, max=cap)      # paper cap (stop over-rewarding)
            if do_force:
                f = log_q[:, continuation_ids[i]]                   # [B]  log q_base(forced token) — free gather
                f_acc = f_acc + pos_w[i] * torch.clamp(f, max=cap)  # same per-token p_threshold cap
        distill_scores.extend((d_acc / max(w_sum, 1e-6)).tolist())
        if do_force:
            force_scores.extend((f_acc / max(w_sum, 1e-6)).tolist())

    if not do_force:
        return distill_scores                                       # legacy: raw distillation only

    def _z(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + 1e-6) if x.numel() > 1 else torch.zeros_like(x)

    d = torch.tensor(distill_scores)
    f = torch.tensor(force_scores)
    combined = w_distillation * _z(d) + w_force * _z(f)
    return combined.tolist()


# ── Auxiliary losses (default off) ───────────────────────────────────────────

@torch.no_grad()
def _fluency_reward(hf, lm_eval, suffix_texts: List[str], fluency_on: str) -> List[float]:
    """Fluency reward = -perplexity(attack tokens). Scored on the target HF model ("target")
    or, when "auditor", still on the target HF model if the auditor is not an HF handle here
    (the auditor is vLLM); we keep it simple and use the target HF model for perplexity.
    Returns higher = more fluent (less negative)."""
    mt, tok, device, pad_id = hf["mt"], hf["tok"], hf["device"], hf["pad_id"]
    seqs = [tok.encode(t, add_special_tokens=False) for t in suffix_texts]
    out_scores: List[float] = []
    for s in seqs:
        if len(s) < 2:
            out_scores.append(0.0)
            continue
        inp = torch.tensor([s], dtype=torch.long, device=device)
        logits = mt(input_ids=inp, use_cache=False).logits[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = torch.tensor(s[1:], device=device)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        xent = float(-tok_lp.mean())
        out_scores.append(-math.exp(min(xent, 20.0)))              # -perplexity, higher = better
    return out_scores


def _repetition_penalty(suffix_token_lists: List[List[int]], exponent: float) -> List[float]:
    """Repetition penalty (paper): sum((count(tok)-1)^exponent) / len. Higher = more repetitive
    (worse). Free — token counts only, no model."""
    out: List[float] = []
    for s in suffix_token_lists:
        if not s:
            out.append(0.0)
            continue
        counts: Dict[int, int] = {}
        for t in s:
            counts[t] = counts.get(t, 0) + 1
        pen = sum(max(c - 1, 0) ** exponent for c in counts.values())
        out.append(pen / len(s))
    return out


# ── The mutation-buffer search loop (one trial) ──────────────────────────────

@torch.no_grad()
def _flrt_single_trial(
    lm_eval, hf, cfg, prefix_tokens: List[int], continuation_ids: List[int],
    p_jail: torch.Tensor, baseline_prefix: str, target_msgs: List[Dict],
    allowed_token_ids: Optional[List[int]], eos_token_id: Optional[int],
    no_think_target: bool,
) -> Tuple[List[List[int]], List[float]]:
    """One FLRT trial: init a suffix, then mutate the buffer's best each iteration, scoring
    every candidate by the combined objective (L_D [+ fluency] [+ repetition]). Returns the
    all-time pool (token sequences, scores) — higher score = better."""
    prefix_length = len(prefix_tokens)
    tok_e = lm_eval.tokenizer

    # ── init suffix: always sampled autoregressively from the auditor ──
    ext = _vllm_sample_extensions(
        lm_eval, [list(prefix_tokens)], n=1, max_tokens=int(cfg.start_tokens),
        temperature=cfg.temperature, top_p=1.0,
        allowed_token_ids=allowed_token_ids, ignore_eos=True,
    )
    suffix = list(ext[0][0]) if ext and ext[0] and ext[0][0] else []

    current = list(prefix_tokens) + suffix

    def _score(seqs: List[List[int]]) -> List[float]:
        # Build victim prefixes (target reads the extracted message body) + aux inputs.
        victim_prefixes: List[List[int]] = []
        suffix_texts: List[str] = []
        suffix_tok_lists: List[List[int]] = []
        for seq in seqs:
            suf_ids = _strip_eos_tail(seq[prefix_length:], eos_token_id)
            suf_txt = tok_e.decode(suf_ids, skip_special_tokens=False)
            full_text = baseline_prefix + suf_txt
            extracted, _, _ = _extract_message_tags(full_text)
            msg_text = extracted if extracted else full_text
            v_prompt = hf["tok"].apply_chat_template(
                list(target_msgs) + [{"role": "user", "content": msg_text}],
                tokenize=False, add_generation_prompt=True,
            )
            if no_think_target:
                v_prompt += hf.get("target_no_think", core._NO_THINK_PREFIX)
            victim_prefixes.append(hf["tok"].encode(v_prompt, add_special_tokens=False))
            suffix_texts.append(suf_txt)
            suffix_tok_lists.append(suf_ids)

        ld = _score_distillation(hf, victim_prefixes, continuation_ids, p_jail,
                                 cfg.p_threshold, cfg.eval_beam_chunk_size,
                                 float(getattr(cfg, "w_distillation", 1.0)),
                                 float(getattr(cfg, "w_force", 0.0)))
        total = list(ld)
        if cfg.w_fluency and cfg.w_fluency > 0:
            fl = _fluency_reward(hf, lm_eval, suffix_texts, cfg.fluency_on)  # f = -perplexity (higher = better)
            total = [t + cfg.w_fluency * f for t, f in zip(total, fl)]
        if cfg.w_repetition and cfg.w_repetition > 0:
            rep = _repetition_penalty(suffix_tok_lists, cfg.repetition_exponent)
            total = [t - cfg.w_repetition * r for t, r in zip(total, rep)]
        return total

    init_score = _score([current])[0]
    buffer: List[Tuple[float, List[int]]] = [(init_score, current)]
    pool_seqs: List[List[int]] = [current]
    pool_scores: List[float] = [init_score]

    for _ in range(int(cfg.max_num_iterations)):
        buffer.sort(key=lambda x: x[0])
        best_score, best_seq = buffer[-1]
        suffix_len = len(best_seq) - prefix_length
        mtype = _pick_mutation(cfg, suffix_len)
        if mtype is None:
            continue
        new_seqs = _mutate(lm_eval, best_seq, prefix_length, mtype, int(cfg.k1),
                           allowed_token_ids, cfg.temperature, cfg.min_tokens, cfg.max_tokens,
                           int(getattr(cfg, "num_mutations", 1)))
        if not new_seqs:
            continue
        new_scores = _score(new_seqs)

        # retain top buffer_size from (buffer minus the mutated best) ∪ new candidates
        remaining = buffer[:-1] + list(zip(new_scores, new_seqs))
        remaining.sort(key=lambda x: x[0])
        buffer = remaining[-int(cfg.buffer_size):]

        pool_seqs.extend(new_seqs)
        pool_scores.extend(new_scores)
        if len(pool_scores) > int(cfg.max_pool_size):
            top = sorted(range(len(pool_scores)), key=lambda i: pool_scores[i])[-int(cfg.max_pool_size):]
            pool_seqs = [pool_seqs[i] for i in top]
            pool_scores = [pool_scores[i] for i in top]

    return pool_seqs, pool_scores


# ── Teacher continuation preparation (single + cross-scenario batch) ─────────

def flrt_prepare_continuation(
    hf: Dict, teacher_cfg: Dict, target_msgs: List[Dict], baseline_msg: str,
    no_think_target: bool, reward_len: int, temperature: float,
) -> Tuple[List[int], Optional["torch.Tensor"], str]:
    """Sample the self-jail teacher's continuation for ONE scenario and read its full per-position
    p_jail. Returns (continuation_ids, p_jail [T,V], continuation_text); ([], None, "") if empty."""
    reward_len = reward_len if (reward_len and reward_len > 0) else 100
    jail_prefix = _jail_teacher_prefix(hf, teacher_cfg, target_msgs, no_think_target, task_msg=baseline_msg)
    cont_ids, p_jail = _teacher_continuation_and_pjail(hf, jail_prefix, reward_len, temperature)
    if not cont_ids or p_jail is None:
        return [], None, ""
    return cont_ids, p_jail, hf["tok"].decode(cont_ids, skip_special_tokens=True)


@torch.no_grad()
def flrt_prepare_continuation_batch(
    hf: Dict, teacher_cfg: Dict, target_msgs_list: List[List[Dict]], baseline_msgs: List[str],
    no_think_target: bool, reward_len: int, temperature: float,
) -> List[Tuple[List[int], Optional["torch.Tensor"], str]]:
    """Cross-scenario batch of flrt_prepare_continuation. The expensive autoregressive teacher
    GENERATION is batched across scenarios in one _hf_generate call; the (cheap, single-forward)
    p_jail teacher-forcing is then read per scenario. Returns a list aligned with the inputs.

    This is the cross-scenario-batchable step (teacher is task-only, so each continuation depends
    only on its scenario's baseline, not on any per-candidate attack)."""
    if not target_msgs_list:
        return []
    mc, device, pad_id, eos_id = hf["mc"], hf["device"], hf["pad_id"], hf["eos_id"]
    reward_len = reward_len if (reward_len and reward_len > 0) else 100
    jail_prefixes = [
        _jail_teacher_prefix(hf, teacher_cfg, tm, no_think_target, task_msg=bm)
        for tm, bm in zip(target_msgs_list, baseline_msgs)
    ]
    gens = _hf_generate(mc, jail_prefixes, max_new=reward_len, temperature=temperature,
                        pad_id=pad_id, eos_id=eos_id, device=device)   # BATCHED generation
    out: List[Tuple[List[int], Optional["torch.Tensor"], str]] = []
    for jp, gen in zip(jail_prefixes, gens):
        cont = [t for t in gen if t != eos_id]
        if not cont:
            out.append(([], None, ""))
            continue
        T = len(cont)
        full = torch.tensor([jp + cont], dtype=torch.long, device=device)
        logits = mc(input_ids=full, use_cache=False, logits_to_keep=T + 1).logits[0, :T, :].float()
        out.append((cont, torch.softmax(logits, dim=-1), hf["tok"].decode(cont, skip_special_tokens=True)))
    return out


# ── Public entry (mirrors search._input_search_beast_one) ────────────────────

def flrt_search_input_one(
    lm_eval, hf: Dict,
    eval_msgs: List[Dict], target_msgs: List[Dict],
    content: str, baseline_msg: str, strategy: str,
    search_cfg, no_think_eval: bool, no_think_target: bool,
    continuation: Tuple[List[int], "torch.Tensor", str],
) -> Tuple[List[Tuple[str, float, str, str]], str, str]:
    """Per-scenario FLRT input search. Given the Phase-1 auditor message (`content` = raw tagged
    output, `baseline_msg` = clean body), generate the self-jail teacher continuation, then run
    the mutation-buffer search scored by the full-vocab distillation loss.

    Returns (pool, continuation_text, strategy), where pool = List[(msg_text, score, baseline, suffix)]
    best-first (baseline + suffix == msg_text) — same shape as _input_search_beast_one so the
    rollout consumes it identically. continuation_text is returned in the trs slot (viewer info)."""
    from .search import _get_or_build_latin_mask, _build_sampling_prefix

    if not search_cfg.enabled:
        return [(baseline_msg, 0.0, baseline_msg, "")], "", strategy

    eos_token_id = _resolve_eos_token_id(lm_eval, search_cfg.truncate_at_eos)

    # ── baseline_prefix: carry over part of Phase-1's <message> body (max_prefix_length) ──
    # Identical slicing logic to search._input_search_beast_one.
    mpl = search_cfg.max_prefix_length
    msg_open = re.search(r"<message>", content)
    msg_close = re.search(r"</message>", content)
    if msg_open and msg_close and msg_open.end() <= msg_close.start():
        before_msg_text = content[:msg_open.end()]
        body_text = content[msg_open.end():msg_close.start()]
        if mpl is None:
            body_prefix = body_text
        elif mpl == 0:
            body_prefix = ""
        else:
            body_ids = lm_eval.tokenizer.encode(body_text, add_special_tokens=False)
            body_prefix = lm_eval.tokenizer.decode(body_ids[:mpl], skip_special_tokens=True)
        baseline_prefix = before_msg_text + body_prefix
    else:
        if mpl is None:
            baseline_prefix = baseline_msg
        elif mpl == 0:
            baseline_prefix = ""
        else:
            bm_ids = lm_eval.tokenizer.encode(baseline_msg, add_special_tokens=False)
            baseline_prefix = lm_eval.tokenizer.decode(bm_ids[:mpl], skip_special_tokens=True)

    # ── self-jail teacher continuation + p_jail (precomputed by the caller) ──
    # The teacher is ALWAYS task-only (conditions on the Phase-1 baseline message, never the
    # searched suffix), so the continuation depends only on the scenario and is prepared ONCE —
    # by flrt_prepare_continuation (serial) or flrt_prepare_continuation_batch (cross-scenario).
    continuation_ids, p_jail, continuation_text = continuation
    if not continuation_ids or p_jail is None:
        return [(baseline_msg, 0.0, baseline_msg, "")], "", strategy

    # ── auditor sampling prefix + Latin mask (reuse the shared helpers) ──
    _, prefix_tokens = _build_sampling_prefix(lm_eval, eval_msgs, no_think_eval, baseline_prefix)
    latin_token_ids = _get_or_build_latin_mask(
        lm_eval, search_cfg.latin_mask, eos_token_id,
        extra_chars="</>" if eos_token_id is not None else "",
        cache_tag="flrt_search_input", label="(flrt)",
    )

    print(f"    flrt input search buffer={search_cfg.buffer_size} k1={search_cfg.k1} "
          f"iters={search_cfg.max_num_iterations} n_trials={search_cfg.n_trials} "
          f"p_append={search_cfg.p_append} (cont_len={len(continuation_ids)}) ...", flush=True)

    prefix_length = len(prefix_tokens)
    global_pool_seqs: List[List[int]] = []
    global_pool_scores: List[float] = []
    for _ in range(int(search_cfg.n_trials)):
        ps, sc = _flrt_single_trial(
            lm_eval, hf, search_cfg, prefix_tokens, continuation_ids, p_jail,
            baseline_prefix, target_msgs, latin_token_ids, eos_token_id, no_think_target,
        )
        global_pool_seqs.extend(ps)
        global_pool_scores.extend(sc)
        if len(global_pool_scores) > int(search_cfg.max_pool_size):
            top = sorted(range(len(global_pool_scores)),
                         key=lambda i: global_pool_scores[i])[-int(search_cfg.max_pool_size):]
            global_pool_seqs = [global_pool_seqs[i] for i in top]
            global_pool_scores = [global_pool_scores[i] for i in top]

    # ── decode pool into (msg_text, score, baseline_view, suffix_view), best-first ──
    order = sorted(range(len(global_pool_scores)),
                   key=lambda i: global_pool_scores[i], reverse=True)
    pool: List[Tuple[str, float, str, str]] = []
    seen: set = set()
    for i in order:
        seq = global_pool_seqs[i]
        score = global_pool_scores[i]
        suffix_ids = _strip_eos_tail(seq[prefix_length:], eos_token_id)
        suffix_text = lm_eval.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        full_text = baseline_prefix + suffix_text
        extracted, _, _ = _extract_message_tags(full_text)
        msg_text = extracted if extracted else full_text
        carried_over, _, _ = _extract_message_tags(baseline_prefix) if baseline_prefix else ("", "", "")
        if carried_over and msg_text.startswith(carried_over):
            baseline_view = carried_over
            suffix_view = msg_text[len(carried_over):]
        else:
            baseline_view = ""
            suffix_view = msg_text
        if msg_text and msg_text not in seen:
            seen.add(msg_text)
            pool.append((msg_text, score, baseline_view, suffix_view))

    if not pool:
        pool = [(baseline_msg, 0.0, baseline_msg, "")]

    return pool, continuation_text, strategy


def flrt_search_input_message(
    lm_eval, hf: Dict, teacher_cfg: Dict, eval_msgs: List[Dict], target_msgs: List[Dict],
    search_cfg, no_think_eval: bool, no_think_target: bool,
    sample_max_tokens: int, sample_temperature: float,
    fixed_kickoff: Optional[Dict] = None,
) -> Tuple[List[Tuple[str, float, str, str]], str, str]:
    """Serial FLRT input search: Phase-1 auditor sample → FLRT mutation-buffer search. Mirrors
    search.input_search_evaluator_message's signature/return so the rollout forks between them
    with a single `if flrt_on`. Returns (pool, continuation_text, strategy).

    `fixed_kickoff` (from the kickoff bank): when present, its content is the reused Phase-1
    message and is used as the search BASELINE (skip the auditor Phase-1 generation), exactly
    like BEAST input_search — FLRT still runs its full search over the banked message (sliced by
    max_prefix_length). NOT frozen/verbatim."""
    from .core import batch_generate_local
    from .search import _parse_phase1

    if fixed_kickoff and fixed_kickoff.get("content"):
        baseline_msg = fixed_kickoff["content"]
        content = baseline_msg
        strategy = fixed_kickoff.get("strategy", "") or ""
    else:
        content, baseline_msg, strategy = _parse_phase1(
            batch_generate_local(lm_eval, [eval_msgs], sample_max_tokens, sample_temperature,
                                 no_think=no_think_eval)[0])
    if not baseline_msg or not search_cfg.enabled:
        return [(baseline_msg, 0.0, baseline_msg, "")], "", strategy
    continuation = flrt_prepare_continuation(
        hf, teacher_cfg, target_msgs, baseline_msg, no_think_target,
        search_cfg.max_reward_output_length, sample_temperature)
    return flrt_search_input_one(
        lm_eval, hf, eval_msgs, target_msgs,
        content, baseline_msg, strategy, search_cfg, no_think_eval, no_think_target, continuation)


__all__ = ["flrt_search_input_one", "flrt_search_input_message",
           "flrt_prepare_continuation", "flrt_prepare_continuation_batch"]
