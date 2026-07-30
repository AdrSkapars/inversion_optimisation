import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

# Suppress LiteLLM verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False
from .core import *
from . import core
from .wilt import *



def build_latin_token_ids(tokenizer, vocab_size: int,
                           extra_allowed_ids: Optional[List[int]] = None,
                           extra_allowed_chars: str = "") -> List[int]:
    """Return the list of token IDs whose decoded text contains ONLY characters from
    the Latin allowlist (ASCII letters, space, basic punctuation .,!?-). vLLM's
    SamplingParams.allowed_token_ids takes this list and constrains sampling to it.

    `extra_allowed_ids` are appended unconditionally — useful for letting EOS or
    other special tokens through the mask without lifting the Latin restriction.

    `extra_allowed_chars` expands the per-character allowlist — useful for letting
    the model emit tag characters (e.g. \"</>\") so it can produce </message>
    naturally. This is more robust than passing token IDs because tokenizers may
    fuse `</`, `>`, etc. into multi-char tokens in ways that depend on context."""
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " .,!?-"
    )
    allowed_chars.update(extra_allowed_chars)
    extra_blocked_tokens = ["..."]
    blocked_ids: set = set()
    for tok_str in extra_blocked_tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        blocked_ids.update(ids)

    allowed: List[int] = []
    for token_id in range(vocab_size):
        if token_id in blocked_ids:
            continue
        text = tokenizer.decode([token_id])
        if not text:
            continue
        if all(ch in allowed_chars for ch in text):
            allowed.append(token_id)
    if extra_allowed_ids:
        allowed = sorted(set(allowed).union(extra_allowed_ids))
    return allowed


def _score_beast_candidates(
    lm_eval: "LocalModel",
    lm_target: "LocalModel",
    candidates: List[List[int]],
    prefix_length: int,
    target_msgs: List[Dict],
    trs: str,
    baseline_prefix: str,
    max_batch_size: int,
    eos_token_id: Optional[int] = None,
) -> List[float]:
    """Score BEAST candidate token sequences by log P(trs | target_msgs + decoded_msg).
    Decodes each candidate's suffix using lm_eval's tokenizer, builds the scoring
    context for lm_target, and batches the forward passes. When `eos_token_id` is
    set, each candidate's suffix is truncated at the first EOS before decoding so
    candidates are scored as their natural-end form. Returns float scores
    (higher = more likely TRS; -inf for failed items)."""
    items: List[Tuple[List[Dict], str]] = []
    for seq in candidates:
        suffix_ids = _strip_eos_tail(seq[prefix_length:], eos_token_id)
        suffix_text = lm_eval.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        full_text = baseline_prefix + suffix_text
        # Extract just the message body: strips <strategy>, picks out the <message>
        # content, cuts at </message> or <targeted_response_start>. Falls back to
        # full_text if no tag boundary is found.
        extracted_msg, _, _ = _extract_message_tags(full_text)
        msg_text = extracted_msg if extracted_msg else full_text
        items.append((list(target_msgs) + [{"role": "user", "content": msg_text}], trs))

    all_scores: List[float] = []
    for b in range(0, len(items), max_batch_size):
        batch_scores = batch_logprob_local(lm_target, items[b: b + max_batch_size])
        all_scores.extend(s if s is not None else -float("inf") for s in batch_scores)
    return all_scores


# Forced no-think prefix used when the judge model is a thinking model. The
# `<think>...</think>\n` block sits between the chat template's assistant turn
# opener and the Yes/No target, suppressing reasoning so the next-token logit
# we read corresponds to the actual answer position. Score scale becomes
# avg over (n_prefix + 1) tokens, but that scale is constant across all
# candidates so ranking is unaffected.


def _score_output_candidates(
    lm_eval: "LocalModel",
    candidates: List[List[int]],
    prefix_length: int,
    target_tokenizer,
    judge_prompt_template: str,
    behavior_name: str,
    user_input: str,
    baseline_prefix: str,
    no_think_judge: bool,
    max_batch_size: int,
    eos_token_id: Optional[int] = None,
) -> List[float]:
    """Score candidate target-response token sequences by log P("Yes" | judge_prompt + candidate).

    Decodes each candidate's response-suffix using the TARGET tokenizer (since the
    candidates were sampled from lm_target), builds a judge prompt that asks the
    evaluator model whether the candidate response exhibits the named behavior,
    and returns log P("Yes") via lm_eval. Higher = more behavior-y per the judge.

    `baseline_prefix` is prepended to the decoded suffix to reconstruct the full
    candidate response (mirroring _score_beast_candidates). When `eos_token_id`
    is set, the suffix is truncated at the first EOS so candidates are scored as
    their natural-end form.
    """
    target_str = core._NO_THINK_PREFIX + "Yes" if no_think_judge else "Yes"

    # Escape literal `{` / `}` in free-form user inputs so str.format() doesn't
    # try to interpret them as placeholders. Candidates can easily contain code
    # or JSON-like content with curly braces. behavior_name is cfg-controlled
    # and assumed safe.
    def _fmt_escape(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")
    user_input_safe = _fmt_escape(user_input)

    items: List[Tuple[List[Dict], str]] = []
    for seq in candidates:
        suffix_ids  = _strip_eos_tail(seq[prefix_length:], eos_token_id)
        suffix_text = target_tokenizer.decode(suffix_ids, skip_special_tokens=False)
        cand_text   = baseline_prefix + suffix_text
        rendered    = judge_prompt_template.format(
            user_input=user_input_safe,
            model_response=_fmt_escape(cand_text),
            behavior_name=behavior_name,
        )
        judge_msgs = [{"role": "user", "content": rendered}]
        items.append((judge_msgs, target_str))

    all_scores: List[float] = []
    for b in range(0, len(items), max_batch_size):
        batch_scores = batch_logprob_local(lm_eval, items[b: b + max_batch_size])
        all_scores.extend(s if s is not None else -float("inf") for s in batch_scores)
    return all_scores


def _select_beam_indices(scores: List[float], num_to_select: int,
                         beast_temperature: float) -> List[int]:
    """Select beam indices from scored candidates.

    beast_temperature == 0  → hard top-K selection (classic BEAST).
    beast_temperature  > 0  → SMC-style stochastic resampling: indices drawn
                              with replacement from softmax(scores / T).
                              T → ∞ approaches uniform (Best-of-N-ish).

    Falls back to top-K if all scores are -inf (numerical breakdown).
    """
    n = len(scores)
    if beast_temperature == 0 or n == 0:
        return sorted(range(n), key=lambda i: scores[i])[-num_to_select:]

    # Numerically stable softmax over (scores / T).
    scaled = [s / beast_temperature for s in scores]
    finite = [x for x in scaled if math.isfinite(x)]
    if not finite:
        return sorted(range(n), key=lambda i: scores[i])[-num_to_select:]
    m = max(finite)
    exps = [math.exp(x - m) if math.isfinite(x) else 0.0 for x in scaled]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        return sorted(range(n), key=lambda i: scores[i])[-num_to_select:]
    weights = [e / total for e in exps]
    return random.choices(range(n), weights=weights, k=num_to_select)


# ── Shared search-stage helpers (used by both input_search and output_search) ──

# Cfg keys passed directly into _beast_single_trial_local. Both input_search and
# output_search cfg blocks define all of these (same names, same meanings).
_TRIAL_KWARGS_KEYS: Tuple[str, ...] = (
    "num_beams", "candidates_per_beam",
    "scored_candidate_length", "kept_candidate_length",
    "max_num_iterations", "max_pool_size",
    "temperature", "top_p", "beast_temperature", "eval_beam_chunk_size",
)


def _trial_kwargs(search_cfg) -> Dict[str, Any]:
    """Pluck the trial hyperparams out of a search cfg block as a dict suitable
    for `**`-unpacking into `_beast_single_trial_local`."""
    return {k: getattr(search_cfg, k) for k in _TRIAL_KWARGS_KEYS}


def _get_or_build_latin_mask(
    lm: "LocalModel",
    enabled: bool,
    eos_token_id: Optional[int],
    extra_chars: str,
    cache_tag: str,
    label: str,
) -> Optional[List[int]]:
    """Cached Latin token-id mask for a given (lm, EOS-inclusion, cache_tag) combo.

    Returns None when `enabled` is False. When True, builds (or fetches from
    `_LATIN_MASK_CACHE`) the allowed-token list, including:
      - the EOS token (if eos_token_id is set) so the sampler can terminate
      - any tokens whose decoded form contains chars from `extra_chars` (e.g. "</>"
        so the model can produce </message> closers under input_search's tag flow)

    `cache_tag` discriminates separate caches when the same `lm` is used in
    multiple search modes (e.g. input_search vs output_search may want
    different `extra_chars` even though the underlying tokenizer is the same).
    """
    if not enabled:
        return None
    cache_key = (id(lm), eos_token_id is not None, cache_tag)
    cached = _LATIN_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vocab_size = lm.tokenizer.vocab_size
    extra_ids = [eos_token_id] if eos_token_id is not None else None
    token_ids = build_latin_token_ids(
        lm.tokenizer, vocab_size,
        extra_allowed_ids=extra_ids, extra_allowed_chars=extra_chars,
    )
    _LATIN_MASK_CACHE[cache_key] = token_ids
    extras: List[str] = []
    if eos_token_id is not None: extras.append("EOS")
    if extra_chars:              extras.append(extra_chars)
    note = f" (+{'+'.join(extras)})" if extras else ""
    print(f"    Latin mask {label}{note}: {len(token_ids)}/{vocab_size} tokens allowed", flush=True)
    return token_ids


def _build_sampling_prefix(
    lm: "LocalModel",
    msgs: List[Dict],
    no_think: bool,
    baseline_prefix: str,
) -> Tuple[str, List[int]]:
    """Render chat-template prompt + optional no-think prefix + baseline_prefix,
    then tokenize. Returns (prompt_str, prefix_token_ids). Mirrors exactly how
    `batch_generate_local` builds prompts so searched candidates come from the
    same distribution as the model's natural generation.
    """
    prompt_str = lm.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    if no_think:
        prompt_str += core._NO_THINK_PREFIX
    prompt_str += baseline_prefix
    prefix_tokens = lm.tokenizer.encode(prompt_str, add_special_tokens=False)
    return prompt_str, prefix_tokens


def _beast_single_trial_local(
    lm_sampler: "LocalModel",
    prefix_tokens: List[int],
    scorer_fn: Callable[[List[List[int]], int], List[float]],
    num_beams: int,
    candidates_per_beam: int,
    scored_candidate_length: int,
    kept_candidate_length: int,
    max_num_iterations: int,
    max_pool_size: int,
    temperature: float,
    top_p: float,
    latin_token_ids: Optional[List[int]] = None,
    beast_temperature: float = 0.0,
    eval_beam_chunk_size: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[float]]:
    """One BEAST trial: token-level beam search with optional lookahead.

    Role-agnostic: `lm_sampler` generates candidate token sequences; `scorer_fn` scores
    them with whatever reward signal the caller wires up (input_search → log P(TRS | ...);
    output_search → log P("Yes" | judge prompt + candidate)).

    Per iteration:
      1. Branch:  each beam → `candidates_per_beam` continuations of `scored_candidate_length`
                  tokens (vLLM call with n=candidates_per_beam, max_tokens=scored_candidate_length).
                  `eval_beam_chunk_size=None` issues one batched call across all beams (default,
                  cheap for small n); set to 1 when n is large to avoid OOM after iter-1 beam
                  divergence (vLLM can no longer share KV pages across beams).
      2. Score:   all `num_beams * candidates_per_beam` candidates → scorer_fn(candidates, prefix_length).
      3. Commit:  select `num_beams` candidates and truncate each to `kept_candidate_length`
                  tokens. Selection is hard top-K when `beast_temperature == 0` (classic BEAST),
                  or SMC-style multinomial resampling on softmax(scores / T) with replacement
                  when `beast_temperature > 0`. Setting kept < scored gives lookahead (score
                  with more context, commit fewer tokens).

    Per-iteration token growth = kept_candidate_length.
    Implicit max suffix length = max_num_iterations * kept_candidate_length.

    Sampling is via vLLM SamplingParams (allowed_token_ids enforces the Latin mask).
    Returns (pool_seqs, pool_scores) — token sequences and their log-prob scores.
    """
    if kept_candidate_length > scored_candidate_length:
        raise ValueError(
            f"kept_candidate_length ({kept_candidate_length}) must be <= "
            f"scored_candidate_length ({scored_candidate_length})"
        )
    if num_beams < 1 or candidates_per_beam < 1 or scored_candidate_length < 1 or kept_candidate_length < 1:
        raise ValueError("num_beams, candidates_per_beam, scored/kept_candidate_length must all be >= 1")

    prefix_length = len(prefix_tokens)

    # All beams start as identical copies of the prefix; iter-1 branch diverges them.
    beam: List[List[int]] = [list(prefix_tokens) for _ in range(num_beams)]
    pool_seqs: List[List[int]] = []
    pool_scores: List[float] = []

    for iteration in range(max_num_iterations):
        # ── Phase 1: Branch — sample candidates_per_beam extensions per beam ──
        # chunk bounds peak KV memory once beams diverge (see docstring); None = one batched call.
        chunk = eval_beam_chunk_size or len(beam)
        extensions: List[List[List[int]]] = []
        for start in range(0, len(beam), chunk):
            ext = _vllm_sample_extensions(
                lm_sampler, beam[start:start + chunk], n=candidates_per_beam,
                max_tokens=scored_candidate_length, temperature=temperature, top_p=top_p,
                allowed_token_ids=latin_token_ids,
                ignore_eos=(eos_token_id is not None),
            )
            extensions.extend(ext)

        # Flatten to num_beams * candidates_per_beam full candidates.
        candidates: List[List[int]] = []
        for i, beam_seq in enumerate(beam):
            for ext in extensions[i]:
                candidates.append(beam_seq + ext)

        # ── Phase 2: Score all candidates via the caller-provided scorer ──────
        # scorer_fn supplies the reward signal (see docstring).
        scores = scorer_fn(candidates, prefix_length)

        # ── Phase 3: Select num_beams; truncate to kept_candidate_length ────
        # All beams had the same length L at iteration start; all candidates have
        # length L + scored_candidate_length. Truncate to L + kept_candidate_length.
        # beast_temperature=0 → hard top-K (classic BEAST). >0 → SMC resampling.
        beam_len_at_start = len(beam[0])
        truncate_to = beam_len_at_start + kept_candidate_length
        sel_idx = _select_beam_indices(scores, num_beams, beast_temperature)
        beam = [candidates[i][:truncate_to] for i in sel_idx]
        beam_scores = [scores[i] for i in sel_idx]

        # ── Merge into trial pool ────────────────────────────────────────────
        pool_seqs.extend(beam)
        pool_scores.extend(beam_scores)
        if len(pool_scores) > max_pool_size:
            top_pool_idx = sorted(range(len(pool_scores)), key=lambda i: pool_scores[i])[-max_pool_size:]
            pool_seqs = [pool_seqs[i] for i in top_pool_idx]
            pool_scores = [pool_scores[i] for i in top_pool_idx]

    return pool_seqs, pool_scores


# =============================================================================
def input_search_evaluator_message(
    lm_eval: "LocalModel",
    lm_target: "LocalModel",
    eval_msgs: List[Dict],
    target_msgs: List[Dict],
    search_cfg: Dict,
    no_think_eval: bool,
    sample_max_tokens: int,
    sample_temperature: float,
    batch_size: int = 4,
    lm_jail: Optional["LocalModel"] = None,
    jail_runtime_cfg: Optional[Dict] = None,
) -> Tuple[List[Tuple[str, float, str, str]], str, str]:
    """
    Two-phase adversarial evaluator message search.

    Phase 1 — normal sample: generate an evaluator message to obtain:
      • baseline_msg: the suggested user message (from <message> tags)
      • trs:          targeted response start (reward signal for scoring)
      • strategy:     evaluator's between-turns reasoning (from <strategy> tags)

    Phase 2 — token-level beam/BoN search: sample candidate continuations from
      lm_eval and score each with log P(trs | target_msgs + decoded_candidate)
      via lm_target. The top candidate replaces the Phase 1 baseline.

    Returns:
      pool     — List[(msg_text, score, baseline, suffix)] sorted best-first;
                 baseline + suffix == msg_text; suffix is the searched part
      trs      — the targeted response start string used as the reward signal
      strategy — the evaluator's <strategy> reasoning block (empty string if absent)
    """
    # ── Phase 1: normal sample ──────────────────────────────────────────────
    raw = batch_generate_local(lm_eval, [eval_msgs], sample_max_tokens, sample_temperature,
                                no_think=no_think_eval)[0]
    parsed = parse_message(_make_local_response(raw))
    content = parsed["content"] or raw
    baseline_msg, trs, strategy = _extract_message_tags(content)

    # ── Overwrite TRS with the self-jail response (default reward source) ───
    # jail_in_loss: the BEAST reward target is a SELF-JAIL continuation — the TARGET
    # model itself under the jail system prompt + prefill (carried in jail_runtime_cfg),
    # conditioned on the eval's Phase-1 message (`baseline_msg`, i.e. the input BEFORE
    # the search modifies it). We reuse the already-loaded target model (`lm_target`) as
    # the generator; no separate jail model is loaded. The eval-generated
    # <targeted_response_start> is discarded. Length = max_reward_output_length tokens.
    if (jail_runtime_cfg is not None
            and jail_runtime_cfg.get("jail_in_loss", False)
            and baseline_msg):
        _rlen = search_cfg.max_reward_output_length
        jail_trs = _jail_generate_trs(
            lm_target, jail_runtime_cfg,
            target_msgs + [{"role": "user", "content": baseline_msg}],
            max_tokens=(_rlen if _rlen and _rlen > 0 else 100),
            temperature=sample_temperature,
        )
        if jail_trs:
            trs = jail_trs

    if not trs:
        # No TRS generated — return just the normal message as the single pool entry
        return [(baseline_msg, 0.0, baseline_msg, "")], "", strategy

    # Short-circuit if disabled — return Phase 1 baseline as-is (vanilla bloom).
    if not search_cfg.enabled:
        return [(baseline_msg, 0.0, baseline_msg, "")], trs, strategy

    eos_token_id = _resolve_eos_token_id(lm_eval, search_cfg.truncate_at_eos)

    # ── Compute baseline_prefix: raw Phase 1 output up to a cursor position ──
    # The prefix is the RAW Phase 1 content (with <strategy>/<message> tags +
    # preamble) so the search sees full context. Cursor anchors at </message>;
    # `max_prefix_length` controls how much of the message BODY is kept before it:
    #   None → cursor right before </message>  (keep full body, classic suffix attack)
    #   0    → cursor right after <message>    (no body — sampler regenerates whole body)
    #   N>0  → cursor after first N tokens of body
    #   N<0  → cursor |N| tokens before end of body
    # Downstream extraction (_extract_message_tags) strips the surrounding tags
    # so the target only receives the clean message body.
    mpl = search_cfg.max_prefix_length
    msg_open  = re.search(r"<message>", content)
    msg_close = re.search(r"</message>", content)
    if msg_open and msg_close and msg_open.end() <= msg_close.start():
        before_msg_text = content[:msg_open.end()]  # includes <message> opener
        body_text       = content[msg_open.end():msg_close.start()]
        if mpl is None:
            body_prefix = body_text
        elif mpl == 0:
            body_prefix = ""
        else:
            body_ids = lm_eval.tokenizer.encode(body_text, add_special_tokens=False)
            body_prefix = lm_eval.tokenizer.decode(body_ids[:mpl], skip_special_tokens=True)
        baseline_prefix = before_msg_text + body_prefix
    else:
        # Phase 1 output malformed — slice the extracted baseline_msg directly.
        if mpl is None:
            baseline_prefix = baseline_msg
        elif mpl == 0:
            baseline_prefix = ""
        else:
            bm_ids = lm_eval.tokenizer.encode(baseline_msg, add_special_tokens=False)
            baseline_prefix = lm_eval.tokenizer.decode(bm_ids[:mpl], skip_special_tokens=True)

    # ── Truncate TRS to first max_reward_output_length target-model tokens ──
    if search_cfg.max_reward_output_length > 0 and trs:
        trs_ids = lm_target.tokenizer.encode(trs, add_special_tokens=False)
        trs = lm_target.tokenizer.decode(
            trs_ids[:search_cfg.max_reward_output_length], skip_special_tokens=False,
        )

    # ── Build sampling prefix + Latin mask via shared helpers ─────────────
    _, prefix_tokens = _build_sampling_prefix(lm_eval, eval_msgs, no_think_eval, baseline_prefix)
    prefix_length = len(prefix_tokens)
    # Allow `<`, `/`, `>` chars so the sampler can naturally emit </message> when
    # truncate_at_eos is on; otherwise stay strict.
    latin_token_ids = _get_or_build_latin_mask(
        lm_eval, search_cfg.latin_mask, eos_token_id,
        extra_chars="</>" if eos_token_id is not None else "",
        cache_tag="input_search", label="",
    )

    # ── Bind input-search scorer (closure captures lm_target + scoring ctx) ──
    def _scorer(candidates: List[List[int]], pfx_len: int) -> List[float]:
        return _score_beast_candidates(
            lm_eval, lm_target, candidates, pfx_len,
            target_msgs, trs, baseline_prefix, batch_size,
            eos_token_id=eos_token_id,
        )

    print(f"    input search {dict(_trial_kwargs(search_cfg))} ...", flush=True)
    global_pool_seqs, global_pool_scores = _beast_single_trial_local(
        lm_sampler=lm_eval,
        prefix_tokens=prefix_tokens,
        scorer_fn=_scorer,
        latin_token_ids=latin_token_ids,
        eos_token_id=eos_token_id,
        **_trial_kwargs(search_cfg),
    )

    # ── Decode pool into (msg_text, score, baseline, suffix) tuples, best first ──
    # msg_text     = the clean message body delivered to the target (extracted)
    # baseline     = the portion of msg_text carried over from Phase 1's <message> body
    # suffix       = the portion of msg_text generated by BoN (used for viewer highlight)
    order = sorted(range(len(global_pool_scores)),
                   key=lambda i: global_pool_scores[i], reverse=True)
    pool: List[Tuple[str, float, str, str]] = []
    seen: set = set()
    for i in order:
        seq         = global_pool_seqs[i]
        score       = global_pool_scores[i]
        suffix_ids  = _strip_eos_tail(seq[prefix_length:], eos_token_id)
        suffix_text = lm_eval.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        full_text   = baseline_prefix + suffix_text
        # _extract_message_tags → clean body (see helper); falls back to full_text.
        extracted_msg, _, _ = _extract_message_tags(full_text)
        msg_text    = extracted_msg if extracted_msg else full_text
        # Viewer baseline/suffix highlight: derive the carried-over body using the
        # same _extract_message_tags pipeline that produced msg_text, so the two
        # are directly comparable (identical whitespace handling, identical tag
        # parsing). This is more robust than ad-hoc .strip() on body_prefix.
        carried_over, _, _ = _extract_message_tags(baseline_prefix) if baseline_prefix else ("", "", "")
        if carried_over and msg_text.startswith(carried_over):
            baseline_view = carried_over
            suffix_view   = msg_text[len(carried_over):]
        else:
            baseline_view = ""
            suffix_view   = msg_text
        if msg_text and msg_text not in seen:
            seen.add(msg_text)
            pool.append((msg_text, score, baseline_view, suffix_view))

    if not pool:
        pool = [(baseline_msg, 0.0, baseline_msg, "")]  # fallback to Phase 1 baseline

    return pool, trs, strategy


def output_search_target_response(
    lm_eval: "LocalModel",
    lm_target: "LocalModel",
    target_msgs: List[Dict],
    user_input: str,
    baseline_response: str,
    output_cfg: Dict,
    no_think_target: bool,
    no_think_judge: bool,
    judge_prompt_template: str,
    behavior_name: str,
    batch_size: int = 4,
    lm_jail: Optional["LocalModel"] = None,
    jail_runtime_cfg: Optional[Dict] = None,
) -> List[Tuple[str, float, str, str]]:
    """
    Search over the TARGET's response tokens, scored by lm_eval's log P("Yes")
    on a judge prompt asking whether the response exhibits the named behavior.

    Mirror of `input_search_evaluator_message` with roles swapped:
      • lm_target generates candidates (sampler)
      • lm_eval scores them (judge)

    No Phase 1 sampling here — `baseline_response` is the target's natural reply
    already produced upstream in the rollout loop; it serves as the Phase 1
    baseline that `max_prefix_length` slices into.

    Returns pool sorted best-first as
      List[(response_text, score, baseline, suffix)]
    where baseline + suffix == response_text. `baseline` is the slice of the
    natural response that was carried over (the body_prefix); `suffix` is the
    searched continuation.
    """
    # ── Short-circuit if output search is disabled ─────────────────────────
    if not output_cfg.enabled:
        return [(baseline_response, 0.0, baseline_response, "")]

    eos_token_id = _resolve_eos_token_id(lm_target, output_cfg.truncate_at_eos)

    # ── Compute baseline_prefix: slice of natural target response ──────────
    # Target response has no XML scaffolding (unlike input_search's <message> tags),
    # so baseline_prefix is just a token-level slice of baseline_response.
    #   None → keep full response  (cursor at end, classic suffix attack)
    #   0    → keep nothing        (cursor at start, BoN regenerates whole response)
    #   N>0  → first N tokens
    #   N<0  → drop last |N| tokens
    mpl = output_cfg.max_prefix_length
    if not baseline_response or mpl == 0:
        baseline_prefix = ""
    elif mpl is None:
        baseline_prefix = baseline_response
    else:
        body_ids = lm_target.tokenizer.encode(baseline_response, add_special_tokens=False)
        kept_ids = body_ids[:mpl] if mpl > 0 else body_ids[:max(0, len(body_ids) + mpl)]
        baseline_prefix = lm_target.tokenizer.decode(kept_ids, skip_special_tokens=True)

    # ── Build sampling prefix + Latin mask via shared helpers ─────────────
    _, prefix_tokens = _build_sampling_prefix(lm_target, target_msgs, no_think_target, baseline_prefix)
    prefix_length = len(prefix_tokens)
    # Target output has no <message> tags, so no `</>" extras in the mask.
    latin_token_ids = _get_or_build_latin_mask(
        lm_target, output_cfg.latin_mask, eos_token_id,
        extra_chars="", cache_tag="output_search", label="(output)",
    )

    # ── Bind judge scorer (lm_eval scores, lm_target decodes candidate text) ──
    def _scorer(candidates: List[List[int]], pfx_len: int) -> List[float]:
        return _score_output_candidates(
            lm_eval=lm_eval,
            candidates=candidates,
            prefix_length=pfx_len,
            target_tokenizer=lm_target.tokenizer,
            judge_prompt_template=judge_prompt_template,
            behavior_name=behavior_name,
            user_input=user_input,
            baseline_prefix=baseline_prefix,
            no_think_judge=no_think_judge,
            max_batch_size=batch_size,
            eos_token_id=eos_token_id,
        )

    use_contrastive_sampling = (
        lm_jail is not None and jail_runtime_cfg is not None
        and jail_runtime_cfg.get("enabled", False)
    )
    use_jail_scoring = (
        lm_jail is not None and jail_runtime_cfg is not None
        and jail_runtime_cfg.get("jail_out_loss", False)
    )

    # Build jail's prefix once if we need it (sampling and/or scoring).
    j_prefix: Optional[List[int]] = None
    if lm_jail is not None and jail_runtime_cfg is not None and (use_contrastive_sampling or use_jail_scoring):
        sys_prompt = jail_runtime_cfg.get("system_prompt", "")
        prefill    = jail_runtime_cfg.get("prefill", "") or ""
        j_msgs = [m for m in target_msgs if m.get("role") != "system"]
        if sys_prompt:
            j_msgs = [{"role": "system", "content": sys_prompt}] + j_msgs
        j_prompt = lm_jail.tokenizer.apply_chat_template(
            j_msgs, tokenize=False, add_generation_prompt=True,
        )
        # Close Qwen3's auto-opened <think> block before prefill so the next-
        # token distribution isn't dominated by </think>.
        j_prompt += core._CORRUPT_NO_THINK_PREFIX
        if prefill:
            j_prompt += prefill
        if baseline_prefix:
            j_prompt += baseline_prefix
        j_prefix = lm_jail.tokenizer.encode(j_prompt, add_special_tokens=False)

    # ── Sampling ──────────────────────────────────────────────────────────
    if use_contrastive_sampling:
        beta  = float(jail_runtime_cfg.get("b2", 2.0))
        top_k = int(jail_runtime_cfg.get("top_k_logprobs") or 1000)
        n       = int(output_cfg.candidates_per_beam)
        max_tok = int(output_cfg.scored_candidate_length)
        temp    = float(output_cfg.get("temperature", 1.0))
        top_p   = float(output_cfg.get("top_p", 1.0))
        print(
            f"    output search [contrastive PoE: β={beta}, K={top_k}, n={n}, "
            f"len={max_tok}, score={'jail' if use_jail_scoring else 'judge'}] "
            f"(behavior={behavior_name!r}) ...", flush=True,
        )
        extensions = _contrastive_sample_extensions(
            lm_target=lm_target, lm_jail=lm_jail,
            target_prefixes=[prefix_tokens], jail_prefixes=[j_prefix],
            n=n, max_tokens=max_tok,
            beta=beta, top_k_logprobs=top_k,
            temperature=temp, top_p=top_p,
            allowed_token_ids=latin_token_ids,
            ignore_eos=(eos_token_id is not None),
            eos_token_id=eos_token_id,
        )[0]
        candidates = [prefix_tokens + ext for ext in extensions]
        global_pool_seqs = candidates
    else:
        print(
            f"    output search {dict(_trial_kwargs(output_cfg))} "
            f"(behavior={behavior_name!r}, score={'jail' if use_jail_scoring else 'judge'}) ...",
            flush=True,
        )
        global_pool_seqs, _ = _beast_single_trial_local(
            lm_sampler=lm_target,
            prefix_tokens=prefix_tokens,
            scorer_fn=_scorer,
            latin_token_ids=latin_token_ids,
            eos_token_id=eos_token_id,
            **_trial_kwargs(output_cfg),
        )

    # ── Scoring ───────────────────────────────────────────────────────────
    if use_jail_scoring:
        # Score each candidate as log p_jail(ext | jail_prefix) via teacher forcing.
        items: List[Tuple[List[int], int, int]] = []
        for seq in global_pool_seqs:
            ext = seq[prefix_length:]
            if not ext:
                items.append(([], 0, 0))
            else:
                items.append((j_prefix + ext, len(j_prefix), len(ext)))
        raw = lm_jail.worker.compute_target_logprobs(items)
        global_pool_scores = [s if s is not None else -float("inf") for s in raw]
    else:
        global_pool_scores = _scorer(global_pool_seqs, prefix_length)

    # ── Decode pool into (response_text, score, baseline, suffix), best first ──
    order = sorted(range(len(global_pool_scores)),
                   key=lambda i: global_pool_scores[i], reverse=True)
    pool: List[Tuple[str, float, str, str]] = []
    seen: set = set()
    for i in order:
        seq         = global_pool_seqs[i]
        score       = global_pool_scores[i]
        suffix_ids  = _strip_eos_tail(seq[prefix_length:], eos_token_id)
        suffix_text = lm_target.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        full_text   = baseline_prefix + suffix_text
        # No XML extraction for target responses — content IS the response body.
        if full_text and full_text not in seen:
            seen.add(full_text)
            pool.append((full_text, score, baseline_prefix, suffix_text))

    if not pool:
        # All candidates degenerated; fall back to the natural target response.
        pool = [(baseline_response, 0.0, baseline_response, "")]

    return pool

__all__ = ['build_latin_token_ids', '_score_beast_candidates', '_score_output_candidates', '_select_beam_indices', '_TRIAL_KWARGS_KEYS', '_trial_kwargs', '_get_or_build_latin_mask', '_build_sampling_prefix', '_beast_single_trial_local', 'input_search_evaluator_message', 'output_search_target_response']
