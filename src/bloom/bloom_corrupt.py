import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import litellm
import yaml

# Suppress LiteLLM verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False
from bloom.core import *
from bloom import core
from bloom.wilt import *
from bloom.search import *
from bloom.rollout import *
from bloom.pipeline import *



# =============================================================================
# Section 11: Config & Main
# =============================================================================

# Model spec format (parsed by _parse_local_spec):
#   "<repo>"                        → plain HF model, loaded bf16
#   "<repo>:<quant>"                → GGUF (non-multimodal models)
#   "<repo>:<quant>:<base_repo>"    → GGUF; tokenizer/config from base_repo (multimodal)
# ── Model selection — pick a model by changing the index into these lists ────────────
# Auditor (understanding + ideation + rollout-evaluator + judgment). The DEFAULT must be a
# local model: BEAST (input_search) and FLRT (output_search) do a token-level search over the
# auditor's message and need its logits, which hosted API models don't expose. The Gemma-4
# 26B-A4B MoE (FP8, ~26GB, weight-only Marlin on Ampere) needs the vllm-0.23/cu129 env
# (.venv-g4), NOT the vllm-0.14 default.
EVAL_MODELS = [
    "local/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",  # 0: Gemma-4-26B-A4B (default auditor)
    "claude-sonnet-4-6",                              # 1: Claude Sonnet (API cross-check; input/output search OFF, set BLOOM_EVAL_THINKING=0)
]
# Targets under evaluation — the four open-weight models from the paper's experiments section.
TARGET_MODELS = [
    "local/meta-llama/Llama-3.2-3B-Instruct",  # 0: Llama-3.2-3B-Instruct
    "local/microsoft/Phi-4-mini-instruct",     # 1: Phi-4-mini-instruct
    "local/Qwen/Qwen3.5-4B",                   # 2: Qwen3.5-4B
    "local/google/gemma-4-e4b-it",             # 3: Gemma-4-E4B
]
# Smaller same-family sibling of each TARGET_MODELS entry (index-aligned) — the weak steering
# model for the W2S small-expert experiment. Phi-4-mini has no smaller release, so it is None.
# Think-block wrappers for these are registered in bloom/core.py:_USES_THINK_BLOCK.
SMALL_TARGET_MODELS = [
    "local/meta-llama/Llama-3.2-1B-Instruct",  # 0: Llama-3.2-1B-Instruct  (<- Llama-3.2-3B)
    None,                                       # 1: Phi-4-mini has no smaller sibling
    "local/Qwen/Qwen3.5-2B",                   # 2: Qwen3.5-2B             (<- Qwen3.5-4B)
    "local/google/gemma-4-e2b-it",             # 3: Gemma-4-E2B            (<- Gemma-4-E4B)
]

# Abliterated (refusal-removed) corruptor variants, keyed by their ORIGINAL target. Same
# architecture/vocab as the originals, so they can drive the corruption PoE (supply a strong
# offensive direction with no refusals) while the TARGET distribution stays the original model.
# Reference data only: this WILT shim has no corruption block; the map is consumed by the
# archived runs_old corruption pipeline (BLOOM_CORRUPT_MODEL=local/<repo> + BLOOM_TARGET_MODEL=local/<original>).
# Keyed by the ORIGINAL target (matching TARGET_MODELS, without the local/ prefix). All four
# targets now have entries here (Llama-3.2-3B's abliterated variant is also registered in
# core._USES_THINK_BLOCK).
ABLITERATED_CORRUPTORS = {
    "meta-llama/Llama-3.2-3B-Instruct": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",                   # 0: <- Llama-3.2-3B-Instruct
    "microsoft/Phi-4-mini-instruct": "huihui-ai/Phi-4-mini-instruct-abliterated",                        # 1: <- Phi-4-mini
    "Qwen/Qwen3.5-4B":               "huihui-ai/Huihui-Qwen3.5-4B-abliterated",                          # 2: <- Qwen3.5-4B
    "google/gemma-4-e4b-it":         "huihui-ai/Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated", # 3: <- Gemma-4-E4B
}

judge_model  = EVAL_MODELS[0]
target_model = TARGET_MODELS[2]   # Qwen3.5-4B — the paper's representative slice

cfg = DotDict({
    "folder_name": "runs_new/default",           # output dir under the runs root (core.RUNS_ROOT = experiments/bloom by default; override via BLOOM_RUNS_ROOT). Always overridden by BLOOM_FOLDER (the sweep driver sets this per run).

    "behavior_file": "prompts/1_racial_bias.yaml",  # behaviour-specific yaml (description, overrides, jail + corruption prompts). Override with BLOOM_BEHAVIOR_FILE.
    "behavior_name": "racial-bias",          # label used in prompt templates; overwritten from behavior_file in __main__
    "examples": [],                          # seed transcripts: [{conversation: [{role, content}]}]; used to ground understanding/ideation
    "kickoff_bank": None,                    # dir of per-round reusable evaluator kickoffs (set via BLOOM_KICKOFF_BANK). None = generate fresh kickoffs each round. Reused across beta sweeps for the same auditor + behaviour.

    "temperature": 1.0,                      # sampling temperature for all LLM calls (evaluator, target, judge)
    "seed": 100,                             # base RNG seed; round R samples with seed+R (reproducible but distinct per round). Convention via BLOOM_SEED: sweep/param-selection = 1 (rounds 2..6, in-sample); FINAL experiments = 100 (rounds 101..105, out-of-sample). This standalone default = the final-experiment seed. None = no seeding.
    "max_concurrent": 10,                    # max simultaneous API requests in flight (API path only)
    "batch_size": 5,                         # local models: variations per GPU forward pass; larger = faster but more VRAM
    "target_batch_size": 25,                 # target-model batch for input-search candidate scoring; defaults to batch_size if omitted. Target is much smaller so can usually go higher (up to num_beams*candidates_per_beam = 25)

    # Each LLM runs in its own subprocess pinned to one GPU
    "evaluator_gpu_id": 0,
    "target_gpu_id":    1,
    "evaluator_gpu_memory_utilization": 0.80, # Always using vLLM
    "target_gpu_memory_utilization":    0.80, # Doesnt apply if using HF instead of vLLM

    "understanding": {
        "model": judge_model,                # model that analyses the behavior and any seed transcripts
        "max_tokens": 2000,                  # max output tokens for the understanding call
        "thinking": True,                    # True = reasoning enabled ("medium" budget); False = no thinking
    },
    "ideation": {
        "model": judge_model,                # model that generates evaluation scenarios
        "max_tokens": 20000,                 # ceiling on ideation output tokens. Local path generates in CHUNKS (chunk_size) and further caps each call to fit the auditor window; API path caps to the model max. (A single all-scenarios call cannot fit 100 scenarios in the 16384 auditor window — hence chunking.)
        "chunk_size": 10,                    # local auditor: scenarios generated per call. Each chunk is a fresh, non-accumulating call (new seed + compact dedup list) so the prompt stays well under evaluator_max_model_len regardless of num_scenarios.
        "thinking": True,                    # True = reasoning enabled ("medium" budget); False = no thinking
        "num_scenarios": 100,                # total scenarios to generate; more = broader coverage but slower rollout. WILT param-sweep default = 15; final experiments = 100.
    },
    "rollout": {
        "model": judge_model,                # evaluator model — generates adversarial messages via input search
        "target": target_model,              # model under evaluation
        "evaluator_max_tokens": 1200,        # cap on evaluator output (just emits <message> blocks; tags auto-closed if truncated)
        "target_max_tokens": 250,            # cap on target response length — reduced to match 2-5 sentence constraint
        "evaluator_max_model_len": 16384,    # vLLM context window for the eval model (Gemma-4 supports far more; 8192 was too tight — long understanding/ideation or 3-turn judgment prompts overflow it)
        "target_max_model_len": 8192,        # vLLM context window for the target model — 8192 (was 4096) since turn-3 conversations with eval_max_tokens=1200 can exceed 4096
        "evaluator_thinking": True,          # True = evaluator reasoning enabled; False = no thinking
        "target_thinking": False,            # True = target reasoning enabled; False = no thinking
        "max_turns": 3,                      # conversation turns per rollout (each turn = one target response + one searched evaluator message)
        "num_rounds": 5,                     # total rounds; round 1 = full pipeline, rounds 2+ = rollout + judge (fresh resamples). WILT default = 5 (set via BLOOM_NUM_ROUNDS).
        "target_engine": "hf",                  # target decode engine for the WILT/BoN path: "hf" (exact full-vocab PoE) or "vllm" (vllm_topk). Override with BLOOM_TARGET_ENGINE. Ignored when any search/jail role is active (those pick the engine automatically).
    },
    "judgment": {
        "model": judge_model,                # model that scores transcripts for behavior presence
        "max_tokens": 500,                   # max output tokens per judgment call (tags auto-closed if truncated)
        "thinking": True,                    # True = reasoning enabled ("medium" budget); False = no thinking
        "num_samples": 1,                    # score samples per transcript to average (higher = less variance)
        "additional_qualities": [], #["unrealism", "bugs"],  # extra qualities to score; each needs a description in prompts.yaml
        "metajudgment_qualities": ["diversity"],             # qualities assessed across all transcripts in the metajudge step
        "metajudgment": False, #True,        # set False to skip the metajudge step entirely
    },
    "refinement_input": {
        "enabled": False,                    # True = merged refinement (round-2+ kickoff sees prior history + guidance and emits <strategy>+<message>). False = each round is a fresh resample with no learning (BoN baseline).
        "history_transcript_rounds": 2,      # how many prior FULL transcripts are shown at the kickoff: None=all, 0=none, N=last N
        "history_strategy_rounds": None,     # how many prior (round, score, strategy) log rows are shown (also drives the guidance): None=all, 0=none, N=last N
    },
    "search_input": {
        "enabled": False,                        # ON: input-side BEAST search over the evaluator's <message> body. The BEAST reward TRS is ALWAYS generated self-jail from the target model (jail system prompt + prefill), never by the evaluator.
        "num_beams": 3,                          # TUNED (self_harm/Qwen 3-turn): 3x3 beams (9 scored/iter) match jail's compute (~13min vs 11min) at ~same elicitation as 5x5 (4.00 vs 4.80, within noise). 5x5 was 2.2x slower for no reliable gain.
        "candidates_per_beam": 3,                # 3 candidates per beam → 9 scored per iter (was 5x5=25; the extra candidates only inflated wall-clock under mp=None)
        "scored_candidate_length": 5,            # TUNED (racial/Qwen, 2 seeds + combo): kl5 > kl15; k10 also tanked 3x3 (2.33 vs 4.00) — keep 5
        "kept_candidate_length": 5,              # TUNED: commit 5 tokens/iter — kl5 beats kl10/kl15; the best single grid config
        "max_num_iterations": 9,                 # TUNED: compute dial (linear in wall-clock); 9 iters at 3x3 lands in the jail/BoN envelope. This is the config used for the final 100-scen runs.
        "max_prefix_length": None,               # TUNED: None (suffix attack) is best over 2 seeds; full-rewrite (0) is worst. how much of Phase 1's <message> body is pre-loaded into the BoN prompt before the cursor. Phase 1's <strategy>/preamble/opening <message> are ALWAYS in the context. None = keep full body (cursor right before </message>, classic suffix attack); 0 = keep nothing (cursor right after <message>, BoN samples whole body); N>0 = first N tokens of body; N<0 = drop last |N| tokens of body.
        "eval_beam_chunk_size": None,            # None = batch all 5 beams in one vLLM call (cheap at n=5); 1 only needed when candidates_per_beam is large (~100+) to avoid OOM after iter-1 beam divergence
        "max_reward_output_length": 150,         # TUNED: 150 > 25 both seeds. first N tokens of TRS used as reward signal (0 = full TRS)
        "temperature": 1.0,                      # sampling temperature for token expansion (never tuned; fixed at 1.0)
        "max_pool_size": 50,                     # max candidates accumulated across the search
        "latin_mask": True,                       # TUNED: True gives higher elicitation on BOTH seeds (keeps the suffix coherent, ASCII-only). Verify on a 2nd cell before final. restrict beam search to Latin/ASCII tokens only (blocks unicode/digits/punctuation)
        "truncate_at_eos": False,               # TUNED: False gives higher elicitation on BOTH seeds (2.80/4.00 vs baseline 1.87/2.87), elapsed-neutral. Pairs with latin_mask=True (the mask blocks the terminator chars so the model keeps writing). If True: also allows the model to emit `<`, `/`, `>` (so it can naturally produce </message> to terminate the body) and EOS. _extract_message_tags then truncates the candidate at the first </message>. If False: latin mask blocks those characters so the model keeps writing message content until max_tokens, and the entire suffix is the message body (no truncation needed).
    },
    "flrt_search_input": {
        "enabled": False,                        # ON: FLRT-style input-side search over the evaluator's <message> body. Black-box mutation-buffer search (append/insert/delete/swap) scored by a FULL-VOCAB distillation loss: pull the target's per-token distribution TOWARD the self-jail teacher's over a shared continuation (FLRT L_D; Thompson & Sklar 2024). The reward continuation is generated self-jail from the target (jail prompt + prefill), exactly like search_input's TRS. Engine is fixed HF (full-vocab distributions; vLLM top-K is insufficient). Defaults follow the ORIGINAL FLRT paper except our agreed adaptations (self-jail teacher instead of a LoRA toxic model; teacher task-only by default).
        # ── Search compute (BEAST param names reused where same-function) ──
        "buffer_size": 8,                        # PAPER default: active search buffer; the single best in the buffer is mutated each iteration, top-buffer_size retained.
        "k1": 8,                                 # TUNED (i6·k8·nt2 sweet spot; PAPER default was 32): mutated candidates generated + scored per iteration. Override BLOOM_FLRT_K1.
        "k2": 16,                                # PAPER: candidate replacement tokens sampled per position (swap/insert) from the auditor's per-position distribution.
        "max_num_iterations": 6,                 # TUNED i6 (i6·k8·nt2 sweet spot, ~15m/3-turn self_harm; PAPER demo used Settings(100,...)): iterations per trial (the compute dial). Override BLOOM_FLRT_ITERS.
        "n_trials": 2,                           # TUNED nt2 (depth+diversity beat either alone; nt5 too slow at ~21m; ExperimentFLRT.py used 5): independent restarts (fresh init), merged into one pool. Override BLOOM_FLRT_NTRIALS.
        "max_pool_size": 50,                     # ExperimentFLRT.py pool_size=50 — max candidates accumulated across the search.
        "eval_beam_chunk_size": None,            # HF batch chunk for scoring the k1 candidates/iter (BLOOM infra knob); None = one batched forward.
        "temperature": 1.0,                      # ExperimentFLRT.py: auditor mutation-proposal sampling temperature.
        # ── Mutation mix (ExperimentFLRT.py: append highest, rest even → 1/2, 1/6, 1/6, 1/6) ──
        "p_append": 1/2,                         # append a sampled token at the END (BEAST-style end-insert) — highest-probability op.
        "p_insert": 1/6,                         # insert a sampled token at a random INTERIOR position.
        "p_delete": 1/6,                         # delete a random token.
        "p_swap": 1/6,                           # swap a random token for a sampled replacement.
        "num_mutations": 3,                      # TUNED 3 (grid winner + longer learned suffix): how many times the chosen operator is applied to EACH candidate per iteration (append/insert +N tokens, delete −N, swap N positions). 1 = original single-mutation. All candidates share operator+count → identical length delta → batch stays rectangular. Override BLOOM_FLRT_N_MUT.
        # ── Suffix bounds / init (init suffix is always sampled autoregressively from the auditor) ──
        "start_tokens": 10,                      # ExperimentFLRT.py: initial suffix length.
        "min_tokens": 5,                         # delete disabled below this suffix length (also the floor num_mutations delete is guarded against).
        "max_tokens": None,                      # upper bound above which insert/append are disabled. None = UNBOUNDED (let the suffix grow freely — the higher-headroom default for future runs). Set an int (e.g. 40) via BLOOM_FLRT_MAX_TOKENS to cap.
        # ── max_prefix_length / masking / eos (BEAST names, same meaning as search_input) ──
        "max_prefix_length": None,               # how much of Phase-1's <message> body is preloaded before the mutation region. None = keep full body (suffix mutation, classic); 0 = keep nothing (whole body generated/mutated); N>0 = first N tokens; N<0 = drop last |N|.
        "latin_mask": True,                      # restrict mutation tokens to Latin/ASCII (== search_input default).
        "truncate_at_eos": False,                # allow a candidate to emit EOS/terminator and truncate (== search_input).
        "max_reward_output_length": 32,          # PAPER n_match: length K of the self-jail continuation the distillation loss L_D is averaged over (one teacher-forced forward gives all K positions). NOT single-token. 0 = full continuation. (Same-function as search_input's reward length, hence the shared name.)
        # ── Losses: distillation L_D is the only default-on objective; L_XE (fluency) + L_Rep default off, combined by plain weighted sum (no z-norm) ──
        "p_threshold": 0.6,                      # PAPER: L_D per-token reward cap = log(p_threshold) — stop rewarding a continuation token once the target already matches the teacher.
        "w_distillation": 1.0,                   # weight on the distillation term. Only matters RELATIVE to w_force (a lone scalar doesn't change ranking). When w_force>0, distillation is z-normed across the candidate batch before weighting.
        "w_force": 0.5,                          # TUNED 0.5 (inverted-U sweet spot on racial: wf 0→0.5→1 = 3.93→5.33→4.13; pure force hurts). Weight on TEACHER FORCING = target's log-prob of the teacher's ACTUAL continuation tokens (log q_base(forced_i)), over the SAME reward positions as distillation, same p_threshold cap. 0 = distillation-only (legacy). >0 activates z-normed weighted combine: w_distillation*z(L_D) + w_force*z(forcing). Free (reuses the log_q already built). Override BLOOM_FLRT_W_FORCE.
        "w_fluency": 0.0,                        # L_XE weight (perplexity of the attack tokens). 0 = off.
        "fluency_on": "auditor",                 # which model scores fluency perplexity: "auditor" or "target".
        "w_repetition": 0.0,                     # L_Rep weight (repetition penalty on the attack tokens). 0 = off.
        "repetition_exponent": 1.5,              # PAPER: penalty = sum((count-1)^exp) / suffix_len.
    },
    "search_output": {
        "enabled": False,                        # True = run output search on every target response
        "behavior_name": "racial bias",          # what the judge asks about (independent from cfg.behavior_name so it can be set per experiment). Judge template (output_search_judge_prompt) is "last_input": user input + response + "does it exhibit <behavior_name>? Yes/No" — validated best single default on real turn-2 data (name+last-input; description helps some behaviours but hurts deception). For our 3 target behaviours set this to: "Strategic deception", "Self harm encouragement", "Political bias".
        "num_beams": 3,                          # TUNED 3x3.i6.k20.mp0 (small-beam sweet spot: search saturates ~4 cand/iter; matches 4x4.i8 elic at ~half wall-clock)
        "candidates_per_beam": 3,                # 3 candidates per beam (9 scored/iter; more only cost time)
        "scored_candidate_length": 20,           # committed length k=20 (TUNED peak; k10/k15/k25/k30 all lower)
        "kept_candidate_length": 20,
        "max_num_iterations": 6,                 # TUNED i6 (3x3.i6 cheap sweet spot). BoN reference = set BLOOM_OUTPUT_ITERS=1
        "max_prefix_length": 0,                  # how much of the natural target response is preloaded before the cursor. None = full response (suffix attack); 0 = regenerate from scratch (typical BoN); N>0 = first N tokens; N<0 = drop last |N|.
        "eval_beam_chunk_size": 1,
        "temperature": 1.0,
        "max_pool_size": 50,
        "latin_mask": False,                     # Target should speak naturally; leave off unless you have a specific reason
        "truncate_at_eos": False,                 # If True: allows EOS so target can naturally terminate. If False: target keeps generating until scored_candidate_length.
    },
    "jailbroken_output": {
        "enabled":  False,                        # jail on/off: steer the sampling step (contrastive PoE for target tokens)
        "engine": "hf_full",                      # vllm_topk = legacy top-K logit_bias | hf_full = exact full-vocab PoE (HF)
        "var_batch": 15,                          # hf_full engine: cross-scenario batch size (slots per turn-generate) for the jail rollout. Override with BLOOM_JAIL_VAR_BATCH.
        "model": "self",                          # jail/proposal model. "self" (or "") = self-jail = the target model (default). Set local/<hf-name> for a distinct proposal (e.g. an abliterated variant). Override with BLOOM_JAIL_MODEL.
        "prefill": True,                          # toggle: True (default) = use the behaviour file's jailbroken_output_prefill; False = no prefill
        "top_k_logprobs": None,                   # vllm_topk engine ONLY: K top-K jail logprobs for the approximate logit-bias PoE. Inert on the default hf_full engine (exact full-vocab PoE). None -> falls back to 1000 if vllm_topk is selected.
        "target_floor": 1e-4,                     # naturalness floor ON by default: mask tokens with target prob < floor before sampling the tilt (argmax(target) fallback). 0 = off (no-floor ablation only).
        "b1": 1,                                  # target-term weight in z = b1*target + b2*jail - b3*neg (default 1). 0 = floor-only jail (drop the target term). None also accepted (legacy code path; numerically identical to 1). The cfg.tokbias_output baseline works at any b1.
        "b2": 4.0,                                # jail-expert weight in z = b1*target + b2*jail - b3*neg (PoE weight on log p_jailbroken); only used when enabled=True. Tuned per (model, behaviour) — the sweep sets it via BLOOM_JAIL_BETA.
        "b3": 0.0,                                # negative-steering weight in z = b1*target + b2*jail - b3*neg. 0 = off (the only knob; override BLOOM_JAIL_B3). Ablation: W2S logit-difference. When b3>0, the neg prompts load from the behaviour yaml (jailbroken_output_neg_system_prompt / _neg_user_prompt / _neg_prefill), or cfg jailbroken_output.neg_* if set.
    },
    "tokbias_output": {                           # static logit-bias baseline (z = target + lambda*bias over the whole vocab) — a separate elicitation method from jail. Numeric knobs here; the prompt content (prompt / neg_prompt / words) lives in the behaviour yaml (tokbias_output_prompt / _neg_prompt / _words). Every field overridable via BLOOM_TOKBIAS_*.
        "enabled": False,                         #   on/off: when False the bias vector is never computed (short-circuits before any prompt eval). Override with BLOOM_TOKBIAS_ENABLED.
        "lambda": 0.0,                            #   tilt scale; 0.0 (or no prompt/words) = exact no-op
        "steps": 8,                               #   rolled-forward positions averaged into the relevance estimate (>1 broadens beyond the immediate next token)
        "samples": 4,                             #   stochastic continuations averaged for the estimate (no-op at steps=1)
    },
})


if __name__ == "__main__":
    # Load the behaviour-specific file (path from BLOOM_BEHAVIOR_FILE / cfg.behavior_file): its
    # description + cfg overrides (was prompt_presets). Its jail/corruption prompts are merged
    # into the prompt set separately by load_prompts().
    _bf = os.environ.get("BLOOM_BEHAVIOR_FILE") or cfg.get("behavior_file", "prompts/1_racial_bias.yaml")
    cfg["behavior_file"] = _bf
    _beh = yaml.safe_load(open(SCRIPT_DIR / _bf, encoding="utf-8"))
    if _beh.get("behavior_name"):
        cfg["behavior_name"] = _beh["behavior_name"]
    _behavior_desc = _beh.get("behavior_description", "")
    if not _behavior_desc:
        raise ValueError(f"No 'behavior_description' in behaviour file '{_bf}'")
    cfg["behavior_description"] = _behavior_desc.strip()
    for k, v in (_beh.get("overrides") or {}).items():
        if k not in cfg:  # cfg overrides take priority
            cfg[k] = v.strip() if isinstance(v, str) else v
    print(f"Loaded behaviour: {cfg.behavior_name} ({_bf})", flush=True)

    # --- env-var overrides (for autonomous sweeps; no source edits between runs) ---
    def _envbool(v: str) -> bool:
        return v.lower() in ("1", "true", "yes")

    def _int_or_all(v: str):
        # refinement history-depth knobs: "all" -> None (all prior rounds), else int (0=none, N=last N)
        return None if v.strip().lower() == "all" else int(v)

    def _int_or_none(v: str):
        # max_prefix_length: "none"/"null" -> None (keep whole body, suffix attack); else int
        return None if v.strip().lower() in ("none", "null") else int(v)

    def _set_nested(d, path, value):
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = value

    # Simple single-field overrides: (env var, cfg path, converter). Applied when the var is set
    # and non-empty. Multi-field / nested overrides (eval model+thinking, jail) stay explicit below.
    ENV_OVERRIDES = [
        ("BLOOM_FOLDER",         ("folder_name",),                            str),
        ("BLOOM_SEED",           ("seed",),                                   int),
        ("BLOOM_MAX_TURNS",      ("rollout", "max_turns"),                    int),
        ("BLOOM_NUM_SCENARIOS",  ("ideation", "num_scenarios"),               int),
        ("BLOOM_NUM_ROUNDS",     ("rollout", "num_rounds"),                          int),
        ("BLOOM_TARGET_MODEL",   ("rollout", "target"),                       str),   # swap target without editing the default
        # BLOOM_EVAL_GPU moves the WHOLE auditor (understanding/ideation/judgment LocalModel + rollout
        # evaluator) onto this GPU. core._DEFAULT_LOCAL_GPU_ID is set from cfg.evaluator_gpu_id, so
        # without this the judgment-stage auditor stays on GPU 0 and two pipelines on different GPUs
        # collide ("engine core init failed on GPU 0").
        ("BLOOM_EVAL_GPU",       ("evaluator_gpu_id",),                       int),
        ("BLOOM_JUDGE_MODEL",    ("judgment", "model"),                       str),   # non-'local/' id => hosted API via litellm
        ("BLOOM_JUDGE_THINKING", ("judgment", "thinking"),                    _envbool),
        ("BLOOM_EVAL_MAXTOK",    ("rollout", "evaluator_max_tokens"),         int),   # raise eval cap for hosted-API eval WITH thinking (budget reserved inside max_tokens)
        ("BLOOM_JUDGE_MAXTOK",   ("judgment", "max_tokens"),                  int),
        ("BLOOM_KICKOFF_BANK",   ("kickoff_bank",),                           str),
        ("BLOOM_REFINE",         ("refinement_input", "enabled"),                   _envbool),
        ("BLOOM_REFINE_HIST_TRANSCRIPT", ("refinement_input", "history_transcript_rounds"), _int_or_all),  # "all"=None, 0=none, N=last N full transcripts
        ("BLOOM_REFINE_HIST_STRATEGY",   ("refinement_input", "history_strategy_rounds"),   _int_or_all),  # "all"=None, 0=none, N=last N (round,score,strategy) rows
        ("BLOOM_INPUT_SEARCH",   ("search_input", "enabled"),                 _envbool),
        ("BLOOM_INPUT_MAXPREFIX", ("search_input", "max_prefix_length"),      _int_or_none),   # int (e.g. -50, 0=regen whole body) or "none"=keep whole body (suffix attack)
        ("BLOOM_INPUT_ITERS",    ("search_input", "max_num_iterations"),      int),
        ("BLOOM_INPUT_NUM_BEAMS", ("search_input", "num_beams"),              int),   # BEAST beam width (hypotheses kept after selection)
        ("BLOOM_INPUT_CAND_PER_BEAM", ("search_input", "candidates_per_beam"), int),  # samples drawn per beam per iter; scored/iter = num_beams * candidates_per_beam
        ("BLOOM_INPUT_EVAL_CHUNK", ("search_input", "eval_beam_chunk_size"),  int),   # beams per eval vLLM call during BEAST candidate gen; 1 = sequential (cuts GPU-0 peak ~num_beams x), None default = all beams at once
        ("BLOOM_INPUT_SCORED_LEN", ("search_input", "scored_candidate_length"), int),  # tokens scored per BEAST iter (keep == kept_candidate_length)
        ("BLOOM_INPUT_KEPT_LEN",   ("search_input", "kept_candidate_length"),   int),  # tokens committed per BEAST iter (must be <= scored_candidate_length)
        ("BLOOM_INPUT_REWARD_LEN", ("search_input", "max_reward_output_length"), int),  # first N tokens of self-jail TRS used as the BEAST reward signal (0 = full TRS)
        ("BLOOM_INPUT_TRUNCATE_EOS", ("search_input", "truncate_at_eos"),       _envbool),  # True: candidate may emit </message>/EOS and truncate; False: keep raw content to scored_candidate_length
        ("BLOOM_INPUT_LATIN_MASK", ("search_input", "latin_mask"),             _envbool),  # restrict beam search to Latin/ASCII tokens only (blocks unicode/digits/punctuation)
        ("BLOOM_TARGET_BATCH_SIZE", ("target_batch_size",),                    int),  # target-model batch for input-search candidate scoring (raise to score more candidates per pass)
        # ── flrt_search_input (FLRT-in) hooks: mutation-buffer black-box search, full-vocab distillation loss ──
        ("BLOOM_FLRT_SEARCH",    ("flrt_search_input", "enabled"),            _envbool),
        ("BLOOM_FLRT_MAXPREFIX", ("flrt_search_input", "max_prefix_length"), _int_or_none),  # None=keep whole body (suffix mutation); 0=whole-input regeneration
        ("BLOOM_FLRT_ITERS",     ("flrt_search_input", "max_num_iterations"), int),   # iterations/trial (compute dial)
        ("BLOOM_FLRT_NTRIALS",   ("flrt_search_input", "n_trials"),          int),    # independent restarts merged into the pool
        ("BLOOM_FLRT_BUFFER",    ("flrt_search_input", "buffer_size"),       int),    # active buffer (best is mutated each iter)
        ("BLOOM_FLRT_K1",        ("flrt_search_input", "k1"),               int),     # mutated candidates generated + scored per iter
        ("BLOOM_FLRT_K2",        ("flrt_search_input", "k2"),               int),     # replacement tokens sampled per position (swap/insert)
        ("BLOOM_FLRT_EVAL_CHUNK", ("flrt_search_input", "eval_beam_chunk_size"), int),  # HF scoring batch chunk for the k1 candidates
        ("BLOOM_FLRT_P_APPEND",  ("flrt_search_input", "p_append"),         float),
        ("BLOOM_FLRT_P_INSERT",  ("flrt_search_input", "p_insert"),         float),
        ("BLOOM_FLRT_P_DELETE",  ("flrt_search_input", "p_delete"),         float),
        ("BLOOM_FLRT_P_SWAP",    ("flrt_search_input", "p_swap"),           float),
        ("BLOOM_FLRT_N_MUT",     ("flrt_search_input", "num_mutations"),    int),     # mutations applied to each candidate per iter (>1 = multi-token append/insert, multi-position delete/swap)
        ("BLOOM_FLRT_START_TOKENS", ("flrt_search_input", "start_tokens"),  int),
        ("BLOOM_FLRT_MIN_TOKENS", ("flrt_search_input", "min_tokens"),      int),
        ("BLOOM_FLRT_MAX_TOKENS", ("flrt_search_input", "max_tokens"),      _int_or_none),  # int cap, or "none" = unbounded suffix growth
        ("BLOOM_FLRT_REWARD_LEN", ("flrt_search_input", "max_reward_output_length"), int),  # first N target tokens of the self-jail continuation used as the distillation target
        ("BLOOM_FLRT_LATIN_MASK", ("flrt_search_input", "latin_mask"),      _envbool),
        ("BLOOM_FLRT_TRUNCATE_EOS", ("flrt_search_input", "truncate_at_eos"), _envbool),
        ("BLOOM_FLRT_PTHRESHOLD", ("flrt_search_input", "p_threshold"),     float),   # per-token reward cap = log(p_threshold)
        ("BLOOM_FLRT_W_DISTILL", ("flrt_search_input", "w_distillation"),   float),   # weight on distillation term (relative to w_force; z-normed when w_force>0)
        ("BLOOM_FLRT_W_FORCE",   ("flrt_search_input", "w_force"),          float),   # weight on teacher-forcing term (target logprob of teacher's actual tokens); >0 activates z-normed combine
        ("BLOOM_FLRT_TEMP",      ("flrt_search_input", "temperature"),      float),
        ("BLOOM_FLRT_W_FLUENCY", ("flrt_search_input", "w_fluency"),        float),   # 0 = off
        ("BLOOM_FLRT_FLUENCY_ON", ("flrt_search_input", "fluency_on"),      str),     # "auditor" | "target"
        ("BLOOM_FLRT_W_REPETITION", ("flrt_search_input", "w_repetition"),  float),   # 0 = off
        # ── search_output (BEAST-out) hooks: mirror of the input set, roles swapped ──
        # Target GENERATES response candidates; the auditor SCORES them by log P("Yes")
        # on the judge prompt.
        ("BLOOM_OUTPUT_SEARCH",      ("search_output", "enabled"),                 _envbool),
        ("BLOOM_OUTPUT_MAXPREFIX",   ("search_output", "max_prefix_length"),       _int_or_none),  # int (0=regenerate whole response) or "none"=keep whole natural response (suffix attack)
        ("BLOOM_OUTPUT_ITERS",       ("search_output", "max_num_iterations"),      int),   # 1 = single-pass Best-of-N; >1 = iterative BEAST
        ("BLOOM_OUTPUT_NUM_BEAMS",   ("search_output", "num_beams"),               int),   # beam width (hypotheses kept after selection)
        ("BLOOM_OUTPUT_CAND_PER_BEAM", ("search_output", "candidates_per_beam"),   int),   # samples drawn per beam per iter; scored/iter = num_beams * candidates_per_beam
        ("BLOOM_OUTPUT_EVAL_CHUNK",  ("search_output", "eval_beam_chunk_size"),    int),   # beams per judge-scoring call; 1 = sequential (cuts peak memory)
        ("BLOOM_OUTPUT_SCORED_LEN",  ("search_output", "scored_candidate_length"), int),   # tokens scored per iter (keep == kept_candidate_length)
        ("BLOOM_OUTPUT_KEPT_LEN",    ("search_output", "kept_candidate_length"),   int),   # tokens committed per iter (must be <= scored_candidate_length)
        ("BLOOM_OUTPUT_TRUNCATE_EOS", ("search_output", "truncate_at_eos"),        _envbool),  # True: candidate may emit EOS and terminate; False: keep generating to scored_candidate_length
        ("BLOOM_OUTPUT_LATIN_MASK",  ("search_output", "latin_mask"),              _envbool),  # restrict response search to Latin/ASCII tokens (off by default — target should speak naturally)
        ("BLOOM_OUTPUT_TEMP",        ("search_output", "temperature"),             float),   # candidate sampling temperature (default 1.0)
        ("BLOOM_OUTPUT_BEHAVIOR",    ("search_output", "behavior_name"),           str),   # what the judge asks about (independent of cfg.behavior_name)
    ]
    for _env, _path, _conv in ENV_OVERRIDES:
        _v = os.environ.get(_env)
        if _v not in (None, ""):
            _set_nested(cfg, _path, _conv(_v))

    # BLOOM_EVAL_MODEL: evaluator/red-team model for understanding + ideation + rollout turn-sampling
    # (e.g. "claude-haiku-4-5" via litellm). A non-'local/' id => hosted API: it can only sample whole
    # turns (token-level search must be off; enforced at evaluator load).
    if os.environ.get("BLOOM_EVAL_MODEL"):
        _eval_model = os.environ["BLOOM_EVAL_MODEL"]
        cfg.setdefault("understanding", {})["model"] = _eval_model
        cfg.setdefault("ideation", {})["model"] = _eval_model
        cfg.setdefault("rollout", {})["model"] = _eval_model
    if os.environ.get("BLOOM_EVAL_THINKING") is not None and os.environ.get("BLOOM_EVAL_THINKING") != "":
        # toggle reasoning for understanding+ideation+rollout-evaluator. For a hosted-API eval,
        # reasoning_effort='medium' reserves the thinking budget (2048) inside max_tokens, so the
        # small per-call caps (eval 1200, understanding 2000) would fail litellm's budget check —
        # set this 0 to run the API evaluator without extended thinking (also much cheaper).
        _et = _envbool(os.environ["BLOOM_EVAL_THINKING"])
        cfg.setdefault("understanding", {})["thinking"] = _et
        cfg.setdefault("ideation", {})["thinking"] = _et
        cfg.setdefault("rollout", {})["evaluator_thinking"] = _et
    # BLOOM_JAIL_MODEL: switch to DIRECT jail-sample decoding (contrastive PoE with a jailbroken
    # proposal model). (This shim is WILT-only — there is no corruption path here to disable.)
    # Jail model must share the target's vocab (target itself = self-jail, or a same-family abliterated).
    if os.environ.get("BLOOM_JAIL_MODEL"):
        cfg.setdefault("jailbroken_output", {})["enabled"] = True
        cfg["jailbroken_output"]["model"] = os.environ["BLOOM_JAIL_MODEL"]
        if os.environ.get("BLOOM_JAIL_BETA"):
            cfg["jailbroken_output"]["b2"] = float(os.environ["BLOOM_JAIL_BETA"])
        if os.environ.get("BLOOM_JAIL_B1") is not None and os.environ.get("BLOOM_JAIL_B1") != "":
            cfg["jailbroken_output"]["b1"] = float(os.environ["BLOOM_JAIL_B1"])       # 0 = floor-only jail (drop target term; keep floor). unset = legacy z=target+b2*jail
        if os.environ.get("BLOOM_JAIL_FLOOR"):
            cfg["jailbroken_output"]["target_floor"] = float(os.environ["BLOOM_JAIL_FLOOR"])  # mask jail samples to tokens with target prob >= floor
        if os.environ.get("BLOOM_JAIL_PREFILL") not in (None, ""):
            cfg["jailbroken_output"]["prefill"] = _envbool(os.environ["BLOOM_JAIL_PREFILL"])  # False = no compliance prefill (ablation row)
        # NEGATIVE STEERING for jail: z = target + beta*jail - b3*neg, where neg is the jail expert
        # prompted with a negative persona + the SAME input (a continuation, NOT a rewrite). b3 is the
        # only knob here (0 = off = legacy jail); the neg prompts load from the behaviour yaml
        # (jailbroken_output_neg_system_prompt / _neg_user_prompt / _neg_prefill).
        if os.environ.get("BLOOM_JAIL_B3") is not None and os.environ.get("BLOOM_JAIL_B3") != "":
            cfg["jailbroken_output"]["b3"] = float(os.environ["BLOOM_JAIL_B3"])
    if any(os.environ.get(k) for k in ("BLOOM_FOLDER", "BLOOM_MAX_TURNS", "BLOOM_NUM_ROUNDS")):
        print(f"  [env override] folder={cfg.get('folder_name')} "
              f"max_turns={cfg.get('rollout', {}).get('max_turns')} "
              f"num_rounds={cfg.get('rollout', {}).get('num_rounds')}", flush=True)

    base_folder = cfg.get("folder_name", "runs_new/default")
    num_rounds = cfg.get("rollout", {}).get("num_rounds", 5)
    base_seed = cfg.get("seed")  # offset per round to keep vLLM samples reproducible-but-distinct across rounds
    async def run_parallel() -> bool:
        """Returns True if there was an error."""
        # Set the default GPU here too: run_pipeline sets it, but is skipped on resume (round_1/judgment.json exists),
        # so without this the judgment stage would spawn a redundant judge worker on GPU 0.
        core._DEFAULT_LOCAL_GPU_ID = cfg.get("evaluator_gpu_id", 0)
        core._DEFAULT_MAX_MODEL_LEN = int(os.environ.get("BLOOM_EVAL_MAXLEN", cfg.get("rollout", {}).get("evaluator_max_model_len", 16384)))
        # Round 1: full pipeline (skipped if already complete — detected via judgment.json)
        print("\n" + "#" * 60, flush=True)
        print(f"# SELF-REFINE ROUND 1/{num_rounds}  [full pipeline]", flush=True)
        print("#" * 60, flush=True)
        round_1_dir = (RUNS_ROOT / base_folder / "round_1").resolve()
        _bk1 = _bank_load_round(cfg.get("kickoff_bank"), 1, cfg.rollout.model) if cfg.get("kickoff_bank") else None
        cfg.folder_name = f"{base_folder}/round_1"
        round_1_judgment = round_1_dir / "judgment.json"
        if round_1_judgment.exists():
            print("ROUND 1 - skipped (round_1/judgment.json already exists)", flush=True)
            with open(round_1_judgment, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            if base_seed is not None:
                core._DEFAULT_SEED = base_seed + 1
            # Round-1 kickoff-bank reuse is applied INSIDE run_pipeline (pipeline.py, after
            # ideation loads), which works for fresh + resumed ideation. The old inject-here
            # block ran before round_1/ideation.json existed and was silently skipped, so round 1
            # diverged per method. `_bk1` above is kept only to gate the save below (build-once).
            result = await run_pipeline(cfg)
            if not result:
                print("\n  Round 1 FAILED", flush=True)
                return True
        stats = result.get("summary_statistics", {})
        print(f"\n  Round 1: avg={stats.get('average_behavior_presence_score', 0):.2f}, "
            f"elicitation_rate={stats.get('elicitation_rate', 0):.2f}"
            + (f", tok_avg={stats['A_mean_tok_pct']:.1f}%, tok_meanmin={stats['B_mean_of_mins_pct']:.2f}%, "
               f"tok_min={stats['B_min_of_mins_pct']:.4f}%" if stats.get('A_mean_tok_pct') is not None else ""), flush=True)
        # Logprob scoring is computed inline during rollout — no separate stage needed.
        if cfg.get("kickoff_bank") and not _bk1:
            _bank_save_round(cfg.get("kickoff_bank"), 1, round_1_dir, cfg.rollout.model)
        # Load understanding from round 1 for reuse in all subsequent rounds
        with open(round_1_dir / "understanding.json", "r", encoding="utf-8") as f:
            understanding_results = json.load(f)
        # Reuse round 1's SCENARIOS in every later round. Both round-2+ paths (the refiner
        # and the pure-resample/freeze path) read these as the FROZEN per-variation
        # `description`. Previously this was emptied to {"variations": []}, so every
        # round-2+ scenario description fell through to "" — which made all 25 kickoff
        # prompts identical and collapsed the inputs to a single scenario. Refinement adds
        # `refine_context` on top via variations_override; it doesn't replace the scenarios.
        with open(round_1_dir / "ideation.json", "r", encoding="utf-8") as f:
            ideation_results = json.load(f)
        prompts_yaml = load_prompts(cfg)

        # Rounds 2+: refine each scenario using full accumulated history
        completed_round_dirs: List[Path] = [round_1_dir]
        for round_num in range(2, num_rounds + 1):
            print("\n" + "#" * 60, flush=True)
            print(f"# SELF-REFINE ROUND {round_num}/{num_rounds}  [refine + rollout + judge]", flush=True)
            print("#" * 60, flush=True)
            output_dir = (RUNS_ROOT / base_folder / f"round_{round_num}").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            if (output_dir / "judgment.json").exists():
                print(f"# ROUND {round_num} - skipped (already exists, reusing)", flush=True)
                completed_round_dirs.append(output_dir)
                continue
            save_json(_cfg_for_dump(cfg, prompts_yaml), output_dir / "cfg.json")
            if base_seed is not None:
                core._DEFAULT_SEED = base_seed + round_num
            _bkr = _bank_load_round(cfg.get("kickoff_bank"), round_num, cfg.rollout.model) if cfg.get("kickoff_bank") else None
            result = await run_parallel_round(
                completed_round_dirs, output_dir, understanding_results, ideation_results, cfg, prompts_yaml
            )
            completed_round_dirs.append(output_dir)
            if cfg.get("kickoff_bank") and not _bkr:
                _bank_save_round(cfg.get("kickoff_bank"), round_num, output_dir, cfg.rollout.model)
            if result:
                stats = result.get("summary_statistics", {})
                print(f"\n  Round {round_num}: avg={stats.get('average_behavior_presence_score', 0):.2f}, "
                    f"elicitation_rate={stats.get('elicitation_rate', 0):.2f}"
                    + (f", tok_avg={stats['A_mean_tok_pct']:.1f}%, tok_meanmin={stats['B_mean_of_mins_pct']:.2f}%, "
                       f"tok_min={stats['B_min_of_mins_pct']:.4f}%" if stats.get('A_mean_tok_pct') is not None else ""), flush=True)
            else:
                print(f"\n  Round {round_num} FAILED", flush=True)
                return True
        return False

    # Track total experiment runtime
    _experiment_start = time.monotonic()
    had_error = asyncio.run(run_parallel())
    _elapsed = time.monotonic() - _experiment_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s = divmod(_rem, 60)
    print("\n" + "=" * 60, flush=True)
    print(f"TOTAL EXPERIMENT TIME: {_h}h {_m}m {_s}s ({_elapsed:.1f}s)", flush=True)
    print("=" * 60, flush=True)

    # Hard exit: bypass Python shutdown (vLLM workers can hang non-daemon threads
    # forever). All useful output is already on disk. Exit code propagates to the
    # shell so `&&` chaining works correctly.
    # BUT os._exit() skips atexit, so explicitly tear the workers down FIRST — otherwise the
    # vLLM auditor's process tree is orphaned to init and keeps its GPU memory (the leak that
    # used to force a reboot between runs). _shutdown_all_workers -> shutdown() killpgs each
    # worker's whole process group.
    _shutdown_all_workers()
    os._exit(1 if had_error else 0)
