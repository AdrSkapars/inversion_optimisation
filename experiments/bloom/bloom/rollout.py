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
from .wilt import *
from .search import *



# =============================================================================
# Section 6: Stage 1 - Understanding
# =============================================================================
def run_understanding(cfg: DotDict, prompts_yaml: Dict, output_dir: Path) -> Dict[str, Any]:
    """Run the understanding stage."""
    print("\n" + "=" * 60, flush=True)
    print("UNDERSTANDING STAGE - STARTED", flush=True)
    print("=" * 60, flush=True)

    behavior_name = cfg.behavior_name
    behavior_description = cfg.behavior_description
    model_id = cfg.understanding.model
    max_tokens = cfg.understanding.get("max_tokens", 4000)
    reasoning_effort = _effort(cfg.understanding.get("thinking", False))
    temperature = cfg.get("temperature")
    seed = cfg.get("seed")

    # Build system prompt
    system_prompt = build_understanding_system(prompts_yaml)

    # Step 1: Behavior understanding
    messages = []
    understanding_prompt = build_behavior_understanding_prompt(
        behavior_name, behavior_description, prompts_yaml
    )
    messages.append({"role": "user", "content": understanding_prompt})

    debug_print("Calling model for behavior understanding...")
    response = litellm_chat(
        model_id=model_id, messages=messages, system_prompt=system_prompt,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort, temperature=temperature,
        seed=seed,
    )
    parsed = parse_message(response)
    understanding_response = parsed["content"] or ""
    reasoning_content = parsed["reasoning"]

    # Parse understanding and motivation
    understanding_match = re.search(r"<behavior_understanding>(.*?)</behavior_understanding>", understanding_response, re.DOTALL)
    understanding = understanding_match.group(1).strip() if understanding_match else ""

    motivation_match = re.search(r"<scientific_motivation>(.*?)</scientific_motivation>", understanding_response, re.DOTALL)
    scientific_motivation = motivation_match.group(1).strip() if motivation_match else ""

    understanding_reasoning = reasoning_content or ""
    if reasoning_content:
        understanding_response = f"<thinking>\n{reasoning_content.strip()}\n</thinking>\n\n{understanding_response}"

    messages.append({"role": "assistant", "content": understanding_response})

    # Step 2: Process example transcripts
    examples = cfg.get("examples", [])
    transcript_analyses = []

    for i, example in enumerate(examples):
        debug_print(f"Processing example {i + 1}/{len(examples)}...")
        # example should be a dict with 'conversation' key: [{role, content}, ...]
        transcript_text = ""
        if isinstance(example, dict) and "conversation" in example:
            for msg in example["conversation"]:
                role = msg.get("role", "unknown").upper()
                if role == "ASSISTANT":
                    role = "TARGET"
                elif role == "USER":
                    role = "EVALUATOR"
                transcript_text += f"{role}: {msg.get('content', '')}\n\n"
        elif isinstance(example, str):
            transcript_text = example
        else:
            continue

        analysis_prompt = build_transcript_analysis_prompt(
            behavior_name, understanding, scientific_motivation,
            transcript_text, prompts_yaml
        )
        messages.append({"role": "user", "content": analysis_prompt})

        try:
            response = litellm_chat(
                model_id=model_id, messages=messages, system_prompt=system_prompt,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort, temperature=temperature,
                seed=seed,
            )
            parsed = parse_message(response)
            analysis_response = parsed["content"] or ""
            reasoning_content = parsed["reasoning"]
        except Exception as e:
            debug_print(f"API call failed for example {i+1}: {e}")
            continue

        analysis_reasoning = reasoning_content or ""
        if reasoning_content:
            analysis_response = f"<thinking>\n{reasoning_content.strip()}\n</thinking>\n\n{analysis_response}"

        messages.append({"role": "assistant", "content": analysis_response})

        # Parse summary and attribution
        summary_match = re.search(r"<transcript_summary>(.*?)</transcript_summary>", analysis_response, re.DOTALL)
        attribution_match = re.search(r"<attribution>(.*?)</attribution>", analysis_response, re.DOTALL)

        transcript_analyses.append({
            "example_name": f"example_{i+1}",
            "transcript_summary": summary_match.group(1).strip() if summary_match else "",
            "attribution": attribution_match.group(1).strip() if attribution_match else "",
            "reasoning": analysis_reasoning,
        })

    # Save results
    results = {
        "behavior_name": behavior_name,
        "model": model_id,
        "understanding": understanding,
        "scientific_motivation": scientific_motivation,
        "understanding_reasoning": understanding_reasoning,
        "transcript_analyses": transcript_analyses,
    }
    save_json(results, output_dir / "understanding.json")

    print("UNDERSTANDING STAGE - COMPLETED", flush=True)
    return results


# =============================================================================
# Section 7: Stage 2 - Ideation
# =============================================================================
def parse_scenarios_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse scenarios from <scenario> tags or heading patterns.

    Handles three formats the model may produce:
      1. Proper wrapping:  <scenario>...</scenario> per scenario
      2. Separator style:  content <scenario> content <scenario> ... </scenario>
         (model uses <scenario> as an end-marker; content may also precede the first tag)
      3. No tags:          **Scenario N: ...** headings only
    """
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _has_heading(text: str) -> bool:
        return bool(re.search(r"\*\*Scenario \d+:", text))

    scenarios: list[dict] = []

    # ── Strategy 1: <scenario> as separator / end-marker ─────────────────────
    # Split on every <scenario> tag; each resulting block is one scenario's text.
    # Also handles proper wrapping because the splits land between tags.
    if "<scenario>" in response_text:
        parts = re.split(r"<scenario>", response_text)
        for part in parts:
            # Strip a trailing </scenario> (and anything after it)
            part = re.sub(r"</scenario>.*", "", part, flags=re.DOTALL).strip()
            if not part:
                continue
            # A block may still contain multiple "**Scenario N:**" headings if the
            # model wrapped several in one tag — split those out too.
            if _has_heading(part):
                sub_blocks = re.split(r"(?=\*\*Scenario \d+:)", part)
                for sub in sub_blocks:
                    sub = sub.strip()
                    if sub and _has_heading(sub):
                        scenarios.append({"description": _clean(sub)})
            else:
                # Content without a heading (e.g. brainstorm preamble) — skip
                pass
        if scenarios:
            return scenarios

    # ── Strategy 2: proper <scenario>...</scenario> pairs ────────────────────
    matches = re.findall(r"<scenario>(.*?)</scenario>", response_text, re.DOTALL)
    if matches:
        for m in matches:
            sub_blocks = re.split(r"(?=\*\*Scenario \d+:)", m)
            for sub in sub_blocks:
                sub = sub.strip()
                if sub:
                    scenarios.append({"description": _clean(sub)})
        if scenarios:
            return scenarios

    # ── Strategy 3: no tags — split on **Scenario N: headings ────────────────
    splits = re.split(r"\*\*Scenario \d+:", response_text)
    if len(splits) > 1:
        for split in splits[1:]:
            clean = _clean(split)
            if clean:
                scenarios.append({"description": clean})

    return scenarios


def run_ideation(cfg: DotDict, prompts_yaml: Dict, output_dir: Path,
                 understanding_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run the ideation stage."""
    print("\n" + "=" * 60, flush=True)
    print("IDEATION STAGE - STARTED", flush=True)
    print("=" * 60, flush=True)

    behavior_name = cfg.behavior_name
    model_id = cfg.ideation.model
    num_scenarios = cfg.ideation.get("num_scenarios", 10)
    max_tokens = cfg.ideation.get("max_tokens", 50000)
    reasoning_effort = _effort(cfg.ideation.get("thinking", False))
    temperature = cfg.get("temperature")
    seed = cfg.get("seed")
    max_turns = cfg.rollout.get("max_turns", 5)

    behavior_understanding = understanding_results["understanding"]
    scientific_motivation = understanding_results["scientific_motivation"]
    transcript_analyses = understanding_results.get("transcript_analyses", [])

    # Build system prompt
    system_prompt = build_ideation_system(
        behavior_understanding, scientific_motivation, transcript_analyses,
        behavior_name, prompts_yaml
    )

    if model_id.startswith("local/"):
        # ── Local model fast path ────────────────────────────────────────────
        # No API output token cap, so generate all scenarios in a single call.
        print(f"Generating {num_scenarios} scenarios in 1 call (local model)...", flush=True)
        prompt = build_scenarios_prompt(
            behavior_name, num_scenarios,
            behavior_understanding, scientific_motivation, transcript_analyses,
            prompts_yaml,
            start_idx=1, end_idx=num_scenarios,
        )
        hf_name = model_id[len("local/"):]
        lm = _get_local_model(hf_name)
        raw = batch_generate_local(
            lm,
            [[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]],
            max_new_tokens=max_tokens,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            no_think=reasoning_effort == "none",
            seed=seed,
        )[0]
        all_scenarios = parse_scenarios_response(raw)
        print(f"Got {len(all_scenarios)} scenarios (expected {num_scenarios})", flush=True)
        # Robust top-up: the local model sometimes emits N-1 scenarios and stops at EOS. Rather
        # than fail, continue from the accumulated context asking ONLY for the missing ones
        # (renumbered to resume), up to a few attempts — mirrors the API multi-batch path.
        _ide_msgs = [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": prompt},
                     {"role": "assistant", "content": raw}]
        _attempt = 0
        while len(all_scenarios) < num_scenarios and _attempt < 3:
            _attempt += 1
            _have = len(all_scenarios)
            _topup = build_scenarios_prompt(
                behavior_name, num_scenarios,
                behavior_understanding, scientific_motivation, transcript_analyses,
                prompts_yaml, start_idx=_have + 1, end_idx=num_scenarios,
            )
            print(f"  Top-up {_attempt}/3: have {_have}, requesting scenarios {_have + 1}-{num_scenarios}...", flush=True)
            _ide_msgs.append({"role": "user", "content": _topup})
            _raw2 = batch_generate_local(
                lm, [_ide_msgs], max_new_tokens=max_tokens,
                temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
                no_think=reasoning_effort == "none",
                seed=seed,   # keep the run's seed; the growing context (prior answer + new ask) differs each attempt
            )[0]
            _ide_msgs.append({"role": "assistant", "content": _raw2})
            _more = parse_scenarios_response(_raw2)
            all_scenarios.extend(_more)
            print(f"    Got {len(_more)} more (total {len(all_scenarios)}/{num_scenarios})", flush=True)
    else:
        # ── API path: multi-batch with accumulated context ───────────────────
        # Needed because API models have a hard output token cap.
        model_max = get_model_max_output_tokens(model_id)
        if max_tokens > model_max:
            debug_print(f"Capping max_tokens from {max_tokens} to model max {model_max}")
            max_tokens = model_max

        batch_size, num_batches = calculate_batch_size(
            total_scenarios=num_scenarios, model_id=model_id,
            reasoning_effort=reasoning_effort, safety_margin=0.75,
        )

        print(f"Generating {num_scenarios} scenarios in {num_batches} batch(es)...", flush=True)

        all_scenarios = []
        messages = []  # Accumulated context across batches

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size + 1
            end_idx = min(start_idx + batch_size - 1, num_scenarios)
            batch_count = end_idx - start_idx + 1

            print(f"  Batch {batch_num + 1}/{num_batches}: scenarios {start_idx}-{end_idx}", flush=True)

            batch_prompt = build_scenarios_prompt(
                behavior_name, num_scenarios,
                behavior_understanding, scientific_motivation, transcript_analyses,
                prompts_yaml,
                start_idx=start_idx, end_idx=end_idx,
            )

            messages.append({"role": "user", "content": batch_prompt})

            response = litellm_chat(
                model_id=model_id, messages=messages, system_prompt=system_prompt,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort, temperature=temperature,
                seed=seed,
            )
            parsed = parse_message(response)
            ideation_response = parsed["content"] or ""
            reasoning_content = parsed["reasoning"]

            if reasoning_content:
                ideation_response = f"<thinking>\n{reasoning_content.strip()}\n</thinking>\n\n{ideation_response}"

            messages.append({"role": "assistant", "content": ideation_response})

            batch_scenarios = parse_scenarios_response(ideation_response)
            print(f"    Got {len(batch_scenarios)} scenarios (expected {batch_count})", flush=True)
            all_scenarios.extend(batch_scenarios)

    print(f"Total scenarios generated: {len(all_scenarios)}", flush=True)

    # Trim any overshoot from top-up, then proceed. Only a total parse failure (0 scenarios) is
    # fatal; a small shortfall that survived the top-up attempts is tolerated (the rollout just
    # runs fewer variations) rather than crashing the whole pipeline.
    all_scenarios = all_scenarios[:num_scenarios]
    if not all_scenarios:
        raise RuntimeError(
            f"Ideation produced 0 scenarios (requested {num_scenarios}) — the parser likely failed "
            f"to split the model output. Check the raw response format."
        )
    if len(all_scenarios) < num_scenarios:
        print(f"  WARNING: proceeding with {len(all_scenarios)}/{num_scenarios} scenarios "
              f"(top-up could not reach the target) — rollout will run fewer.", flush=True)

    # Save results - use "variations" key for compatibility
    results = {
        "behavior_name": behavior_name,
        "model": model_id,
        "num_scenarios": num_scenarios,
        "total_evals": len(all_scenarios),
        "variations": all_scenarios,  # Each is {"description": "..."}
    }
    save_json(results, output_dir / "ideation.json")

    print("IDEATION STAGE - COMPLETED", flush=True)
    return results


# =============================================================================
# Section 8: Stage 3 - ConversationOrchestrator + Rollout
# =============================================================================

# Shared evaluator instruction strings, sourced from prompts.yaml (behaviour-agnostic base keys
# with no per-behaviour override), so all prompt text lives in the yaml. Loaded once at import.
_BASE_PROMPTS = yaml.safe_load(open(SCRIPT_DIR / "prompts" / "prompts.yaml", encoding="utf-8"))
_STEERING_INSTRUCTIONS = _BASE_PROMPTS["steering_instructions"]
_TAG_INSTRUCTIONS_MSG_FIRST = _BASE_PROMPTS["tag_instructions_msg_first"]
_TAG_INSTRUCTIONS_TRS_FIRST = _BASE_PROMPTS["tag_instructions_trs_first"]
_TAG_INSTRUCTIONS_MSG_ONLY = _BASE_PROMPTS["tag_instructions_msg_only"]


def _tag_instructions(target_before_input: bool, include_trs: bool = True) -> str:
    """Return the per-turn output-format instructions. When `target_before_input=True`,
    asks the evaluator to commit to a target response (TRS) BEFORE writing the user
    message, so the message can be crafted to fit the planned response. Otherwise the
    original order (message then TRS) is used.

    When `include_trs=False` (input_search disabled), the TRS is not requested at all —
    the evaluator only writes the <message>, since the TRS would serve no purpose."""
    if not include_trs:
        return _TAG_INSTRUCTIONS_MSG_ONLY
    return _TAG_INSTRUCTIONS_TRS_FIRST if target_before_input else _TAG_INSTRUCTIONS_MSG_FIRST
# Used at turn 2+ only: forces the evaluator to plan its next move before writing it.
_STRATEGY_INSTRUCTIONS_HEAD = _BASE_PROMPTS["strategy_instructions_head"]


def _strategy_instructions(include_trs: bool = True) -> str:
    """Turn-2+ planning block. The trailing clause names the tags that follow, which
    depends on whether the TRS is being requested (input_search on/off)."""
    tail = "<message> and <targeted_response_start> tags below." if include_trs else "<message> tag below."
    return _STRATEGY_INSTRUCTIONS_HEAD + tail


def _build_evaluator_followup(content: str, reasoning: Optional[str], current_turn: int, max_turns: int,
                              between_turns_strategise: bool = True,
                              partial_history: bool = False,
                              target_before_input: bool = False,
                              include_trs: bool = True) -> str:
    """Build the evaluator follow-up prompt after a target response.

    When `between_turns_strategise=True`, asks the evaluator to deliberately plan in <strategy>
    tags before writing the next message. When False, the strategy block is omitted and the
    evaluator just generates the message directly (vanilla baseline).

    When `partial_history=True`, prepends a hint that the conversation context shown is
    truncated, so the evaluator doesn't try to reference earlier turns it can no longer see.

    When `target_before_input=True`, asks for <targeted_response_start> BEFORE <message>, so
    the message can be designed to fit a pre-committed target response. Helps BoN regenerate
    on-topic messages by giving the model TRS context before sampling the body.
    """
    reasoning_block = f"<target_reasoning>\n{reasoning}\n</target_reasoning>\n\n" if reasoning else ""
    strategy_block = f"{_strategy_instructions(include_trs)} " if between_turns_strategise else ""
    history_note = _BASE_PROMPTS["evaluator_followup_history_note"] if partial_history else ""
    return _BASE_PROMPTS["evaluator_followup"].format(
        history_note=history_note,
        reasoning_block=reasoning_block,
        content=content,
        current_turn=current_turn,
        max_turns=max_turns,
        steering=_STEERING_INSTRUCTIONS,
        strategy_block=strategy_block,
        tag_instructions=_tag_instructions(target_before_input, include_trs),
    )


def _kickoff_message(generate_kickoff_additional: str, target_before_input: bool,
                     include_trs: bool, refined_strategy: str = "") -> str:
    """Shared kickoff (first evaluator user turn) builder for both rollout paths.
    Round 1 turn 1 has no strategy block; rounds 2+ inject a refined-strategy block."""
    gka = (generate_kickoff_additional.strip() + " ") if generate_kickoff_additional else ""
    kp = _BASE_PROMPTS["kickoff_prompt"].format(generate_kickoff_additional=gka) + "\n\n"
    if refined_strategy:
        kp += _BASE_PROMPTS["kickoff_refined_strategy_block"].format(refined_strategy=refined_strategy)
    kp += _tag_instructions(target_before_input, include_trs)
    return kp


class ConversationOrchestrator:
    """Conducts a conversation rollout between an evaluator model and a target model."""

    def __init__(
        self,
        evaluator_model: str,
        target_model: str,
        evaluator_system_prompt: str,
        target_system_prompt: str,
        max_turns: int = 5,
        evaluator_reasoning_effort: str = "none",
        target_reasoning_effort: str = "none",
        evaluator_max_tokens: int = 4000,
        target_max_tokens: int = 4000,
        evaluator_temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC,
        target_temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC,
        target_kickoff_prefix: str = "",
        generate_kickoff_additional: str = "",
        target_before_input: bool = False,
        between_turns_strategise: bool = True,
        include_trs: bool = True,
    ):
        self.evaluator_model = evaluator_model
        self.target_model = target_model
        self.evaluator_system_prompt = evaluator_system_prompt
        self.target_system_prompt = target_system_prompt
        self.target_before_input = target_before_input
        self.between_turns_strategise = between_turns_strategise
        self.include_trs = include_trs
        # Set by setup() — merged into the first user message in run().
        self.pending_rollout_prompt: str = ""
        self.max_turns = max_turns
        self.evaluator_reasoning_effort = evaluator_reasoning_effort
        self.target_reasoning_effort = target_reasoning_effort
        self.evaluator_max_tokens = evaluator_max_tokens
        self.target_max_tokens = target_max_tokens
        self.evaluator_temperature = evaluator_temperature
        self.target_temperature = target_temperature
        self.target_kickoff_prefix = target_kickoff_prefix
        self.generate_kickoff_additional = generate_kickoff_additional
        self.current_turn = 0

        # Message histories
        self.evaluator_messages = []
        if self.evaluator_system_prompt:
            self.evaluator_messages.append({"role": "system", "content": self.evaluator_system_prompt})
        self.target_messages = []
        if self.target_system_prompt:
            self.target_messages.append({"role": "system", "content": self.target_system_prompt})

        # Simplified transcript (list of messages)
        self.transcript_messages = []
        if self.target_system_prompt:
            self.transcript_messages.append({
                "role": "system", "content": self.target_system_prompt, "source": "target_system"
            })

    def evaluator(self) -> Optional[Dict[str, Any]]:
        """Generate an evaluator message."""
        try:
            response = litellm_chat(
                model_id=self.evaluator_model,
                messages=self.evaluator_messages,
                max_tokens=self.evaluator_max_tokens,
                temperature=self.evaluator_temperature,
                reasoning_effort=self.evaluator_reasoning_effort,
            )
        except Exception as e:
            debug_print(f"Evaluator API call failed: {e}")
            return None

        parsed = parse_message(response)
        content = parsed["content"] or ""

        # Add to evaluator history (full response with tags stays in evaluator context)
        self.evaluator_messages.append(parsed["cleaned_message"])

        # Extract message, targeted response start, and strategy from tagged output
        user_message, targeted_response_start, strategy = _extract_message_tags(content)

        # Add user message to target history (without tags)
        target_content = user_message
        if self.target_kickoff_prefix and len(self.target_messages) == 1:
            target_content = self.target_kickoff_prefix.strip() + " " + user_message
        self.target_messages.append({"role": "user", "content": target_content})

        # Add to transcript (message only sent to target; strategy + TRS saved for analysis)
        transcript_entry: Dict[str, Any] = {
            "role": "user", "content": target_content, "source": "evaluator"
        }
        if strategy:
            transcript_entry["strategy"] = strategy
        if targeted_response_start:
            transcript_entry["targeted_response_start"] = targeted_response_start
        self.transcript_messages.append(transcript_entry)

        return parsed

    def target(self) -> Optional[Dict[str, Any]]:
        """Generate a target model response."""
        try:
            response = litellm_chat(
                model_id=self.target_model,
                messages=self.target_messages,
                max_tokens=self.target_max_tokens,
                temperature=self.target_temperature,
                reasoning_effort=self.target_reasoning_effort,
            )
        except Exception as e:
            debug_print(f"Target API call failed: {e}")
            return None

        parsed = parse_message(response)
        content = parsed["content"] or ""
        reasoning = parsed["reasoning"]

        # Add to target history
        self.target_messages.append(parsed["cleaned_message"])

        # Add to transcript
        msg = {"role": "assistant", "content": content, "source": "target"}
        if reasoning:
            msg["reasoning"] = reasoning
        self.transcript_messages.append(msg)

        # Add to evaluator history with scaffolding
        self.evaluator_messages.append({"role": "user", "content": _build_evaluator_followup(
            content, reasoning, self.current_turn, self.max_turns,
            between_turns_strategise=getattr(self, "between_turns_strategise", True),
            target_before_input=getattr(self, "target_before_input", False),
            include_trs=getattr(self, "include_trs", True),
        )})

        return parsed

    @classmethod
    def setup(
        cls,
        evaluator_model_id: str,
        target_model_id: str,
        evaluator_system_prompt: str,
        conversation_rollout_prompt: str,
        target_system_prompt: str = "",
        target_sysprompt_prefix: str = "",
        max_turns: int = 5,
        evaluator_reasoning_effort: str = "none",
        target_reasoning_effort: str = "none",
        evaluator_max_tokens: int = 4000,
        target_max_tokens: int = 4000,
        evaluator_temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC,
        target_temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC,
        target_kickoff_prefix: str = "",
        generate_kickoff_additional: str = "",
        target_before_input: bool = False,
        between_turns_strategise: bool = True,
        include_trs: bool = True,
    ) -> "ConversationOrchestrator":
        """Setup orchestrator with a fixed target system prompt (no LLM setup pass).
        The conversation_rollout_prompt is stashed and merged into the first user
        message in run(), which keeps the evaluator's chat history valid (no two
        consecutive user turns) while skipping the per-variation setup forward pass."""
        # Apply optional prefix from prompts.yaml
        if target_sysprompt_prefix and target_sysprompt_prefix.strip() and target_system_prompt:
            target_system_prompt = f"{target_sysprompt_prefix.strip()}\n\n{target_system_prompt}"

        orchestrator = cls(
            evaluator_model=evaluator_model_id,
            target_model=target_model_id,
            evaluator_system_prompt=evaluator_system_prompt,
            target_system_prompt=target_system_prompt,
            max_turns=max_turns,
            evaluator_reasoning_effort=evaluator_reasoning_effort,
            target_reasoning_effort=target_reasoning_effort,
            evaluator_max_tokens=evaluator_max_tokens,
            target_max_tokens=target_max_tokens,
            evaluator_temperature=evaluator_temperature,
            target_temperature=target_temperature,
            target_kickoff_prefix=target_kickoff_prefix,
            target_before_input=target_before_input,
            between_turns_strategise=between_turns_strategise,
            generate_kickoff_additional=generate_kickoff_additional,
            include_trs=include_trs,
        )

        orchestrator.pending_rollout_prompt = conversation_rollout_prompt

        return orchestrator

    def run(self) -> Dict[str, Any]:
        """Run the full conversation."""
        try:
            self.current_turn = 0

            # Kickoff (round 1 turn 1 path — no strategy block ever; refinement-driven
            # injection is local-batched-only for now).
            kickoff_prompt = _kickoff_message(
                self.generate_kickoff_additional,
                getattr(self, "target_before_input", False),
                getattr(self, "include_trs", True),
            )

            # Merge the stashed rollout prompt (scenario context) with the kickoff
            # into a single user turn — avoids two consecutive user messages now that
            # the per-variation setup-generation pass is gone.
            first_user_msg = (
                f"{self.pending_rollout_prompt}\n\n{kickoff_prompt}"
                if self.pending_rollout_prompt else kickoff_prompt
            )
            self.evaluator_messages.append({"role": "user", "content": first_user_msg})

            # Generate initial evaluator message
            eval_parsed = self.evaluator()
            if eval_parsed is None:
                return self._finalize()

            for turn in range(self.max_turns):
                self.current_turn = turn + 1
                debug_print(f"=====Turn {self.current_turn}=====")

                # Target response
                target_parsed = self.target()
                if target_parsed is None:
                    debug_print(f"Target API failure at turn {self.current_turn}")
                    break

                # Check if max turns reached
                if self.current_turn >= self.max_turns:
                    debug_print(f"Completed {self.max_turns} turns")
                    break

                # Generate evaluator response
                eval_parsed = self.evaluator()
                if eval_parsed is None:
                    break

                # Check for <END>
                if "<END>" in (eval_parsed["content"] or ""):
                    debug_print(f"Evaluator ended conversation at turn {self.current_turn}")
                    break

            return self._finalize()

        except Exception as e:
            debug_print(f"Critical error in run(): {e}")
            return self._finalize()

    def _finalize(self) -> Dict[str, Any]:
        """Return the transcript data."""
        return {
            "metadata": {
                "evaluator_model": self.evaluator_model,
                "target_model": self.target_model,
                "target_system_prompt": self.target_system_prompt,
                "created_at": datetime.now().isoformat(),
            },
            "messages": self.transcript_messages,
            "judgment": None,
        }


async def run_single_rollout(
    variation_number: int, variation_description: str, repetition_number: int,
    cfg: DotDict, prompts_yaml: Dict,
    behavior_understanding: str, scientific_motivation: str,
    transcript_analyses: List[Dict], evaluator_system_prompt: str,
    output_dir: Path, semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run a single rollout variation."""
    async with semaphore:
        rollout_cfg = cfg.rollout
        evaluator_model_id = rollout_cfg.model
        target_model_id = rollout_cfg.target

        # Build rollout prompt
        conversation_rollout_prompt = build_rollout_prompt(
            cfg.behavior_name, behavior_understanding, scientific_motivation,
            transcript_analyses, variation_description, rollout_cfg.max_turns,
            prompts_yaml
        )

        target_sysprompt_prefix = _get_override(prompts_yaml, "target_sysprompt_prefix")
        target_kickoff_prefix = _get_override(prompts_yaml, "target_kickoff_prefix")
        generate_kickoff_additional = _get_override(prompts_yaml, "generate_kickoff_additional")

        # Setup and run orchestrator
        loop = asyncio.get_event_loop()
        agent = await loop.run_in_executor(
            None,
            lambda: ConversationOrchestrator.setup(
                evaluator_model_id=evaluator_model_id,
                target_model_id=target_model_id,
                evaluator_system_prompt=evaluator_system_prompt,
                conversation_rollout_prompt=conversation_rollout_prompt,
                target_system_prompt=cfg.get("target_system_prompt", ""),
                target_sysprompt_prefix=target_sysprompt_prefix,
                max_turns=rollout_cfg.max_turns,
                evaluator_reasoning_effort=_effort(cfg.rollout.get("evaluator_thinking", False)),
                target_reasoning_effort=_effort(cfg.rollout.get("target_thinking", False)),
                evaluator_max_tokens=rollout_cfg.get("evaluator_max_tokens", rollout_cfg.get("max_tokens", 4000)),
                target_max_tokens=rollout_cfg.get("target_max_tokens", rollout_cfg.get("max_tokens", 4000)),
                evaluator_temperature=cfg.get("temperature", DEFAULT_TEMPERATURE_DETERMINISTIC),
                target_temperature=cfg.get("temperature", DEFAULT_TEMPERATURE_DETERMINISTIC),
                target_kickoff_prefix=target_kickoff_prefix,
                generate_kickoff_additional=generate_kickoff_additional,
                target_before_input=cfg.rollout.get("target_before_input", False),
                between_turns_strategise=cfg.rollout.get("between_turns_strategise", True),
                include_trs=bool((cfg.get("input_search", {}) or {}).get("enabled", False)),
            ),
        )

        transcript_data = await loop.run_in_executor(None, agent.run)

        # Add variation metadata
        transcript_data["metadata"]["variation_number"] = variation_number
        transcript_data["metadata"]["repetition_number"] = repetition_number

        # Save transcript file
        transcripts_dir = output_dir / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        filename = f"transcript_v{variation_number}r{repetition_number}.json"
        save_json(transcript_data, transcripts_dir / filename)

        print(f"  Rollout v{variation_number}r{repetition_number} completed", flush=True)

        return {
            "variation_number": variation_number,
            "variation_description": variation_description,
            "repetition_number": repetition_number,
            "num_turns": len(transcript_data.get("messages", [])),
            "transcript_file": filename,
        }


async def run_refinement(
    variation_number: int,
    round_history: List[Dict],  # list of {round_num, system_prompt, conversation, score, summary, justification}
    cfg: DotDict, prompts_yaml: Dict,
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
) -> Dict[str, Any]:
    """Refine a single scenario using the full history of previous rounds."""
    system, user = build_refine_prompt(cfg.behavior_name, round_history, prompts_yaml)

    refine_cfg = cfg.get("refinement", {}) or {}
    model_id = refine_cfg.get("model") or cfg.judgment.get("model")
    max_tokens = refine_cfg.get("max_tokens", cfg.judgment.get("max_tokens", 4000))
    reasoning_effort = _effort(refine_cfg.get("thinking", False))
    temperature = cfg.get("temperature", 1.0)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        executor,
        lambda: litellm_chat(
            model_id=model_id, messages=[{"role": "user", "content": user}],
            system_prompt=system, max_tokens=max_tokens,
            reasoning_effort=reasoning_effort, temperature=temperature,
        ),
    )
    parsed = parse_message(response)
    content = parsed["content"] or ""
    content = _auto_close_tags(content, ["observations", "updated_strategy"])

    obs_match = re.search(r"<observations>(.*?)</observations>", content, re.DOTALL)
    strat_match = re.search(r"<updated_strategy>(.*?)</updated_strategy>", content, re.DOTALL)

    observations = obs_match.group(1).strip() if obs_match else ""
    updated_strategy = strat_match.group(1).strip() if strat_match else content

    latest_score = round_history[-1].get("score", 0) if round_history else 0
    return {
        "variation_number": variation_number,
        "observations": observations,
        "updated_strategy": updated_strategy,
        "refinement_response": content,
        "previous_score": latest_score,
    }


async def run_rollout(cfg: DotDict, prompts_yaml: Dict, output_dir: Path,
                      understanding_results: Dict, ideation_results: Dict,
                      variations_override: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Run all rollouts. Dispatches to the batched-local implementation whenever the TARGET
    is local — that path owns corruption / token-level search and (via ApiModel) also supports
    a hosted-API evaluator. The pure-async orchestrator below is used only when the target
    itself is a hosted API model (no corruption)."""
    # Dispatch on the TARGET (not evaluator): corruption/search need the target's logits.
    # The evaluator may still be a hosted API model via ApiModel.
    if cfg.rollout.target.startswith("local/"):
        return run_rollout_batched_local(
            cfg, prompts_yaml, output_dir,
            understanding_results, ideation_results, variations_override,
        )

    print("\n" + "=" * 60, flush=True)
    print("ROLLOUT STAGE - STARTED", flush=True)
    print("=" * 60, flush=True)

    behavior_name = cfg.behavior_name
    behavior_understanding = understanding_results["understanding"]
    scientific_motivation = understanding_results["scientific_motivation"]
    transcript_analyses = understanding_results.get("transcript_analyses", [])
    variations = variations_override if variations_override is not None else ideation_results["variations"]
    num_reps = cfg.rollout.get("num_reps", 1)

    # Build evaluator system prompt
    evaluator_system_prompt = build_rollout_system(behavior_name, prompts_yaml)

    max_concurrent = cfg.get("max_concurrent", 15)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for var_idx, variation in enumerate(variations, 1):
        var_description = variation.get("description", str(variation)) if isinstance(variation, dict) else str(variation)
        for rep in range(1, num_reps + 1):
            print(f"  Launching rollout v{var_idx}r{rep}...", flush=True)
            tasks.append(
                run_single_rollout(
                    var_idx, var_description, rep, cfg, prompts_yaml,
                    behavior_understanding, scientific_motivation, transcript_analyses,
                    evaluator_system_prompt, output_dir, semaphore,
                )
            )

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    rollouts = []
    failed = []
    for i, result in enumerate(results_list):
        if isinstance(result, Exception):
            print(f"  Rollout failed: {type(result).__name__}: {result}", flush=True)
            failed.append({"error": str(result), "error_type": type(result).__name__})
        else:
            rollouts.append(result)

    rollouts.sort(key=lambda x: (x["variation_number"], x["repetition_number"]))

    rollout_results = {
        "metadata": {
            "evaluator": cfg.rollout.model,
            "target": cfg.rollout.target,
            "max_turns": cfg.rollout.max_turns,
        },
        "rollouts": rollouts,
        "successful_count": len(rollouts),
        "failed_count": len(failed),
        "total_count": len(tasks),
    }
    save_json(rollout_results, output_dir / "rollout.json")

    print(f"ROLLOUT STAGE - COMPLETED ({len(rollouts)} successful, {len(failed)} failed)", flush=True)
    return rollout_results


def run_rollout_batched_local(
    cfg: DotDict,
    prompts_yaml: Dict,
    output_dir: Path,
    understanding_results: Dict,
    ideation_results: Dict,
    variations_override: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Input-search rollout — processes one variation at a time (serial).
    For each variation: setup → kickoff input search → run top suffixes_per_scenario
    candidates as separate transcript reps → (optionally) input search for subsequent turns.
    Saves beast_pool.json with the full search pool for every variation.
    """
    print("\n" + "=" * 60, flush=True)
    print("ROLLOUT STAGE - STARTED (input search)", flush=True)
    print("=" * 60, flush=True)

    behavior_name = cfg.behavior_name
    behavior_understanding = understanding_results["understanding"]
    scientific_motivation = understanding_results["scientific_motivation"]
    transcript_analyses = understanding_results.get("transcript_analyses", [])
    variations = variations_override if variations_override is not None else ideation_results["variations"]
    max_turns    = cfg.rollout.get("max_turns", 5)
    # Split eval vs target token budgets — evaluator only needs to emit a short <message>
    # + <targeted_response_start> block, while the target produces the actual response.
    # Falls back to legacy unified `max_tokens` if the split keys aren't set.
    _legacy_mt   = cfg.rollout.get("max_tokens", 4000)
    eval_max_tokens   = cfg.rollout.get("evaluator_max_tokens", _legacy_mt)
    target_max_tokens = cfg.rollout.get("target_max_tokens", _legacy_mt)
    temperature  = cfg.get("temperature", DEFAULT_TEMPERATURE)
    no_think_eval   = not cfg.rollout.get("evaluator_thinking", False)
    no_think_target = not cfg.rollout.get("target_thinking", False)

    evaluator_model_id = cfg.rollout.model
    target_model_id    = cfg.rollout.target

    # Each LocalModel gets its own subprocess pinned to a specific GPU (CUDA_VISIBLE_DEVICES
    # is set per-worker before vLLM init). With one LLM per GPU each worker can use most of
    # the device's memory. Defaults: eval → GPU 0, target → GPU 1. Override via cfg.rollout.
    eval_gpu_id     = int(os.environ.get("BLOOM_EVAL_GPU", cfg.get("evaluator_gpu_id", 0)))
    target_gpu_id   = int(os.environ.get("BLOOM_TARGET_GPU", cfg.get("target_gpu_id", 1)))
    eval_gpu_util   = float(os.environ.get("BLOOM_EVAL_UTIL", cfg.get("evaluator_gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTIL)))
    target_gpu_util = cfg.get("target_gpu_memory_utilization",    DEFAULT_GPU_MEMORY_UTIL)
    eval_max_len    = core._DEFAULT_MAX_MODEL_LEN   # cfg.rollout.evaluator_max_model_len (BLOOM_EVAL_MAXLEN overrides), set at pipeline start
    target_max_len  = cfg.rollout.get("target_max_model_len",    4096)

    # Jailbroken-model contrastive output decoding: when enabled, the jail model
    # shares the target GPU. Auto-halve target_gpu_util so both vLLM workers fit.
    # Jail is loaded if any of:
    #   - `enabled` (jail steers the sampling proposal during target generation)
    #   - `output_search.jail_search_loss` AND output_search.enabled (BoN reward signal)
    #   - `input_search.jail_search_loss`  AND input_search.enabled  (TRS reward signal for BEAST)
    jail_cfg = cfg.get("jailbroken_output", {}) or {}
    output_cfg_peek = cfg.get("output_search", {}) or {}
    input_cfg_peek  = cfg.get("input_search", {}) or {}
    jail_use_rollout = bool(jail_cfg.get("enabled", False))
    jail_use_out_loss = bool(output_cfg_peek.get("jail_search_loss", False)) and bool(output_cfg_peek.get("enabled", False))
    jail_use_in_loss  = bool(input_cfg_peek.get("jail_search_loss",  False)) and bool(input_cfg_peek.get("enabled",  False))
    jail_on          = jail_use_rollout or jail_use_out_loss or jail_use_in_loss
    # Two jail engines, mirroring corruption: vllm_topk (legacy top-K logit_bias,
    # the current default) and hf_full (exact full-vocab PoE via HF, basic rollout
    # PoE only — no search-loss integration).
    jail_engine = str(jail_cfg.get("engine", "vllm_topk"))
    if jail_on and jail_engine not in ("vllm_topk", "hf_full"):
        raise RuntimeError(
            f"jailbroken_output.engine={jail_engine!r} not supported (vllm_topk | hf_full)")
    if jail_engine == "hf_full" and (jail_use_out_loss or jail_use_in_loss):
        raise RuntimeError(
            "jailbroken_output.engine=hf_full supports enabled (basic PoE) only, "
            "not output_search.jail_search_loss / input_search.jail_search_loss")
    jail_vllm = jail_on and jail_engine == "vllm_topk"
    jail_hf   = jail_on and jail_engine == "hf_full"

    # Corruption-model rewrite PoE. Standalone in v1 (mutually exclusive with jail and
    # all input/output/io search). Two engines:
    #   hf_full (default) — exact full-vocab PoE; target+corruption load as HF models in
    #                       THIS process on the target GPU (no vLLM target worker).
    #   vllm_topk         — legacy top-K logit_bias (approximate; see verify_poe_topk.py).

    # Target ENGINE policy: the target ALWAYS runs as an in-process HF model (vLLM is reserved for
    # the auditor). A no-search, no-jail run (plain baseline/BoN) is realised as a beta=0 jail run
    # in target_only mode (plain HF sampling, no jail model stepped), reusing the HF-target
    # machinery below. The input/output/jail search baselines still need the vLLM target worker, so
    # they keep it. Opt back into a vLLM target with BLOOM_TARGET_ENGINE=vllm.
    _no_search_jail = not (jail_on
                           or bool((cfg.get("input_search", {})  or {}).get("enabled", False))
                           or bool((cfg.get("output_search", {}) or {}).get("enabled", False)))
    _force_hf_target = os.environ.get("BLOOM_TARGET_ENGINE", cfg.rollout.get("target_engine", "hf")).lower() != "vllm"
    if _no_search_jail and _force_hf_target:
        # Vanilla/BoN baseline (beta=0 WILT): route through the jail hf_full path in target_only
        # mode — plain target sampling, no jail model stepped, no naturalness floor. Reuses the
        # jail path's free on-policy token-prob capture; the proposal model mirrors the target and
        # is never stepped.
        jail_cfg = {"model": target_model_id, "enabled": True, "engine": "hf_full",
                    "b2": 0.0, "b1": 1.0, "target_floor": 0.0, "target_only": True}
        jail_use_rollout = True; jail_on = True; jail_engine = "hf_full"
        jail_vllm = False; jail_hf = True

    # vllm_topk jail proposals share the target GPU (halve its util so both fit). hf_full
    # corruption loads HF models in-process instead (no vLLM target worker).
    if jail_vllm:
        target_gpu_util = target_gpu_util / 2.0
        roles = []
        if jail_use_rollout:  roles.append("sampling")
        if jail_use_out_loss: roles.append("output_score")
        if jail_use_in_loss:  roles.append("input_score")
        print(f"  [jailbroken_output] enabled for {'+'.join(roles)} — halving target_gpu_memory_utilization to {target_gpu_util:.3f}", flush=True)

    # The evaluator may be a local vLLM model (cfg.rollout.model = "local/...") or a hosted
    # API model (e.g. "claude-haiku-4-5") routed through litellm. An API evaluator can only
    # SAMPLE whole turns — it has no logits/tokenizer — so every token-level search must be
    # off. Understanding/ideation/judgment are independent (they take a model id, not lm_eval).
    eval_is_local = evaluator_model_id.startswith("local/")
    if not eval_is_local:
        _api_eval_conflicts = []
        if bool((cfg.get("input_search", {})  or {}).get("enabled", False)): _api_eval_conflicts.append("input_search")
        if bool((cfg.get("output_search", {}) or {}).get("enabled", False)): _api_eval_conflicts.append("output_search")
        if _api_eval_conflicts:
            raise RuntimeError(
                f"rollout.model={evaluator_model_id!r} is a hosted API model (non-'local/'). It can only "
                f"SAMPLE evaluator turns (no logits/tokenizer for token-level search). Disable: "
                + ", ".join(_api_eval_conflicts))

    if eval_is_local:
        lm_eval = _get_local_model(evaluator_model_id[len("local/"):],
                                   gpu_id=eval_gpu_id,
                                   gpu_memory_utilization=eval_gpu_util,
                                   max_model_len=eval_max_len)
    else:
        lm_eval = ApiModel(
            evaluator_model_id,
            max_tokens=eval_max_tokens,
            reasoning_effort=_effort(cfg.rollout.get("evaluator_thinking", False)),
        )
        print(f"  [evaluator] hosted API model {evaluator_model_id!r} via litellm "
              f"(reasoning_effort={lm_eval.reasoning_effort!r}) — token-level search disabled, "
              f"sampling evaluator turns only", flush=True)
    # In hf_full corruption mode the target runs as an HF model (loaded below); no
    # vLLM target worker is spawned.
    lm_target = None
    if not jail_hf:
        lm_target = _get_local_model(target_model_id[len("local/"):],
                                     gpu_id=target_gpu_id,
                                     gpu_memory_utilization=target_gpu_util,
                                     max_model_len=target_max_len)

    lm_jail = None
    jail_runtime_cfg: Optional[Dict] = None
    if jail_on:
        jail_model_id = (jail_cfg.get("model", "") or "").strip()
        if jail_model_id in ("", "self", "local/self"):
            jail_model_id = target_model_id      # self-jail: proposal model = the target model
        if not jail_model_id.startswith("local/"):
            raise RuntimeError(
                f"jailbroken_output.model must start with 'local/' (or be 'self'/'' for self-jail), got {jail_model_id!r}"
            )
        jail_system_prompt = prompts_yaml.get("jailbroken_output_system_prompt", "")
        if not jail_system_prompt and not bool(jail_cfg.get("target_only")):
            raise RuntimeError(
                "jailbroken_output.enabled=True requires "
                "'jailbroken_output_system_prompt' in prompts.yaml"
            )
        # tokbias baseline: numeric knobs from cfg.tokbias_output; prompt content from the
        # behaviour yaml (tokbias_output_prompt / _neg_prompt / _words), cfg value taking
        # precedence if set. BLOOM_TOKBIAS_* still overrides inside _tokbias_vector.
        _tb = dict(cfg.get("tokbias_output", {}) or {})
        for _tk, _yk in (("prompt", "tokbias_output_prompt"),
                         ("neg_prompt", "tokbias_output_neg_prompt"),
                         ("words", "tokbias_output_words")):
            _tb[_tk] = _tb.get(_tk) or prompts_yaml.get(_yk, "") or ""
        jail_runtime_cfg = {
            "target_only":        bool(jail_cfg.get("target_only", False)),  # beta=0 vanilla/BoN fast path (plain target sampling, no jail model stepped)
            "engine":             jail_engine,
            "enabled":            jail_use_rollout,
            "jail_out_loss":      jail_use_out_loss,
            "jail_in_loss":       jail_use_in_loss,
            "system_prompt": jail_system_prompt,
            "prefill":     (prompts_yaml.get("jailbroken_output_prefill", "") or "") if jail_cfg.get("prefill", True) else "",  # True -> behaviour-file prefill; False -> none
            "b2":          float(jail_cfg.get("b2", 2.0)),
            "b1":          (float(jail_cfg["b1"]) if jail_cfg.get("b1") is not None else None),  # None=legacy target+beta*jail; 0=floor-only jail
            "target_floor": float(jail_cfg.get("target_floor", 1e-4)),  # naturalness floor ON by default (0 only via explicit no-floor ablation)
            "b3":              float(jail_cfg.get("b3", 0.0) or 0.0),          # negative-steering weight; 0=off (legacy jail)
            # neg prompts: cfg (jailbroken_output.neg_*) takes precedence, else the behaviour yaml.
            "neg_system_prompt": (jail_cfg.get("neg_system_prompt") or prompts_yaml.get("jailbroken_output_neg_system_prompt", "") or ""), # input-conditioned negative persona
            "neg_user_prompt":   (jail_cfg.get("neg_user_prompt")   or prompts_yaml.get("jailbroken_output_neg_user_prompt", "")   or ""), # harmful user turn -> ELICITED refusal direction
            "neg_prefill":       (jail_cfg.get("neg_prefill")       or prompts_yaml.get("jailbroken_output_neg_prefill", "")       or ""),
            "top_k_logprobs": (int(jail_cfg["top_k_logprobs"]) if jail_cfg.get("top_k_logprobs") is not None else None),  # vllm_topk only; None on the default hf_full path
            "tokbias":      _tb,  # static logit-bias baseline: cfg.tokbias_output knobs + yaml prompt content; BLOOM_TOKBIAS_* override
        }
        if jail_vllm:
            lm_jail = _get_local_model(jail_model_id[len("local/"):],
                                       gpu_id=target_gpu_id,
                                       gpu_memory_utilization=target_gpu_util,
                                       max_model_len=target_max_len)
        else:  # hf_full — load HF target + jail in-process on the target GPU
            jail_runtime_cfg["hf"] = _load_hf_poe_models(
                target_model_id[len("local/"):], jail_model_id[len("local/"):], target_gpu_id)
        print(f"  [jailbroken_output] engine={jail_engine} loaded {jail_model_id} on GPU {target_gpu_id} "
              f"(b2={jail_runtime_cfg['b2']})", flush=True)

    search_cfg = cfg.input_search
    suffixes_per_scenario = search_cfg.suffixes_per_scenario


    # Output search (optional): regenerate target responses to maximise
    # log P("Yes") on a behavior-presence judge prompt. When disabled, the natural
    # target response is used as-is.
    output_cfg       = cfg.output_search
    output_search_on = output_cfg.enabled
    output_judge_template = ""
    output_behavior_name  = ""
    if output_search_on:
        output_judge_template = prompts_yaml.get("output_search_judge_prompt", "")
        if not output_judge_template:
            raise RuntimeError(
                "output_search.enabled=True but 'output_search_judge_prompt' is missing "
                "from prompts.yaml"
            )
        output_behavior_name = output_cfg.behavior_name
        if not output_behavior_name:
            raise RuntimeError(
                "output_search.enabled=True requires output_search.behavior_name to be set"
            )
        # Single-token assertion on "Yes" — fails loudly if the judge tokenizer
        # splits it. Run once at startup so any failure surfaces before rollouts.
        _resolve_yes_token_id(lm_eval)

    evaluator_system_prompt  = build_rollout_system(behavior_name, prompts_yaml)
    target_sysprompt_prefix  = _get_override(prompts_yaml, "target_sysprompt_prefix")
    target_kickoff_prefix    = _get_override(prompts_yaml, "target_kickoff_prefix")
    generate_kickoff_additional = _get_override(prompts_yaml, "generate_kickoff_additional")

    between_turns_strategise = cfg.rollout.get("between_turns_strategise", True)
    target_before_input      = cfg.rollout.get("target_before_input", False)
    history_turns = cfg.rollout.get("history_turns", None)  # None = full history
    input_search_on = bool(search_cfg.enabled)  # only ask the eval for a TRS when input_search will use it
    # Batched-jail eligibility: the clean jail hf_full case (no per-variation token
    # search) can roll variations out in LOCKSTEP like corruption instead of one-at-a-
    # time. Defined here (after input/output_search_on are known); consumed below to
    # gate the serial fallback.
    jail_batched = (jail_hf and jail_use_rollout
                    and not input_search_on and not output_search_on)
    if cfg.rollout.get("max_turns", 1) <= 1:
        between_turns_strategise = False  # no subsequent turns to strategise for

    def _build_kickoff_prompt(refined_strategy: str = "") -> str:
        """Build the per-variation kickoff prompt via the shared _kickoff_message helper.
        The <strategy> block only appears on rounds 2+ (a refined strategy to inject)."""
        return _kickoff_message(generate_kickoff_additional, target_before_input,
                                input_search_on, refined_strategy)

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    rollouts: List[Dict] = []
    beast_pool_data: List[Dict] = []   # one entry per variation, saved to beast_pool.json
    output_pool_data: List[Dict] = []  # one entry per (variation, turn), saved to output_pool.json
    batch_size = cfg.get("batch_size", 4)
    target_batch_size = cfg.get("target_batch_size", batch_size)  # used only for input-search target scoring; eval-side batching is handled by vLLM internally

    # ── Build per-variation rollout prompts (no setup-generation pass) ────────────
    # The per-variation setup LLM call has been removed: the target system prompt
    # is now a fixed config value (cfg.target_system_prompt), and the scenario
    # context is delivered to the evaluator merged into the first user turn alongside
    # the kickoff prompt. setup_content stays as an unused field (always "") for
    # back-compat with downstream code that reads it from transcript metadata.
    var_descs: List[str] = []
    rollout_prompt_texts: List[str] = []

    # Resume: figure out which variations are already complete on disk so we can skip them
    # entirely (no input search, no rollout). A variation counts as complete if all of its expected
    # reps (transcript_v{var_idx}r{1..suffixes_per_scenario}.json) exist.
    def _variation_done(var_idx_1based: int) -> bool:
        return all(
            (transcripts_dir / f"transcript_v{var_idx_1based}r{rep}.json").exists()
            for rep in range(1, suffixes_per_scenario + 1)
        )

    for var_idx_0based, variation in enumerate(variations):
        var_idx = var_idx_0based + 1
        vd = variation.get("description", str(variation)) if isinstance(variation, dict) else str(variation)
        var_descs.append(vd)
        if _variation_done(var_idx):
            rollout_prompt_texts.append("")
            continue
        # Drop scientific_motivation in round 2+ when a refined_strategy is doing the
        # heavy lifting — it duplicates the high-level framing already covered by
        # behavior_understanding and the strategy injected via the kickoff.
        has_refined_strategy = bool(
            variation.get("refined_strategy", "") if isinstance(variation, dict) else ""
        )
        rp = build_rollout_prompt(
            behavior_name, behavior_understanding, scientific_motivation,
            transcript_analyses, vd, max_turns, prompts_yaml,
            skip_motivation=has_refined_strategy,
        )
        rollout_prompt_texts.append(rp)

    n_skipped = sum(1 for v in range(1, len(variations) + 1) if _variation_done(v))
    if n_skipped:
        print(f"  Resume: {n_skipped}/{len(variations)} variations already have transcripts — skipping", flush=True)
    print(f"  Setup-generation pass disabled — using fixed target_system_prompt from config", flush=True)
    setup_contents: List[str] = [""] * len(variations)

    def _resume_load_variation(var_idx: int, var_desc: str) -> None:
        """Load an already-complete variation's transcripts into `rollouts` (resume path)."""
        for rep in range(1, suffixes_per_scenario + 1):
            tf_path = transcripts_dir / f"transcript_v{var_idx}r{rep}.json"
            try:
                with open(tf_path, "r", encoding="utf-8") as f:
                    td = json.load(f)
            except Exception as e:
                print(f"    Could not read existing {tf_path.name}: {e} — will be missing from rollout summary", flush=True)
                continue
            turn_lps = [m["targeted_response_start_logprob"] for m in td.get("messages", [])
                        if m.get("targeted_response_start_logprob") is not None]
            avg_lp = round(sum(turn_lps) / len(turn_lps), 4) if turn_lps else None
            entry: Dict[str, Any] = {
                "variation_number": var_idx, "variation_description": var_desc,
                "repetition_number": rep, "num_turns": len(td.get("messages", [])),
                "transcript_file": tf_path.name,
            }
            if avg_lp is not None:
                entry["avg_logprob"] = avg_lp
            rollouts.append(entry)

    if jail_batched:
        # ══ Batched JAIL rollout (variations in LOCKSTEP) ════════════════════
        # Mirror of the corruption batched path for the jail hf_full PoE decode. The
        # gate guarantees no input/output token-search, so each turn is just: sample an
        # evaluator message (vLLM, batched natively) then generate the target reply via
        # jail PoE. _jail_generate_hf batches the whole active chunk in ONE call per turn
        # (B slots), far more GPU-efficient than the per-variation serial path below.
        jail_var_batch = max(1, int(os.environ.get("BLOOM_JAIL_VAR_BATCH",
                                    (cfg.get("jailbroken_output", {}) or {}).get("var_batch", 12))))
        _jail_hf = jail_runtime_cfg["hf"]

        # One "seed" per transcript (variation x rep), honoring resume/skip. freeze_input
        # rounds 2+ replicate a variation into n_reps seeds that share the frozen kickoff
        # but resample the target output independently.
        seeds = []
        for var_idx, (variation, var_desc, rollout_prompt_text, _sc) in enumerate(
            zip(variations, var_descs, rollout_prompt_texts, setup_contents), 1
        ):
            if isinstance(variation, dict) and variation.get("skip_rollout"):
                print(f"  Variation {var_idx}/{len(variations)}: skipped (already finished)", flush=True)
                continue
            if _variation_done(var_idx):
                print(f"  Variation {var_idx}/{len(variations)}: skipped (transcripts exist)", flush=True)
                _resume_load_variation(var_idx, var_desc)
                continue
            fixed_kickoff = variation.get("fixed_kickoff") if isinstance(variation, dict) else None
            frozen = bool(fixed_kickoff and fixed_kickoff.get("content"))
            n_reps = max(1, int(variation.get("n_reps", 1))) if (frozen and isinstance(variation, dict)) else 1
            for rep in range(1, n_reps + 1):
                seeds.append({"var_idx": var_idx, "variation": variation, "var_desc": var_desc,
                              "rollout_prompt_text": rollout_prompt_text, "rep": rep,
                              "frozen": frozen, "fixed_kickoff": fixed_kickoff})

        # ── Kickoff for ALL seeds up front in ONE batched evaluator call ──
        gen_seeds, gen_ctxs = [], []
        for sd in seeds:
            variation = sd["variation"]
            frozen_tsp = variation.get("target_system_prompt", "") if isinstance(variation, dict) else ""
            target_sysprompt = frozen_tsp or cfg.get("target_system_prompt", "")
            if target_sysprompt_prefix and target_sysprompt_prefix.strip() and target_sysprompt and not frozen_tsp:
                target_sysprompt = f"{target_sysprompt_prefix.strip()}\n\n{target_sysprompt}"
            per_var_refined_strategy = variation.get("refined_strategy", "") if isinstance(variation, dict) else ""
            kickoff_prompt = _build_kickoff_prompt(refined_strategy=per_var_refined_strategy)
            eval_ctx = [
                {"role": "system", "content": evaluator_system_prompt},
                {"role": "user", "content": f"{sd['rollout_prompt_text']}\n\n{kickoff_prompt}"},
            ]
            sd["target_sysprompt"] = target_sysprompt
            sd["eval_kickoff_ctx"] = eval_ctx
            sd["setup_ctx_len"] = len(eval_ctx)
            sd["kickoff_strategy"] = ""
            if sd["frozen"]:
                sd["kickoff_msg"] = sd["fixed_kickoff"]["content"]
                sd["kickoff_strategy"] = sd["fixed_kickoff"].get("strategy", "") or ""
            else:
                gen_seeds.append(sd); gen_ctxs.append(_strip_thinking_from_msgs(eval_ctx))
        if gen_ctxs:
            raws = batch_generate_local(lm_eval, gen_ctxs, eval_max_tokens, temperature, no_think=no_think_eval)
            for sd, raw in zip(gen_seeds, raws):
                parsed = parse_message(_make_local_response(raw))
                content = parsed["content"] or raw
                msg, _trs, strat = _extract_message_tags(content)
                sd["kickoff_msg"] = msg
                sd["kickoff_strategy"] = strat

        # ── Roll out mini-batches (chunks) in lockstep — ONE jail PoE call per turn ──
        for _b in range(0, len(seeds), jail_var_batch):
            chunk = seeds[_b:_b + jail_var_batch]
            for sd in chunk:
                tsp = sd["target_sysprompt"]
                tmsgs, trmsgs = [], []
                if tsp:
                    tmsgs.append({"role": "system", "content": tsp})
                    trmsgs.append({"role": "system", "content": tsp, "source": "target_system"})
                kmsg = sd["kickoff_msg"]
                target_content = kmsg
                if target_kickoff_prefix and not sd["frozen"]:
                    target_content = target_kickoff_prefix.strip() + " " + kmsg
                tmsgs.append({"role": "user", "content": target_content})
                kick_entry = {"role": "user", "content": target_content, "source": "evaluator"}
                if sd["kickoff_strategy"]:
                    kick_entry["strategy"] = sd["kickoff_strategy"]
                trmsgs.append(kick_entry)
                sd["target_msgs"] = tmsgs
                sd["transcript_msgs"] = trmsgs
                sd["eval_msgs"] = list(sd["eval_kickoff_ctx"]) + [{"role": "assistant", "content": kmsg}]
                sd["current_turn"] = 0
                sd["done"] = False
                beast_pool_data.append({
                    "variation_number": sd["var_idx"], "turn": "kickoff", "trs": "",
                    "pool": [{"baseline": kmsg, "suffix": "", "message": kmsg, "score": None}],
                })

            for turn in range(max_turns):
                active = [sd for sd in chunk if not sd["done"]]
                if not active:
                    break
                jail_results = _jail_generate_hf(
                    _jail_hf, jail_runtime_cfg,
                    [sd["target_msgs"] for sd in active], target_max_tokens, temperature, no_think_target,
                )
                for sd, _jr in zip(active, jail_results):
                    raw_target = _jr["best_text"]
                    _ids = _jr.get("best_ids")
                    _tprobs = _jr.get("best_token_probs")
                    parsed_t = parse_message(_make_local_response(raw_target))
                    target_resp = parsed_t["content"] or raw_target
                    target_reason = parsed_t["reasoning"]
                    sd["target_msgs"].append({"role": "assistant", "content": target_resp})
                    sd["current_turn"] = turn + 1
                    tmsg = {"role": "assistant", "content": target_resp, "source": "target"}
                    if _ids and target_resp == raw_target:
                        tmsg["gen_token_ids"] = _ids           # exact ids -> free token stats
                    if _tprobs and target_resp == raw_target:
                        tmsg["gen_token_probs"] = _tprobs      # on-policy probs (plausibility)
                        tmsg["prob_stats"] = _prob_summary(_tprobs)
                    if target_reason:
                        tmsg["reasoning"] = target_reason
                    sd["transcript_msgs"].append(tmsg)
                    if sd["current_turn"] >= max_turns:
                        sd["done"] = True

                cont = [sd for sd in chunk if not sd["done"]]
                if not cont:
                    break
                gen_ctxs = []
                for sd in cont:
                    last = sd["transcript_msgs"][-1]
                    followup = _build_evaluator_followup(
                        last["content"], last.get("reasoning"), sd["current_turn"], max_turns,
                        between_turns_strategise=between_turns_strategise,
                        partial_history=history_turns is not None,
                        target_before_input=target_before_input,
                        include_trs=input_search_on,
                    )
                    eval_msgs_turn = list(sd["eval_msgs"]) + [{"role": "user", "content": followup}]
                    sd["_eval_msgs_turn"] = eval_msgs_turn
                    gen_ctxs.append(_strip_thinking_from_msgs(
                        _truncate_eval_history(eval_msgs_turn, sd["setup_ctx_len"], history_turns)))
                raws = batch_generate_local(lm_eval, gen_ctxs, eval_max_tokens, temperature, no_think=no_think_eval)
                for sd, raw in zip(cont, raws):
                    parsed = parse_message(_make_local_response(raw))
                    content = parsed["content"] or raw
                    next_msg, _trs, strat = _extract_message_tags(content)
                    if "<END>" in next_msg:
                        sd["done"] = True
                        continue
                    sd["eval_msgs"] = sd["_eval_msgs_turn"] + [{"role": "assistant", "content": next_msg}]
                    sd["target_msgs"].append({"role": "user", "content": next_msg})
                    turn_entry = {"role": "user", "content": next_msg, "source": "evaluator"}
                    if strat:
                        turn_entry["strategy"] = strat
                    sd["transcript_msgs"].append(turn_entry)

            # ── Save transcripts for this chunk ──
            for sd in chunk:
                var_idx, rep = sd["var_idx"], sd["rep"]
                transcript_data = {
                    "metadata": {
                        "evaluator_model": evaluator_model_id,
                        "target_model": target_model_id,
                        "target_system_prompt": sd["target_sysprompt"],
                        "setup_content": "",
                        "refined_strategy": sd["variation"].get("refined_strategy", "") if isinstance(sd["variation"], dict) else "",
                        "variation_number": var_idx,
                        "repetition_number": rep,
                        "created_at": datetime.now().isoformat(),
                    },
                    "messages": sd["transcript_msgs"],
                    "judgment": None,
                }
                filename = f"transcript_v{var_idx}r{rep}.json"
                save_json(transcript_data, transcripts_dir / filename)
                print(f"  Rollout v{var_idx}r{rep} done ({sd['current_turn']} turns) [batched-jail]", flush=True)
                rollouts.append({
                    "variation_number": var_idx, "variation_description": sd["var_desc"],
                    "repetition_number": rep, "num_turns": len(sd["transcript_msgs"]),
                    "transcript_file": filename, "kickoff_score": None,
                })

    # Non-corruption path (input/output search, jail, plain): one variation at a time.
    _noncorr_iter = [] if jail_batched else list(enumerate(
        zip(variations, var_descs, rollout_prompt_texts, setup_contents), 1))
    for var_idx, (variation, var_desc, rollout_prompt_text, setup_content) in _noncorr_iter:
        # skip_finished (option B): this scenario already reached finish_score in an earlier
        # round — don't roll it out again; it keeps its best earlier transcript.
        if isinstance(variation, dict) and variation.get("skip_rollout"):
            print(f"\n  Variation {var_idx}/{len(variations)}: skipped (already finished)", flush=True)
            continue
        # Resume: variation already has all its transcripts on disk → load and skip search.
        if _variation_done(var_idx):
            print(f"\n  Variation {var_idx}/{len(variations)}: skipped (transcripts exist)", flush=True)
            for rep in range(1, suffixes_per_scenario + 1):
                tf_path = transcripts_dir / f"transcript_v{var_idx}r{rep}.json"
                try:
                    with open(tf_path, "r", encoding="utf-8") as f:
                        td = json.load(f)
                except Exception as e:
                    print(f"    Could not read existing {tf_path.name}: {e} — will be missing from rollout summary", flush=True)
                    continue
                turn_lps = [
                    m["targeted_response_start_logprob"]
                    for m in td.get("messages", [])
                    if m.get("targeted_response_start_logprob") is not None
                ]
                avg_lp = round(sum(turn_lps) / len(turn_lps), 4) if turn_lps else None
                entry: Dict[str, Any] = {
                    "variation_number":      var_idx,
                    "variation_description": var_desc,
                    "repetition_number":     rep,
                    "num_turns":             len(td.get("messages", [])),
                    "transcript_file":       tf_path.name,
                }
                if avg_lp is not None:
                    entry["avg_logprob"] = avg_lp
                rollouts.append(entry)
            continue

        print(f"\n  Variation {var_idx}/{len(variations)}: input search ...", flush=True)

        # Resolve target system prompt:
        #   1. Frozen value from the variation override (round 2+ carries round-1's value).
        #   2. Fall back to cfg.target_system_prompt (the fixed default for this run).
        # Empty string is valid — target then runs with no system prompt at all.
        frozen_tsp = (
            variation.get("target_system_prompt", "") if isinstance(variation, dict) else ""
        )
        target_sysprompt = frozen_tsp or cfg.get("target_system_prompt", "")
        if target_sysprompt_prefix and target_sysprompt_prefix.strip() and target_sysprompt and not frozen_tsp:
            # Don't double-prefix when reusing an already-prefixed sysprompt from round 1.
            target_sysprompt = f"{target_sysprompt_prefix.strip()}\n\n{target_sysprompt}"

        # Base message lists (shared across reps for this variation)
        target_msgs_base: List[Dict] = []
        transcript_msgs_base: List[Dict] = []
        if target_sysprompt:
            target_msgs_base.append({"role": "system", "content": target_sysprompt})
            transcript_msgs_base.append({"role": "system", "content": target_sysprompt, "source": "target_system"})

        # Per-variation kickoff prompt: optionally injects refined_strategy from prior rounds
        # and respects the between_turns_strategise toggle.
        per_var_refined_strategy = (
            variation.get("refined_strategy", "") if isinstance(variation, dict) else ""
        )
        kickoff_prompt = _build_kickoff_prompt(refined_strategy=per_var_refined_strategy)

        # eval_msgs up to the kickoff request (before input search generates the response).
        # The setup-generation pass has been removed, so the scenario context
        # (rollout_prompt_text) and the kickoff instructions are merged into a single
        # user turn. This avoids two consecutive user messages that would otherwise
        # arise from dropping the assistant setup turn.
        eval_msgs_kickoff_ctx = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user",   "content": f"{rollout_prompt_text}\n\n{kickoff_prompt}"},
        ]


        # ── Kickoff input search ──────────────────────────────────────────
        # When a variation carries a fixed kickoff (supplied by the per-round kickoff bank),
        # skip the search entirely and use it verbatim — so only the target OUTPUT is
        # resampled this round. Its content is the already-final user message (any
        # target_kickoff_prefix was already applied in round 1), so don't re-prefix.
        fixed_kickoff = variation.get("fixed_kickoff") if isinstance(variation, dict) else None
        use_frozen_kickoff = bool(fixed_kickoff and fixed_kickoff.get("content"))
        # n_reps: how many independent resamples of this scenario to roll this round (option B
        # gives unfinished scenarios extra reps). Each rep reuses the same frozen kickoff.
        _n_reps = max(1, int(variation.get("n_reps", 1))) if isinstance(variation, dict) else 1
        if use_frozen_kickoff:
            kickoff_pool = [(fixed_kickoff["content"], None,
                             fixed_kickoff.get("baseline", "") or "",
                             fixed_kickoff.get("suffix", "") or "")] * _n_reps
            trs_kickoff = fixed_kickoff.get("trs", "") or ""
            kickoff_strategy = fixed_kickoff.get("strategy", "") or ""
            print(f"  v{var_idx}: fixed kickoff reused x{_n_reps} reps (kickoff bank)", flush=True)
        else:
            # Strip <thinking> blocks from past evaluator messages before passing
            # to search (cheap defense in depth even if parse_message already handled them).
            kickoff_pool, trs_kickoff, kickoff_strategy = input_search_evaluator_message(
                lm_eval, lm_target,
                _strip_thinking_from_msgs(eval_msgs_kickoff_ctx), target_msgs_base,
                search_cfg, no_think_eval, eval_max_tokens, temperature, target_batch_size,
                lm_jail=lm_jail, jail_runtime_cfg=jail_runtime_cfg,
            )
        setup_ctx_len = len(eval_msgs_kickoff_ctx)  # used for history truncation per turn

        # Save full pool for this variation
        beast_pool_data.append({
            "variation_number": var_idx,
            "turn": "kickoff",
            "trs": trs_kickoff,
            "pool": [
                {
                    "baseline": base,
                    "suffix":   suf,
                    "message":  msg,
                    "score":    round(sc, 4) if sc not in (None, -float("inf")) else None,
                }
                for msg, sc, base, suf in kickoff_pool
            ],
        })

        # Frozen kickoff: run all replicated reps (n_reps). Otherwise take the top
        # suffixes_per_scenario input-search candidates as reps.
        top_candidates = kickoff_pool if use_frozen_kickoff else kickoff_pool[:suffixes_per_scenario]
        print(f"  v{var_idx}: pool_size={len(kickoff_pool)}, running {len(top_candidates)} as reps", flush=True)

        # ── Run each top candidate as a separate transcript rep ──────────
        for rep, (kickoff_msg, kickoff_score, kickoff_baseline, kickoff_suffix) in enumerate(top_candidates, 1):
            # Per-rep state (copies so reps are independent)
            target_msgs    = list(target_msgs_base)
            transcript_msgs = list(transcript_msgs_base)
            # eval context after the kickoff message has been "sent"
            eval_msgs = list(eval_msgs_kickoff_ctx) + [{"role": "assistant", "content": kickoff_msg}]

            # Deliver kickoff to target
            target_content = kickoff_msg
            if target_kickoff_prefix and not use_frozen_kickoff:
                target_content = target_kickoff_prefix.strip() + " " + kickoff_msg
            target_msgs.append({"role": "user", "content": target_content})

            kick_entry: Dict[str, Any] = {
                "role": "user", "content": target_content, "source": "evaluator",
                "targeted_response_start": trs_kickoff,
                "beast_baseline": kickoff_baseline,
                "beast_suffix":   kickoff_suffix,
            }
            if kickoff_strategy:
                kick_entry["strategy"] = kickoff_strategy
            if kickoff_score not in (None, -float("inf")):
                kick_entry["targeted_response_start_logprob"] = round(kickoff_score, 4)
            transcript_msgs.append(kick_entry)

            current_turn = 0

            # ── Turn loop ─────────────────────────────────────────────────
            for turn in range(max_turns):
                _corr_gen_ids = None  # exact generated token-ids (corruption path) for accurate scoring
                _jail_tprobs = None   # on-policy per-token probs captured during jail hf_full decode
                # Target responds. When jailbroken contrastive sampling is on,
                # the target's natural reply is generated via PoE-weighted
                # sampling against the jailbroken model; otherwise the standard
                # vLLM generation path is used.
                if (jail_runtime_cfg is not None
                        and jail_runtime_cfg.get("enabled", False)):
                    if jail_runtime_cfg.get("engine") == "hf_full":
                        _jr = _jail_generate_hf(
                            jail_runtime_cfg["hf"], jail_runtime_cfg,
                            [target_msgs], target_max_tokens, temperature, no_think_target,
                        )[0]
                        raw_target    = _jr["best_text"]
                        _corr_gen_ids = _jr.get("best_ids")          # exact ids -> free token stats
                        _jail_tprobs  = _jr.get("best_token_probs")  # on-policy probs (captured at gen time)
                    else:
                        raw_target = batch_generate_contrastive_local(
                            lm_target, lm_jail, jail_runtime_cfg,
                            [target_msgs], target_max_tokens, temperature, no_think_target,
                        )[0]
                else:
                    raw_target = batch_generate_local(
                        lm_target, [target_msgs], target_max_tokens, temperature, no_think=no_think_target
                    )[0]
                parsed_target = parse_message(_make_local_response(raw_target))
                target_resp   = parsed_target["content"] or raw_target
                target_reason = parsed_target["reasoning"]

                # ── Output search (optional) ──────────────────────────────
                # Replace the natural target response with one that maximises
                # log P("Yes") under the judge prompt. baseline_response is the
                # natural target_resp; the searched output replaces it. Reasoning
                # blocks are dropped because the searched response is fresh content.
                output_search_baseline = ""
                output_search_suffix   = ""
                output_search_score: Optional[float] = None
                if output_search_on:
                    # The last user message in target_msgs is what the target is responding to.
                    user_input_for_judge = target_msgs[-1]["content"] if target_msgs and target_msgs[-1]["role"] == "user" else ""
                    out_pool = output_search_target_response(
                        lm_eval, lm_target,
                        list(target_msgs), user_input_for_judge, target_resp,
                        output_cfg, no_think_target, no_think_eval,
                        output_judge_template, output_behavior_name,
                        batch_size=target_batch_size,
                        lm_jail=lm_jail,
                        jail_runtime_cfg=jail_runtime_cfg,
                    )
                    output_pool_data.append({
                        "variation_index": var_idx,
                        "turn": turn + 1,
                        "pool": [
                            {"response": t, "score": s, "baseline": b, "suffix": sx}
                            for (t, s, b, sx) in out_pool
                        ],
                    })
                    searched_resp, output_search_score_val, ob, os_ = out_pool[0]
                    target_resp           = searched_resp
                    target_reason         = ""  # searched response has no reasoning trace
                    output_search_baseline = ob
                    output_search_suffix   = os_
                    output_search_score    = output_search_score_val

                target_msgs.append({"role": "assistant", "content": target_resp})
                current_turn = turn + 1

                tmsg: Dict[str, Any] = {"role": "assistant", "content": target_resp, "source": "target"}
                if _corr_gen_ids and target_resp == raw_target:
                    tmsg["gen_token_ids"] = _corr_gen_ids  # exact ids -> exact token-prob scoring (no re-encode)
                if _jail_tprobs and target_resp == raw_target:
                    tmsg["gen_token_probs"] = _jail_tprobs  # on-policy probs captured at jail gen time -> free token stats
                    tmsg["prob_stats"] = _prob_summary(_jail_tprobs)
                if target_reason:
                    tmsg["reasoning"] = target_reason
                if output_search_on:
                    tmsg["output_search_baseline"] = output_search_baseline
                    tmsg["output_search_suffix"]   = output_search_suffix
                    if output_search_score is not None and output_search_score != -float("inf"):
                        tmsg["output_search_score"] = round(output_search_score, 4)
                transcript_msgs.append(tmsg)

                if current_turn >= max_turns:
                    break

                # Evaluator follow-up context
                followup_prompt = _build_evaluator_followup(
                    target_resp, target_reason, current_turn, max_turns,
                    between_turns_strategise=between_turns_strategise,
                    partial_history=history_turns is not None,
                    target_before_input=target_before_input,
                    include_trs=input_search_on,
                )
                eval_msgs_turn  = list(eval_msgs) + [{"role": "user", "content": followup_prompt}]

                # Truncate evaluator history to last K turn pairs (target still sees full
                # context for genuine search scoring); strip <thinking> blocks for token savings.
                eval_msgs_for_search = _truncate_eval_history(
                    eval_msgs_turn, setup_ctx_len, history_turns,
                )
                eval_msgs_for_search = _strip_thinking_from_msgs(eval_msgs_for_search)

                # Input search for next evaluator message (keep top 1 — conversation committed)
                turn_pool, turn_trs, turn_strategy = input_search_evaluator_message(
                    lm_eval, lm_target,
                    eval_msgs_for_search, target_msgs,
                    search_cfg, no_think_eval, eval_max_tokens, temperature, target_batch_size,
                    lm_jail=lm_jail, jail_runtime_cfg=jail_runtime_cfg,
                )
                next_msg, next_score, next_baseline, next_suffix = turn_pool[0]

                if "<END>" in next_msg:
                    break

                # Advance eval context
                eval_msgs = eval_msgs_turn + [{"role": "assistant", "content": next_msg}]

                target_msgs.append({"role": "user", "content": next_msg})
                turn_entry: Dict[str, Any] = {
                    "role": "user", "content": next_msg, "source": "evaluator",
                    "targeted_response_start": turn_trs,
                    "beast_baseline": next_baseline,
                    "beast_suffix":   next_suffix,
                }
                if turn_strategy:
                    turn_entry["strategy"] = turn_strategy
                if next_score not in (None, -float("inf")):
                    turn_entry["targeted_response_start_logprob"] = round(next_score, 4)
                transcript_msgs.append(turn_entry)

            # ── Save transcript ────────────────────────────────────────────
            transcript_data = {
                "metadata": {
                    "evaluator_model": evaluator_model_id,
                    "target_model":    target_model_id,
                    "target_system_prompt": target_sysprompt,
                    # setup_content: retained (always "") for transcript-schema back-compat; setup pass removed.
                    "setup_content":      setup_content,
                    "refined_strategy":   variation.get("refined_strategy", "") if isinstance(variation, dict) else "",
                    "variation_number":  var_idx,
                    "repetition_number": rep,
                    "created_at": datetime.now().isoformat(),
                },
                "messages":  transcript_msgs,
                "judgment":  None,
            }

            turn_logprobs = [
                m["targeted_response_start_logprob"]
                for m in transcript_msgs
                if m.get("targeted_response_start_logprob") is not None
            ]
            avg_logprob = round(sum(turn_logprobs) / len(turn_logprobs), 4) if turn_logprobs else None

            filename = f"transcript_v{var_idx}r{rep}.json"
            save_json(transcript_data, transcripts_dir / filename)
            print(f"  Rollout v{var_idx}r{rep} done ({current_turn} turns, "
                  f"avg_logprob={avg_logprob})", flush=True)

            rollout_entry: Dict[str, Any] = {
                "variation_number":      var_idx,
                "variation_description": var_desc,
                "repetition_number":     rep,
                "num_turns":             len(transcript_msgs),
                "transcript_file":       filename,
                "kickoff_score":         round(kickoff_score, 4) if kickoff_score is not None else None,
            }
            if avg_logprob is not None:
                rollout_entry["avg_logprob"] = avg_logprob
            rollouts.append(rollout_entry)

    # Save full search pool summaries
    save_json({"beast_pools": beast_pool_data}, output_dir / "beast_pool.json")
    if output_search_on:
        save_json({"output_pools": output_pool_data}, output_dir / "output_pool.json")
    rollouts.sort(key=lambda x: (x["variation_number"], x["repetition_number"]))
    all_lps = [r["avg_logprob"] for r in rollouts if r.get("avg_logprob") is not None]
    mean_avg_logprob = round(sum(all_lps) / len(all_lps), 4) if all_lps else None

    rollout_results = {
        "metadata": {
            "evaluator": evaluator_model_id,
            "target":    target_model_id,
            "max_turns": max_turns,
            "input_search": search_cfg,
            "output_search": output_cfg,
        },
        "rollouts":        rollouts,
        "successful_count": len(rollouts),
        "failed_count":    0,
        "total_count":     len(variations) * suffixes_per_scenario,
        "logprob_summary": {
            "mean_avg_logprob": mean_avg_logprob,
            "num_scored":       len(all_lps),
        },
    }
    # Token-prob stats of the chosen outputs, computed here while the corruption target model
    # is still loaded (reused — no reload). Stashed for the judgment summary + rollout.json.
    #   • target_only (baseline/BoN) → probs captured DURING the HF decode, read from the
    #     transcript — no extra forward pass.
    #   • corruption ON → score the exact PoE outputs with the loaded HF corruption target.
    # Both report the same token metrics (same HF target model, same summary).
    _tok = None
    try:
        _tok = token_stats_from_stored(output_dir)   # on-policy probs captured during the (jail/BoN) HF decode, read from the transcript
    except Exception as e:
        print(f"  [token stats] skipped: {e}", flush=True)
    if _tok:
        rollout_results["token_stats"] = _tok
        print(f"  Token probs: avg={_tok['A_mean_tok_pct']:.2f}% | mean-of-mins="
              f"{_tok['B_mean_of_mins_pct']:.3f}% | min-of-mins={_tok['B_min_of_mins_pct']:.5f}%", flush=True)
    save_json(rollout_results, output_dir / "rollout.json")
    if mean_avg_logprob is not None:
        print(f"  Mean avg logprob: {mean_avg_logprob:.4f} over {len(all_lps)} transcripts", flush=True)
    print(f"ROLLOUT STAGE - COMPLETED ({len(rollouts)} rollouts)", flush=True)
    return rollout_results

__all__ = ['run_understanding', 'parse_scenarios_response', 'run_ideation', '_STEERING_INSTRUCTIONS', '_TAG_INSTRUCTIONS_MSG_FIRST', '_TAG_INSTRUCTIONS_TRS_FIRST', '_TAG_INSTRUCTIONS_MSG_ONLY', '_tag_instructions', '_STRATEGY_INSTRUCTIONS_HEAD', '_strategy_instructions', '_build_evaluator_followup', '_kickoff_message', 'ConversationOrchestrator', 'run_single_rollout', 'run_refinement', 'run_rollout', 'run_rollout_batched_local']
