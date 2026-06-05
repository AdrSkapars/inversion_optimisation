"""Wrapper around bloom_beast_tree to sweep BEAST input-search cfg knobs.

5 scenarios × {3 prefill modes × 5 (scored,kept) rungs} + 1 baseline (no search)
= 16 runs. Same understanding.json + ideation.json shared across all.

Flow per cfg:
  1. cfg.ideation.num_scenarios=5, max_turns=1, num_rounds=1, target_before_input=True
  2. Set input_search knobs per (mode, scored, kept).
  3. Run bloom_beast_tree.run_pipeline(cfg) — produces transcripts that include
     `targeted_response_start_logprob` per turn-1 entry (saved by _score_trs_batch).

After all runs: read every transcript, extract the TRS lp per scenario,
build a sweep table.

`bloom_beast_tree.py` is NOT modified — we just override cfg knobs and call its
async pipeline entry point.

Skipping wasted compute (target reply + judgment) without forking pipeline:
  - target reply: set cfg.rollout.target_max_tokens=1 → target generates one
    token after input_search finishes; the input_search itself uses
    eval_max_tokens (separate budget) so it's unaffected.
  - judgment: monkey-patch bbt.run_judgment to an async no-op returning a
    placeholder so run_pipeline proceeds and saves transcripts cleanly.

Prefill modes → input_search.max_prefix_length:
  - "full"  : None  (keep full Phase-1 body; BEAST appends after it)
  - "half"  : +50   (keep first 50 tokens of body; BEAST writes rest)
  - "none"  : 0     (cursor right after <message>; BEAST writes whole body)

Latin mask + truncate_at_eos=False ensures the `<|im_end|>`/EOS exploit is
blocked (EOS is the same token as <|im_end|> in Qwen; truncate_at_eos=True
would re-allow it).
"""
from __future__ import annotations
import asyncio, copy, json, shutil, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Import from bloom_beast_tree without triggering its __main__ block.
import bloom_beast_tree as bbt  # noqa: E402


# ── Monkey-patch: skip judgment entirely ────────────────────────────────────
# We only care about the input_search BEAST score (saved on transcripts via
# bbt._score_trs_batch). Running the Gemma 27B judge over 5 transcripts per
# sweep cell is wasted compute. Replace with an async no-op that returns a
# minimal placeholder; run_pipeline still completes "successfully".
_ORIGINAL_RUN_JUDGMENT = bbt.run_judgment

async def _noop_run_judgment(cfg, prompts_yaml, output_dir, understanding_results, ideation_results):
    """No-op replacement for bbt.run_judgment. Writes a stub judgment.json so
    run_pipeline's existence checks remain happy on resume."""
    import json as _json
    stub = {
        "judgments": [],
        "summary_statistics": {"average_behavior_presence_score": 0.0, "elicitation_rate": 0.0},
        "_skipped": True, "_reason": "input-search sweep skips judgment",
    }
    out = output_dir / "judgment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        _json.dump(stub, f, indent=2)
    print("  [judgment] skipped (sweep wrapper override)", flush=True)
    return stub

bbt.run_judgment = _noop_run_judgment


# ── Monkey-patch: short-circuit the post-input_search target reply ──────────
# In `run_rollout_batched_local`, after input_search produces the eval message,
# the target generates a reply via `batch_generate_local(lm_target, ...)`
# (line ~4391 of bloom_beast_tree.py). With output_search and io_search and
# jail.use_during_rollout all disabled (which our sweep cfg ensures), that is
# the ONLY call to batch_generate_local with the target LocalModel as first
# arg. All other call sites pass lm_eval or lm_judge — they pass through.
#
# We identify the target instance by its `spec_str` (set in LocalModel.__init__
# from the hf_name argument), comparing against cfg.rollout.target stripped of
# the "local/" prefix.
_ORIGINAL_BATCH_GENERATE_LOCAL = bbt.batch_generate_local
_TARGET_SPEC_STR = bbt.cfg.rollout.target[len("local/"):]  # e.g. "Qwen/Qwen3-4B"

def _patched_batch_generate_local(lm, messages_list, max_new_tokens, temperature, *args, **kwargs):
    """Skip generation entirely when `lm` is the target model — return empty
    strings, one per input message list. The downstream caller appends the
    empty assistant turn and proceeds; with max_turns=1 the rollout loop then
    exits and the transcript is saved with the TRS lp from input_search."""
    if getattr(lm, "spec_str", None) == _TARGET_SPEC_STR:
        return [""] * len(messages_list)
    return _ORIGINAL_BATCH_GENERATE_LOCAL(
        lm, messages_list, max_new_tokens, temperature, *args, **kwargs
    )

bbt.batch_generate_local = _patched_batch_generate_local


# ── Monkey-patch: capture baseline (before-search) TRS lp ───────────────────
# When input_search is disabled, the original input_search_evaluator_message
# returns the Phase 1 baseline message with score=0.0 (a placeholder), and
# that 0.0 gets written into the transcript as targeted_response_start_logprob.
# That breaks the before/after comparison. We patch it to compute the real
# log P(TRS | sys, baseline_msg) when input_search is disabled, so the
# baseline cell's transcript ends up with the actual "before" score.
_ORIGINAL_INPUT_SEARCH_EVAL = bbt.input_search_evaluator_message

def _patched_input_search_evaluator_message(
    lm_eval, lm_target, eval_msgs, target_msgs, search_cfg, *args, **kwargs
):
    pool, trs, strategy = _ORIGINAL_INPUT_SEARCH_EVAL(
        lm_eval, lm_target, eval_msgs, target_msgs, search_cfg, *args, **kwargs
    )
    # Only rescue when input_search was disabled AND we have a TRS to score.
    if not search_cfg.get("enabled", False) and pool and trs:
        baseline_msg, _, baseline_prefix, suffix = pool[0]
        items = [(list(target_msgs) + [{"role": "user", "content": baseline_msg}], trs)]
        scores = bbt.batch_logprob_local(lm_target, items)
        if scores and scores[0] is not None:
            pool = [(baseline_msg, float(scores[0]), baseline_prefix, suffix)] + list(pool[1:])
    return pool, trs, strategy

bbt.input_search_evaluator_message = _patched_input_search_evaluator_message


# ── Sweep config ─────────────────────────────────────────────────────────────
BASE_FOLDER         = "runs_15/input_search_sweep"
NUM_SCENARIOS       = 15
NUM_BEAMS           = 5
CANDIDATES_PER_BEAM = 5
MAX_ITERATIONS      = 20
LADDER: List[Tuple[int, int]] = [(1, 1), (2, 2), (5, 5), (10, 10), (20, 20)]
PREFILL_MODES: Dict[str, Optional[int]] = {
    "full": None,   # cursor right before </message>; full body in prefix
    "half": 50,     # cursor after first 50 body tokens
    "none": 0,      # cursor right after <message>; BEAST writes whole body
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _share_understanding_ideation(base_dir: Path, dest_dir: Path) -> None:
    """Symlink (or copy) understanding.json + ideation.json from base run.

    bloom_beast_tree.run_pipeline skips understanding/ideation when their JSON
    files already exist in the output folder, so this avoids re-running those
    stages for every sweep cfg.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("understanding.json", "ideation.json"):
        src = base_dir / fname
        dst = dest_dir / fname
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(f"Expected {src} after the base run")
        shutil.copy2(src, dst)


def _build_cfg(folder: str, *, input_search_enabled: bool,
               scored_len: int = 1, kept_len: int = 1,
               max_prefix_length: Optional[int] = 0) -> "bbt.DotDict":
    """Build a customised cfg from bbt.cfg defaults."""
    cfg = bbt.DotDict(copy.deepcopy(dict(bbt.cfg)))

    cfg.folder_name = folder

    # Reduce scenario count + single round, single turn.
    cfg.ideation.num_scenarios = NUM_SCENARIOS
    cfg.refinement.num_rounds = 1
    cfg.rollout.max_turns = 1
    cfg.rollout.target_before_input = True

    # Target reply is short-circuited at the batch_generate_local wrapper layer
    # (see _patched_batch_generate_local). Leave target_max_tokens at default;
    # the monkey-patch never reaches vLLM for the target model.

    # Disable all other search modes to keep the comparison clean.
    cfg.output_search.enabled = False
    cfg.input_and_output_search.enabled = False
    cfg.jailbroken_output.use_during_rollout = False
    cfg.jailbroken_output.output_search_loss = False
    cfg.jailbroken_output.input_search_loss = False

    # input_search knobs.
    cfg.input_search.enabled = input_search_enabled
    if input_search_enabled:
        cfg.input_search.num_beams = NUM_BEAMS
        cfg.input_search.candidates_per_beam = CANDIDATES_PER_BEAM
        cfg.input_search.scored_candidate_length = scored_len
        cfg.input_search.kept_candidate_length = kept_len
        cfg.input_search.max_num_iterations = MAX_ITERATIONS
        cfg.input_search.max_prefix_length = max_prefix_length
        cfg.input_search.latin_mask = True
        cfg.input_search.truncate_at_eos = False  # block <|im_end|>/EOS exploit
        cfg.input_search.unscored_filler_length = 0
        cfg.input_search.suffixes_per_scenario = 1
        cfg.input_search.beast_temperature = 0.0

    return cfg


def _resolve_behavior_and_preset(cfg: "bbt.DotDict") -> None:
    """Mirror the bottom-of-file resolution that bloom_beast_tree.__main__ does
    so we can call run_pipeline directly from this wrapper."""
    prompts_path = SCRIPT_DIR / cfg.get("prompts_file", "prompts.yaml")
    import yaml
    with open(prompts_path, "r", encoding="utf-8") as f:
        _prompts = yaml.safe_load(f)
    behavior_desc = _prompts.get("behaviors", {}).get(cfg.behavior_name, "")
    if not behavior_desc:
        raise ValueError(f"No behavior description for {cfg.behavior_name!r}")
    cfg["behavior_description"] = behavior_desc.strip()

    preset_name = cfg.get("prompt_preset", "")
    if preset_name:
        preset = _prompts.get("prompt_presets", {}).get(preset_name)
        if preset:
            for k, v in preset.items():
                if k not in cfg:
                    cfg[k] = v.strip() if isinstance(v, str) else v


# ── Main sweep loop ─────────────────────────────────────────────────────────
async def _run_one(cfg: "bbt.DotDict", label: str) -> None:
    print("\n" + "#" * 70, flush=True)
    print(f"#  SWEEP CELL: {label}", flush=True)
    print(f"#  folder:    {cfg.folder_name}", flush=True)
    print("#" * 70, flush=True)
    _resolve_behavior_and_preset(cfg)
    # run_pipeline writes its own seed defaulting via cfg.seed if set.
    result = await bbt.run_pipeline(cfg)
    if result is None:
        print(f"  ❌ FAILED: {label}", flush=True)
    else:
        stats = result.get("summary_statistics", {})
        print(f"  ✓ {label}: avg_score={stats.get('average_behavior_presence_score', 0):.2f}", flush=True)


def _read_trs_lps(round_dir: Path) -> List[Dict]:
    """Read all transcripts in round_dir/transcripts/ and extract per-variation
    `targeted_response_start_logprob` plus the user message + TRS for context."""
    out = []
    tdir = round_dir / "transcripts"
    if not tdir.exists():
        return out
    for f in sorted(tdir.glob("transcript_v*r*.json")):
        d = json.load(open(f, encoding="utf-8"))
        msgs = d.get("messages", [])
        user_entry = next((m for m in msgs if m["role"] == "user"), None)
        if user_entry is None:
            continue
        out.append({
            "transcript_file": f.name,
            "variation_number": d.get("variation_number"),
            "rep_number":       d.get("rep_number"),
            "user_msg":         user_entry.get("content", ""),
            "targeted_response_start":         user_entry.get("targeted_response_start", ""),
            "targeted_response_start_logprob": user_entry.get("targeted_response_start_logprob"),
        })
    return out


async def main():
    t0 = time.time()
    base_round_dir = SCRIPT_DIR / BASE_FOLDER / "baseline_no_search" / "round_1"

    # 1) BASELINE RUN: input_search disabled — also produces understanding + ideation
    #    that the rest of the sweep can reuse.
    baseline_cfg = _build_cfg(
        folder=f"{BASE_FOLDER}/baseline_no_search/round_1",
        input_search_enabled=False,
    )
    await _run_one(baseline_cfg, label="baseline (no input search)")
    base_round_dir = (SCRIPT_DIR / baseline_cfg.folder_name).resolve()

    # 2) SWEEP RUNS: each cfg gets its own folder, sharing understanding+ideation.
    for mode, max_pre in PREFILL_MODES.items():
        for scored_len, kept_len in LADDER:
            label = f"input_search mode={mode} (scored,kept)=({scored_len},{kept_len})"
            folder = f"{BASE_FOLDER}/{mode}_s{scored_len}_k{kept_len}/round_1"
            dest_dir = SCRIPT_DIR / folder
            _share_understanding_ideation(base_round_dir, dest_dir)
            cfg = _build_cfg(
                folder=folder,
                input_search_enabled=True,
                scored_len=scored_len,
                kept_len=kept_len,
                max_prefix_length=max_pre,
            )
            await _run_one(cfg, label=label)

    # 3) AGGREGATE — read all transcripts and build a per-(mode, scored, kept) table.
    print("\n" + "=" * 70, flush=True)
    print(f"AGGREGATE  (elapsed {time.time()-t0:.0f}s)", flush=True)
    print("=" * 70, flush=True)
    import math

    base_lps = _read_trs_lps(base_round_dir)
    base_by_v: Dict[int, float] = {
        r["variation_number"]: r["targeted_response_start_logprob"]
        for r in base_lps if r["targeted_response_start_logprob"] is not None
    }
    print(f"\nBaseline (before search) — {len(base_lps)} transcripts:")
    for r in base_lps:
        v = r["variation_number"]; lp = r["targeted_response_start_logprob"]
        p_pct = (math.exp(lp) * 100) if lp is not None else None
        print(f"  v={v}  lp_before={lp}  per_token_p={p_pct:.3f}%" if p_pct is not None
              else f"  v={v}  lp_before=None")

    sweep_results: Dict[str, Dict[Tuple[int, int], List[Dict]]] = {}
    print(f"\nPer-cell results (lp_after = BEAST best score per scenario):")
    for mode in PREFILL_MODES:
        sweep_results[mode] = {}
        for scored_len, kept_len in LADDER:
            folder = f"{BASE_FOLDER}/{mode}_s{scored_len}_k{kept_len}/round_1"
            rows = _read_trs_lps(SCRIPT_DIR / folder)
            sweep_results[mode][(scored_len, kept_len)] = rows
            deltas = []
            for r in rows:
                v = r["variation_number"]
                lp_after = r["targeted_response_start_logprob"]
                lp_before = base_by_v.get(v)
                if lp_after is None or lp_before is None: continue
                pct = (math.exp(lp_after) / math.exp(lp_before) - 1) * 100
                deltas.append(pct)
            mean_d = sum(deltas)/len(deltas) if deltas else None
            best_d = max(deltas) if deltas else None
            line = f"  {mode:<5} ({scored_len:>2},{kept_len:>2})  n={len(deltas)}"
            if mean_d is not None:
                line += f"  mean Δp={mean_d:+6.1f}%  best Δp={best_d:+6.1f}%"
            print(line)

    # Save aggregate JSON.
    out_path = SCRIPT_DIR / f"{BASE_FOLDER}/aggregate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "baseline": base_lps,
        "sweep": {mode: {f"{s}_{k}": rows for (s, k), rows in cells.items()}
                  for mode, cells in sweep_results.items()},
        "ladder": LADDER,
        "prefill_modes": PREFILL_MODES,
    }, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved aggregate → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
    # Hard-exit so vLLM workers don't hang on Python shutdown.
    import os
    os._exit(0)
