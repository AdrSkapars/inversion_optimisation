# BLOOM BEAST Experiment Procedure

## 1. Prepare runner

Copy the appropriate template into a per-experiment runner file. Templates:

- `bloom_beast_tree.py` — main template (io_search, MMR, lookahead-via-pool).
- `bloom_beast_tree_combo.py` — io_search with **token-level input_search delegation** (BEAST-inside-beam). Use when combining turn-level + token-level search.
- `bloom_beast_tree_mcts.py` — MCTS rollout (UCT + progressive widening + judge-as-value, optional `value_via_lookahead`).

Naming convention: `bloom_tree_exp_<name>.py`. Edit cfg at the bottom:

- `folder_name`: `runs_NN/<descriptive>` (don't overwrite previous runs).
- `evaluator_gpu_id` and `target_gpu_id` in `cfg.rollout` — preferred GPUs 2 and 3 to leave 0/1 free for labmates.
- Any per-experiment knobs (search cfg, jail flags, mcts cfg, etc.).

## 2. Check GPUs are free

```bash
nvidia-smi
```

## 3. Run in background

From `~/inversion_optimisation/`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run experiments/bloom/bloom_tree_exp_<name>.py > /tmp/bloom_<name>.log 2>&1
```

Use Claude's Bash tool with `run_in_background=true`. The `expandable_segments` env var reduces fragmentation when long contexts (max_turns≥3) push the evaluator GPU near OOM.

Set up monitoring via `ScheduleWakeup` (hourly) or `CronCreate` (longer intervals, e.g. `13 */4 * * *` for 4-hourly).

## 4. Once done (procedure step 4)

1. **Kill lingering GPU workers** — vLLM workers stay alive after the experiment ends:
   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
   ```
   Map PIDs to GPUs by UUID, kill only the PIDs on the GPUs that just freed up. **Do NOT blanket-kill** if other experiments are still running.

2. **Read summary**:
   ```bash
   cat experiments/bloom/runs_NN/<folder>/round_1/judgment.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('summary_statistics'))"
   ```

3. **Inform the user** with avg / elic / runtime, comparing against relevant baselines.

4. **Push results**:
   ```bash
   git add experiments/bloom/runs_NN/<folder>/ && git commit -m "..." && git push
   ```

5. **Delete the per-experiment runner** (`bloom_tree_exp_<name>.py`). It was a snapshot of a template; future experiments get fresh copies. The **templates stay** (`bloom_beast_tree*.py`).

## Setup notes

- Judge model: `local/lmstudio-community/gemma-3-27b-it-GGUF:Q6_K:google/gemma-3-27b-it`
- Target model: `local/Qwen/Qwen3-4B`
- Hardware: 4× RTX A6000 (48GB each)
- GPU routing: all stages (understanding, ideation, rollout eval, judgment) share the evaluator GPU via `_DEFAULT_LOCAL_GPU_ID` (set from `cfg.rollout.evaluator_gpu_id` at startup). Only the target model uses a separate GPU (`cfg.rollout.target_gpu_id`).

## Memory / context

Each experiment is one process spawning its own vLLM worker subprocesses. No shared parent state between concurrent experiments — only OS-level disk/CPU contention, which is negligible.

For combo-style runs (io_search + input_search), jail PoE shares the target GPU and auto-halves `target_gpu_memory_utilization`.
