# BLOOM behavioural evaluation — project status

**Task**: behavioural elicitation pipeline using BLOOM scaffold (understanding → ideation → rollout → judgment). Target = Qwen3-4B. Judge = Gemma-3-27b (Q6_K, with thinking). Behavior = `racial-bias`. All runs are 25 scenarios, 3 turns each, in `experiments/bloom/runs_14/`.

## Headline result so far

**`combo3x3+input3x3+jail`** (in `runs_14/1r3t_iosearch_combo3x3_input3x3_jail/`) — **avg 7.72 / elic 92% / 11h 27m, single round**. The leader on single-round single-shot.

For the 25-round BoN baseline (max-across-rounds aggregation): `25r3t_bloom_jailrollout` hit **8.48 / 100%**, but that's a different scoring regime (max across 25 independent rounds vs single-round single-shot).

## What's been tested (ordered by recency, all single-round 1r3t)

| Folder | avg | elic | runtime | Notes |
|---|---|---|---|---|
| `1r3t_iosearch_beam5x5_bestfirst_mmr07` | 5.48 | 28% | 19h 41m | MMR=0.7 vs no-MMR baseline 5.48/36% — MMR didn't help, slightly hurt elic |
| `1r3t_iosearch_combo3x3_input3x3_jail` | **7.72** | **92%** | 11h 27m | **Best single-round.** io_search 3×3 + token-level input_search 3×3 (19-iter BEAST) + jail PoE |
| `1r3t_iosearch_combo3x3_input5tok_jail_pfx-100` | 7.28 | 76% | 13h 00m | Combo with 100-token searched suffix replacing last 100 body tokens — dropped vs 19-tok suffix |
| `1r3t_iosearch_combo3x3_input5tok_jail` | 7.52 | 84% | 14h 56m | Combo with 100-token searched suffix appended to body — also dropped |
| `1r3t_mcts_cuct5_batch5_budget60` (MCTS A) | 4.48 | 24% | 8h 42m | UCT + PW; partial-transcript judge as value. No jail, no lookahead |
| `1r3t_mcts_cuct2_batch10_budget60` (MCTS B) | 3.36 | 12% | 7h 34m | Lower c_uct → more exploit → worse |
| `1r3t_mcts_pwC3_cuct5_batch5_budget60` (MCTS D) | 4.48 | 28% | 8h 22m | Wider PW; ~same as A |
| `1r3t_mcts_cuct10_batch5_budget60` (MCTS C) | 4.24 | 20% | 8h 38m | Higher c_uct hurt vs c_uct=5 |
| `1r3t_mcts_best_lookahead` | 6.32 | 64% | 12h 38m | MCTS A + `value_via_lookahead=True`. **Big jump (+1.84/+40pp) over MCTS A** |
| `1r3t_mcts_best_jail` | 7.20 | 84% | 15h 09m | MCTS A + jail PoE. Bigger jump than lookahead (+2.72/+60pp over MCTS A) |
| `1r3t_mcts_best_lookahead_jail` | **(running)** | | | MCTS A + lookahead + jail combined. Predicted ~7.6-8.0/90%+ |

Also relevant earlier results (in `runs_14/`):
- `25r3t_bloom_seedfix` (vanilla 25-round BoN, max-across-rounds): 8.04 / 100% / 7h 35m
- `25r3t_bloom_jailrollout` (25-round + jail): **8.48 / 100% / 16h 30m** (overall best with the max-across aggregation)
- `25r3t_bloom_refine_hist2` (25-round + cross-round refinement): 8.24 / 100%
- `1r3t_iosearch_beam5x5_bf_jail` (io_search + jail, no input_search): 7.36 / 84% / 19h 55m
- `1r3t_iosearch_beam5x5_bestfirst` (io_search no jail): 5.48 / 36%

## What works (verified levers)

1. **Jail PoE during target sampling** (`jailbroken_output.use_during_rollout=True`) is the dominant lever. Consistent +1.5–2.7 avg / +30-60pp elic across every search method tested. The tradeoff: outputs are ~5× less probable per token under p_target alone (see `analysis_output_logprobs/`).
2. **Token-level input_search** inside io_search (combo file) adds another +0.36 avg / +8pp elic over io_search+jail without input_search. Combo file: `bloom_beast_tree_combo.py`.
3. **Chain-end value signal** (judge the chain-end transcript, not a partial one) — confirmed for both io_search (1×25 lookahead vs 1×25 best-first) and MCTS (lookahead +1.84/+40pp). Partial-transcript judge is too noisy.

## What didn't work

1. **MCTS structure** (UCT + PW) doesn't beat best-first at the same compute when value signal is just the judge. UCB needs a clean value signal; partial-transcript judge isn't precise enough. With chain-end value (lookahead), MCTS does beat best-first+lookahead by +1.24 avg.
2. **MMR diversity-aware beam selection** (`mmr_lambda=0.7`) gave the same avg as no-MMR, slightly worse elic. Probably noise; single-run.
3. **Extending the searched portion to 100 tokens** (5-tok chunks × 20 iters) underperformed the standard 19-token suffix attack. Possibly because longer searched portions are less coherent than the eval LM's natural body, or because TRS reward isn't as well-aligned to the judge at this length.
4. **Lower c_uct** (more exploit) for MCTS hurt; higher c_uct also hurt slightly. c_uct=5 was the peak.

## Important context / lessons

- **Judge variance is small.** Re-judged all 625 transcripts of `25r3t_bloom_seedfix` 3 times with seeds 1000/2000/3000 (`analysis_rejudge_bon25/`). Per-transcript std avg 0.31, 21/25 scenarios pick the same max-scoring round across all 4 judge passes. BoN-25 aggregate scores are stable.
- **Output probabilities under jail** are ~5× less probable per token (`analysis_output_logprobs/`). Visible in transcripts as weirder phrasing ("arevery", "donvHave", random caps).
- **Cross-round seed determinism bug** (fixed in `bloom_beast_tree.py`) — rounds 2+ used to produce byte-identical transcripts because `_DEFAULT_SEED` wasn't per-round. Now `base_seed + round_num`. Affects pre-seedfix runs.

## Where the project is going next

User has indicated **moving away from MCTS and back to input/output sampling** — i.e. the combo file direction (token-level input search inside turn-level beam) plus exploring output_search. Open avenues:

1. **Implement output_search delegation in `bloom_beast_tree_combo.py`** — currently logs "NOT IMPLEMENTED YET" if `output_search.enabled` is set alongside io_search. Would mirror the input_search delegation: `output_search_target_responses_batched` taking M (target_msgs, baseline_response) contexts and batching the token-level beam across them.
2. **Combo variations not yet tested**:
   - 5×5 turn-beam × 3×3 token-beam (compute-heavy, ~30h).
   - 3×3 turn-beam × variations on input_search params (different `max_prefix_length`, etc.).
   - Combo + post-hoc judge variance check (rejudge the best combo run with 3 seeds).
3. **Multi-behaviour evaluation** — all current results are racial-bias only. Generalizing requires `prompts.yaml` updates (other behavior prompts already exist in the file).

## Key files (paths from repo root)

- **Templates** (do not delete after experiments):
  - `experiments/bloom/bloom_beast_tree.py` — main io_search template (MMR, lookahead-via-pool).
  - `experiments/bloom/bloom_beast_tree_combo.py` — io_search + input_search delegation. Output_search delegation NOT implemented.
  - `experiments/bloom/bloom_beast_tree_mcts.py` — MCTS template with `value_via_lookahead` support.
- **Analysis scripts**: `experiments/bloom/scripts/{analyze_output_logprobs.py, rejudge_bon25_variance.py}`.
- **Cfg helper**: `/tmp/apply_mcts_edits.py` (one-shot text edits for MCTS runners). Not in repo; reconstruct if needed.
- **Procedure**: `experiments/bloom/EXPERIMENT_PROCEDURE.md`.

## Hardware

4× RTX A6000 (48GB each). Eval GPU is the bottleneck (Gemma-27b judge with thinking).
