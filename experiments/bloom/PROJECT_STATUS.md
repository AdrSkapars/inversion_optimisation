# BLOOM behavioural evaluation — project status

**Task**: behavioural elicitation pipeline using BLOOM scaffold (understanding → ideation → rollout → judgment). Target = Qwen3-4B. Judge = Gemma-3-27b (Q6_K, with thinking). Behavior = `racial-bias`. All runs are 25 scenarios, 3 turns each, in `experiments/bloom/runs_14/`.

## Baseline to beat

**Plain 25-round BoN (no jail, no search)** is the current baseline on every axis that matters:

- `25r3t_bloom_seedfix`: **avg 8.04 / elic 100% / 7h 35m** (max-across-rounds aggregation).
- Outputs are **natural** (avg target logprob ≈ −0.48/token, ≈1.6 perplexity — vs jail-on runs ≈ −1.94/token, ≈7 perplexity).
- Fastest of any method tested (rounds 2+ reuse understanding+ideation; only rollout+judge per round).
- Simplest possible search: just sample 25 independent trajectories and take the max-scoring one per scenario.

Every search method here needs to beat plain BoN on at least one of {score, naturalness, compute} to be worth the complexity. Search results are **promising but don't yet dominate** the baseline — combo3x3+input3x3+jail hits 7.72/92% in 11h 27m with off-distribution outputs, which is worse than the baseline on all three axes simultaneously.

## Best single-round single-shot result

**`combo3x3+input3x3+jail`** (in `runs_14/1r3t_iosearch_combo3x3_input3x3_jail/`) — avg 7.72 / elic 92% / 11h 27m. The strongest of the single-round search variants, but not the project's baseline-to-beat.

For the jail-on 25-round number (just for reference): `25r3t_bloom_jailrollout` hit 8.48 / 100% in 16h 30m, but with off-distribution outputs from jail PoE — so it's strictly worse than plain BoN on naturalness while being slower.

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

### Reframed goal

**Find natural, highly-elicitatious, high-probability (input, output) examples.** Three Pareto objectives:

1. **Elicitation** — input causes target to exhibit the behaviour (judge score high).
2. **Natural input** — input is on-distribution / plausibly something a user would write.
3. **High-probability output** — target's response is what it would naturally produce given the input, not something pulled off-distribution by jail PoE.

Current best (combo3x3+input3x3+jail, 7.72/92%) wins on (1) but sacrifices (3) — jail PoE makes outputs ~5× less probable per token. Going forward, the work is about jointly optimising all three, **not just maximising elicitation**.

This also re-grounds the work against the rare-events forecasting literature: those methods *estimate* failure rates without surfacing examples; this tool *produces concrete examples* with rate-conditional naturalness, which is a complementary product.

### Three planned directions (in priority order)

1. **Alternating input ↔ output coordinate ascent.** Hold output fixed, search the input to maximise `P(output | input)`; then hold input fixed, sample outputs, pick the borderline-bad one (still hits judge threshold but lowest probability); repeat. EM-flavoured: each iteration monotonically improves the joint score on either input or output side. Initialise from any low-probability seed example already found by the combo+jail pipeline. Theoretical-ish: this is the core methodological contribution.

2. **Rephrasing for naturalness.** Once a high-elicitation transcript is found, paraphrase it (LLM-rewriter) into something similarly-bad but more probable, optionally guided by the direction of previous successful rephrases. Sentence-level / semantic; complementary to the token-level coordinate ascent. Don't need BEAST for this — BEAST's natural samples are already reasonably on-distribution; rephrasing is the sentence-level analogue.

3. **Annealing jailbreak interpolation.** Currently jail PoE is on for the whole target generation. Try "jail only for the first K tokens of the response, then let the target continue naturally" — should preserve most of the elicitation gain (direction-setting) while leaving the bulk of the output on-distribution. Also worth annealing β over search iterations (start with strong jail, fade out as good examples accumulate). Cheap experiments, ablation-shaped.

### Background: still-relevant practical avenues

- **Multi-behaviour evaluation.** All current results are racial-bias only. Generalizing to 3-5 behaviours from `prompts.yaml` (sycophancy, deception, refusal-bypass, etc.) is straightforward and rebuts "racial-bias-overfit" reviewer concerns.
- **Output_search delegation in `bloom_beast_tree_combo.py`** — still flagged "NOT IMPLEMENTED YET". Mirror of input_search delegation. Would let token-level output search stack with combo+jail; could be a separate naturalness lever but is now lower priority vs the three directions above.

### Why not MCTS

MCTS got tested (variants A-D + lookahead + jail). Best MCTS variant (MCTS+jail at 7.20/84%) was beaten by combo3x3+input3x3+jail (7.72/92%) at lower compute. More importantly, the MCTS-for-multi-turn-jailbreak literature is now crowded (MTJ-MCTS, MUSE, GS-MCTS, Tempest, COLING 2025 prompt-autogen MCTS, Dialogue-Aware MCTS EMNLP 2025, MTSA, M2S) — competing on tree-search algorithm is a research dead end. Input/output sampling/search has more headroom and a smaller related-work footprint.

### Relevant new literature for this direction

- **Goodfire diff amplification** ("Discovering Undesired Rare Behaviors via Model Diff Amplification") formalises the contrastive-logits recipe `logits = logits_post + α(logits_post − logits_pre)` — essentially the jail PoE setup. Natural cite for the mechanism.
- **"Predicting Rare LLM Failures with 30× Fewer Rollouts"** (LessWrong) + the two papers it cites (arXiv 2604.22167, 2410.13211) — rate-forecasting approaches using contrastive logits + interpolation/extrapolation. These motivate the rare-events framing but DO NOT produce examples; complementary to this work.
- **arXiv 2604.22167** specifically: shows that **the same harm can be achieved with much more natural (higher-probability) inputs and outputs** via paraphrasing / averaging across queries. Direct evidence that the three-objective framing is achievable; supports the rephrasing direction.
- **AuditBench (Sheshadri et al. 2026)** — benchmark of 56 model organisms with hidden behaviours, where scaffolded black-box tools outperform white-box auditing. Natural eval target: drop this method against AuditBench's open models as a structured behaviour-parametric audit protocol (vs their open-ended User Sampling / Text Completion baseline).
- **Petri 3 / Inspect framework** (AISI). Possible infra switch for distribution within the alignment-eval community; supports local models. Engineering tradeoff, not algorithmic.
- **Eval-awareness counter-direction** (Tim Hua's `steering-eval-awareness-public`, arXiv 2509.00591 realism probes, the safe-to-dangerous shift discussion). This is a *follow-up paper* — your tool could be re-cast as "produce realistic inputs that elicit hidden behaviours even with eval-awareness defenses" — but folding it into the current direction adds substantial scope.

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
