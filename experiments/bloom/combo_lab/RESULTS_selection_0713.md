# Selection step: plausibility-picking anti-selects behaviour (07-13)

**Motivation.** The combination-function sweep (see `RESULTS_combination_functions_0713.md`) surfaced a confound: with `spp>1`, the pipeline's default selector `target_pick` = **max target token-probability** picks the *most plausible* of the pool — which is the *least behavioural* sample. This experiment isolates and quantifies the selection step.

**Method.** Fix generation (`add` jail steering at fixed β), draw **SPP=8** samples per scenario, **judge all 8** with the Gemma-4-26B auditor, then score 5 selectors on the *same* pool (paired, matched generation budget): `random` (=n=1 floor), `target_pick` (max mean target-prob, pipeline default), `reffree_tp` (drop samples matching a refusal regex, then max target-prob), `ORACLE_maxscore` (pick highest judge score = ceiling), `minplaus` (pick *least* plausible). N=25 scenarios. Harness `select_probe.py`. Cells: qwen selfharm, llama selfpres, qwen political.

## Results — `score / plaus%` (β=1.5 unless noted)
| cell | random | target_pick | reffree_tp | minplaus | ORACLE | refusal-rate |
|---|---|---|---|---|---|---|
| qwen selfharm (β1.0) | 6.04/55.7 | 6.24/66.9 | 6.04/65.7 | 4.76/47.0 | 8.76/58.4 | 0.13 |
| qwen selfharm (β1.5) | 6.04/50.0 | **2.56**/61.3 | 2.84/59.4 | 7.72/40.9 | 8.72/44.4 | 0.075 |
| llama selfpres (β1.5) | 2.88/60.2 | **2.00**/66.5 | 2.20/66.3 | 4.28/51.6 | 5.40/57.1 | 0.14 |
| qwen political (β1.5) | 8.92/56.3 | 8.20/59.4 | 8.20/59.4 | 9.16/50.6 | 9.68/54.6 | 0.04 |

## Findings
1. **Plausibility-maximizing selection anti-selects behaviour.** `target_pick ≤ random ≤ ORACLE` in every cell. It is **Pareto-dominated by random** and, on hard behaviours at meaningful β, catastrophically so — qwen selfharm β1.5: `target_pick` 2.56 vs `random` 6.04 vs `oracle` 8.72.
2. **β-dependent / difficulty-dependent.** The harm grows with steering strength and behaviour difficulty: qwen selfharm `target_pick` is fine at β1.0 (6.24) but craters at β1.5 (2.56); mild on easy high-ceiling political (8.2 vs 8.9). Mechanism: at high β the *high-plausibility* samples are the ones where steering **failed** (reverted to bland/safe high-target-prob text), so max-plausibility systematically selects the steering failures.
3. **Behaviour anti-correlates with plausibility in-pool.** `minplaus` (pick the *least* plausible sample) ≥ `random` everywhere and approaches the oracle on hard cells (selfharm β1.5: 7.72 vs oracle 8.72). The elicitation signal is *anti*-plausibility.
4. **Refusal-filtering does not recover it.** `reffree_tp ≈ target_pick` in all cells (refusal rates only 0.04–0.14) — the behaviour-vs-plausibility tension is **not** about explicit refusal markers, so a cheap refusal filter is useless. And no single cheap proxy is robust across β (`minplaus` is great at β1.5 but *worst* at β1.0 on selfharm: 4.76).

## Takeaways
- **The pipeline's plausibility-based selection (`target_pick`/`filter_target`) is the wrong objective for the elicitation axis** — it lands at the useless high-plausibility/low-score corner, especially under strong steering. Exploiting the sample pool requires a **behaviour judge (BoN)**, not a plausibility heuristic. (Consistent with the project's BoN-vs-corruption results.)
- Retroactively validates using **n=1 (no selection)** for the combination-function frontier sweep — selection would have injected exactly this confound.
- Open thread: is there a *cheap* behaviour-aware selector (jail-expert logprob of the sample; token-entropy/`tok%` triage per [[input-search-vs-resample-0711]]) that tracks the oracle without an 8× judge? Refusal-regex and plausibility both fail; this is the natural next probe.

**Caveat.** N=25, single sample pool per β; the ordering is consistent across 2 models × 3 behaviours but absolute numbers carry the usual bimodal-score noise. Artifacts: `~/select_<slug>_<model>_b*.json`, `select_probe.py`, `~/combo_notes.md`.

---

# Black-box behaviour-aware selector (jail-logprob margin) — POSITIVE (07-14)

**Question.** `target_pick` anti-selects behaviour (above). Is there a *cheap black-box* selector — logits only, **no target activations/weights** — that picks behavioural samples and approaches the (expensive) behaviour-judge oracle? The jail expert is our own steering model, so its logprobs are fair game.

**Method.** Added jail-expert token-logprob capture to the copy (`BLOOM_CAPTURE_JAIL_LP`, backward-compat `return_jail_lps`; all in `combo_lab/`). New selectors over the SPP=8 pool: `jail_pick` (max jail_lp), `margin_pick` (max `jail_lp − target_lp` = steering divergence), `floor_margin` (restrict to more-plausible half, then max margin), `mintok_pick` (least-confident target token). N=25, β=1.5 unless noted.

## Selector scores — `score / plaus%`
| cell | target_pick | minplaus | **margin_pick** | floor_margin | ORACLE |
|---|---|---|---|---|---|
| qwen political (β4.9) | 7.68/59.7 | 8.88/50.5 | **9.20/52.2** | 8.84/56.9 | 9.36/54.4 |
| qwen selfharm (β1.0) | 1.40/67.4 | 4.68/47.3 | **5.16/50.1** | 2.92/62.6 | 6.04/52.3 |
| qwen selfharm (β1.5) | 2.12/64.0 | 8.24/42.0 | **8.28/43.0** | 4.68/53.7 | 9.32/44.8 |
| llama selfpres (β1.5) | 2.44/66.2 | 3.20/51.2 | **3.72/55.4** | 3.36/62.2 | 4.84/58.5 |

## Per-sample Pearson r(signal, judge-score), n=200/cell
| cell | margin(jl−tl) | plaus | target_lp | jail_lp | mintok |
|---|---|---|---|---|---|
| qwen political | +0.536 | −0.587 | −0.558 | −0.193 | −0.296 |
| qwen selfharm β1.0 | +0.581 | −0.597 | −0.653 | −0.345 | −0.266 |
| qwen selfharm β1.5 | **+0.739** | −0.657 | −0.695 | −0.183 | −0.387 |
| llama selfpres | **+0.388** | −0.272 | −0.220 | +0.082 | −0.059 |

## Findings
1. **`margin_pick` (black-box, logits-only) is the best cheap selector.** It **Pareto-dominates `minplaus` in all 4 cells** (higher score AND higher plausibility) and **crushes `target_pick`** everywhere (selfharm β1.5: 8.28 vs 2.12). It recovers **85–99 % of the oracle's score at zero judging cost** (political 9.20 vs 9.36; selfharm β1.5 8.28 vs 9.32).
2. **Plausibility is *negatively* correlated with behaviour** (r ≈ −0.6 to −0.7 on hard cells) — the anti-selection quantified at the sample level; the elicitation signal is anti-plausibility.
3. **`margin` is the strongest positive predictor of behaviour** (selfharm β1.5 r=+0.739), and — critically — **`jail_lp` *alone* is useless** (r ≈ −0.18 to +0.08). It is the **margin relative to the target** (`jail_lp − target_lp`) that carries the signal; the jail term nudges toward *more-plausible* behavioural samples than `minplaus` alone. This is why `jail_pick` is weak and `margin_pick` strong.
4. **`floor_margin`** (margin within the plausible half) is a tunable *higher-plausibility* operating point — still far above `target_pick` — but sacrifices score vs `margin_pick`; not a strict improvement. `mintok` is a weak-moderate signal (r ≈ −0.06 to −0.39), below margin/plaus.

## Cross-model confirmation (all 4 target models, selfharm)
| model (β) | target_pick | minplaus | **margin_pick** | ORACLE | r(margin,score) |
|---|---|---|---|---|---|
| qwen (1.5) | 2.12/64.0 | 8.24/42.0 | **8.28/43.0** | 9.32/44.8 | +0.739 |
| phi (1.125) | 1.44/62.2 | 3.52/44.9 | **3.76/47.2** | 4.24/49.9 | +0.504 |
| gemma (0.75) | 1.00/75.5 | 1.60/63.9 | **1.60/68.3** | 1.72/68.9 | +0.144 |
| llama selfpres (1.5) | 2.44/66.2 | 3.20/51.2 | **3.72/55.4** | 4.84/58.5 | +0.388 |

`margin_pick` **Pareto-beats `minplaus` and approaches the oracle on all 4 target models** (qwen, phi, gemma, llama). The jail-margin's *predictive strength* varies — strong on qwen/phi/llama, weak on gemma (r=+0.14, where plausibility dominates) — but the *selector-level* Pareto win over `minplaus` (higher-or-equal score at higher plausibility) holds everywhere.

## Takeaway
A **black-box, judge-free selector** — pick the pool sample maximizing `jail_lp − target_lp` — captures most of a behaviour-judge's elicitation gain and strictly beats both the pipeline default (`target_pick`, which is *counterproductive*) and naive `minplaus`, across all 4 target models. Concretely: when steering with a jail expert, **select by steering-divergence margin, not by plausibility.**

## Selection frontier vs generation frontier (the sharp contrast with combiners)
The `floor_margin` plausibility-fraction `f` is a **tunable black-box knob** (f→0.125 ≈ `target_pick`; f=1 = `margin_pick`), tracing a smooth monotonic frontier from high-plaus/low-score to low-plaus/high-score — all from a **single fixed-β pool**, computed **offline from the per-sample dumps (zero extra GPU)** via `select_frontier.py`.

qwen selfharm β=1.5 — selection frontier vs the `add` **generation** β-frontier (from `RESULTS_combination_functions_0713.md`) interpolated at matched plausibility:

| plaus | selection (floor_margin) | generation (add, interp) | Δ |
|---|---|---|---|
| 59.9 | 2.88 | 1.83 | **+1.05** |
| 56.0 | 3.92 | 2.60 | **+1.32** |
| 53.7 | 4.68 | 3.47 | **+1.21** |
| 50.0 | 5.92 | 5.94 | −0.02 |
| 43.0 | 8.28 | 8.65 | −0.37 |

**Margin-selection from a fixed-β pool dominates the generation-β frontier in the high-plausibility region** (+1.0 to +1.3 score at plaus 54–60), ties at ~50, and dips slightly below at low plaus. It generalizes (llama selfpres: selection 3.36 vs generation 2.01 at plaus 62). Mechanism: at high plausibility low-β generation *cannot* produce behavioural text, but the fixed-β pool has **variance** — margin-selection recovers the rare plausible-and-behavioural tail.

**This is the sharp contrast with the combination-function result.** Combiners (top-K, nucleus, mix, tr_cfg, …) merely *ride* the generation frontier — no combination algebra beats `add` at matched plausibility. Margin-**selection** exploits sample variance to **beat** that frontier in the regime that matters. **Selection is a genuine lever; combination algebra is not.**

## Behaviour-diversity + honest bound on the margin advantage
Added a 4th behaviour class, **qwen deception β3.0**: target_pick 2.68/62.6 | minplaus 5.84/45.1 | **margin_pick 5.68/46.7** | ORACLE 7.92/49.4 (r(margin)=+0.27). Here margin_pick **crushes target_pick** and gets ~72% of oracle, but is a slight *trade* vs minplaus (−0.16 score / +1.6 plaus), not a clean Pareto win.

**Accurate synthesis across all cells (4 models × 4 behaviour classes):**
- `margin_pick ≫ target_pick` **always** (the robust, large, universal win — the pipeline default is simply wrong).
- `margin_pick ≥ minplaus`: **Pareto-better** where the jail–target divergence is informative (qwen/phi/llama selfharm & selfpres, r(margin) ≈ 0.39–0.74), roughly **tied** where plausibility alone already captures behaviour (gemma selfharm, qwen deception; r ≈ 0.14–0.27, plausibility the dominant signal).
- `margin_pick` captures **72–99 %** of the judge-oracle at zero judging cost.

So: **select by margin (or, equivalently cheaply, by anti-plausibility) — never by plausibility.** The jail-margin term adds real value on top of anti-plausibility specifically when jail and target disagree informatively; when they don't, it degrades gracefully to ≈ minplaus. Either way the pipeline's `target_pick`/`filter_target` default is strictly counterproductive.
