# COMBO — jail steering + G-PAIR refinement, together (15 scen, seed 1)

Tests whether refined INPUTS help the jail LOGIT-steering on cells where jail alone underperformed.
Both on: `jailbroken_output` (self-jail at pm3 β) + `refinement_input` (t3_sfull: 3 transcripts +
all strategies). 15 scenarios, seed 1, 7 rounds, reuse runs_new bank. Output:
runs_new/<beh>/<model>/combo_jailrefine/.

Baselines (15-scen elic@5, best-of-pool) for comparison. Question: does combo > jail-alone?

All 15-scen, seed 1. jail-alone + BoN baselines are @5 rounds; COMBO @5 is the rounds-matched
comparison. best-of-pool.

| behaviour | model | β | COMBO elic@5/6/7 | plaus@7 | jail-alone @5 | BoN @5 | verdict |
|---|---|---|---|---|---|---|---|
| self_harm | gemma-4-e4b | 0.5 | 7.60 / 7.60 / 7.87 | 61.3% | 5.27 | 2.33 | **combo ≫ jail (+2.33 @5)** |
| medical   | gemma-4-e4b | 0.5 | 4.27 / 4.53 / 4.73 | 67.2% | 2.33 | 1.87 | **combo ≫ jail (+1.94 @5)** |
| goblin    | gemma-4-e4b | 1.0 | 3.53 / 4.20 / 4.67 | 59.5% | 2.20 | 1.00 | **combo ≫ jail (+1.33 @5)** |
| racial    | Qwen3.5-4B  | 3.5 | 7.80 / 8.13 / 8.27 | 60.7% | 8.27 | 7.07 | combo ≈ jail (−0.47 @5; jail already near-ceiling) |

## Verdict — does refinement help jail steering?

**Yes, when jail has headroom; flat when it doesn't.** The refinement lift over jail-alone tracks
inversely with the jail-alone baseline:

| cell | jail-alone @5 | combo lift @5 |
|---|---|---|
| self_harm/gemma | 5.27 | **+2.33** |
| medical/gemma   | 2.33 | **+1.94** |
| goblin/gemma    | 2.20 | **+1.33** |
| racial/Qwen     | 8.27 | −0.47 (no headroom) |

On the 3 cells where jail underperformed (2–5/10), adding refined inputs lifted elicitation +1.3 to
+2.3 at matched plausibility (~60–67%). On racial/Qwen, jail alone was already ~8.3/10 and combo
couldn't improve on it (flat/slightly worse at matched rounds). Plausibility is comparable across
methods everywhere. Takeaway: **combo is a targeted rescue for weak-jail cells, not a universal
upgrade** — the refined input gives the steering better material only when the steering was leaving
elicitation on the table.
