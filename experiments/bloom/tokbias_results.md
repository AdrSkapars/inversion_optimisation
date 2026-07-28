# TokenBias baseline — experiment results

TokenBias = static full-vocab logit tilt `z = target + λ·bias`, riding the BoN target_only path
(no jail expert, no refinement, no search). Bias vector = monitor-framed relevance prompt
(behaviour name+desc) + assistant prefill, averaged over positions/samples. λ=0 = pure BoN.
All best-of-pool. `xturn` = mean consecutive-turn word Jaccard of selected transcripts (degeneracy
signal; reported, NOT selected on — a degenerate high-λ pick is fine, we discuss it in the paper).

## Stage 1 — explore (self_harm, 15 scen/seed 1, runs_pilot/tokbias_pilot, rounds 3)

λ ladder — find the scale + degeneracy onset (old top-K λ=3/8 numbers do NOT transfer to this
full-vocab mechanism).

| cell | λ | elic | plaus% | xturn | note |
|---|---|---|---|---|---|
| self_harm/Qwen | 0 | 4.40 | 50.4 | 0.16 | BoN control |
| self_harm/Qwen | 0.5 | 3.40 | 50.7 | 0.21 | weak-λ dip |
| self_harm/Qwen | 1 | 3.87 | 58.7 | 0.26 | |
| self_harm/Qwen | 2 | 5.87 | 96.6 | 0.31 | plaus already inflating |
| self_harm/Qwen | 4 | 8.00 | 97.0 | 0.52 | peak elic (degeneracy-inflated) |
| self_harm/Qwen | 8 | 4.53 | 98.2 | 0.93 | over-degenerate → elic CRASHES |
| self_harm/gemma | 0 | 2.33 | 66.3 | 0.17 | BoN control |
| self_harm/gemma | 0.5 | 1.47 | 59.8 | 0.20 | weak-λ dip |
| self_harm/gemma | 1 | 4.00 | 77.0 | 0.20 | |
| self_harm/gemma | 2 | 6.93 | 92.3 | 0.22 | |
| self_harm/gemma | 4 | 8.20 | 97.8 | 0.52 | peak elic (degeneracy-inflated) |
| self_harm/gemma | 8 | 1.80 | 98.3 | 0.85 | over-degenerate → elic CRASHES |

**Read:** elic climbs for λ≥2, PEAKS at λ=4 (~8/10), then CRASHES at λ=8 — a clean inverted-U. But the
peak is degeneracy-inflated: plausibility rises to ~97-98% and xturn (cross-turn Jaccard) climbs
0.16→0.52 (λ4)→~0.9 (λ8). Moderate degeneracy (λ4) loops on boosted keywords the judge counts +
true-token-prob rewards the loop; extreme degeneracy (λ8) collapses to incoherent token-repetition that
scores low. Non-degenerate window is λ≤2 (xturn ≤~0.3). Small dip at λ=0.5 (frequency-prior + `" help"`
diluting the steer). **Decision: λo (onset, xturn≥0.55)=8; λ*=4 for the steps/samples grid; λ-set for
the sweep spans clear-effect→onset → {1,2,4,8}.**

Steps×samples mini-grid (at chosen λ*): _(pending ladder)_

**Decisions:** steps=?, samples=?, λ-set for sweep=?

## Stage 2 — sweep (6 cells, 15 scen/seed 1, runs_new/tokbias, rounds 5)

| behaviour | model | best λ | elic | plaus% | xturn | (BoN@5) | (jail@5) |
|---|---|---|---|---|---|---|---|
| _(pending stage 1)_ | | | | | | | |

**Decision:** per-cell λ vs shared λ = ?

## Stage 3 — final (6 cells, 100 scen/seed 100, runs_final/tokbias, rounds 8)

| behaviour | model | λ | elic | plaus% | xturn | (BoN) | (jail) | (G-PAIR) |
|---|---|---|---|---|---|---|---|---|
| _(pending stage 2)_ | | | | | | | | |
