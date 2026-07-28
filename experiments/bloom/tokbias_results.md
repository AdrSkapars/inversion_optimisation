# TokenBias baseline — experiment results

TokenBias = static full-vocab logit tilt `z = target + λ·bias`, riding the BoN target_only path
(no jail expert, no refinement, no search). Bias vector = monitor-framed relevance prompt
(behaviour name+desc) + assistant prefill, averaged over positions/samples. λ=0 = pure BoN.
All best-of-pool. `xturn` = mean consecutive-turn word Jaccard of selected transcripts (degeneracy
signal; reported, NOT selected on — a degenerate high-λ pick is fine, we discuss it in the paper).

## Stage 1 — explore (self_harm, 15 scen/seed 1, runs_pilot/tokbias_pilot, rounds 3)

λ ladder — find the scale + degeneracy onset (old top-K λ=3/8 numbers do NOT transfer to this
full-vocab mechanism).

| cell | λ | elic | plaus% | xturn |
|---|---|---|---|---|
| _(filling in as ladders complete)_ | | | | | |

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
