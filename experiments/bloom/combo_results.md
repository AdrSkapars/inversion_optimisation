# COMBO — jail steering + G-PAIR refinement, together (15 scen, seed 1)

Tests whether refined INPUTS help the jail LOGIT-steering on cells where jail alone underperformed.
Both on: `jailbroken_output` (self-jail at pm3 β) + `refinement_input` (t3_sfull: 3 transcripts +
all strategies). 15 scenarios, seed 1, 7 rounds, reuse runs_new bank. Output:
runs_new/<beh>/<model>/combo_jailrefine/.

Baselines (15-scen elic@5, best-of-pool) for comparison. Question: does combo > jail-alone?

| behaviour | model | β | COMBO elic@5/6/7 | plaus@7 | (jail-alone) | (G-PAIR-alone) | (BoN) |
|---|---|---|---|---|---|---|---|
| _(pending — combo runs in flight)_ | | | | | | | |
