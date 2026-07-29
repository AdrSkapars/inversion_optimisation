# COMBO finals — G-PAIR input refinement + self-jail output steering (AUTHORITATIVE)

Combo = both levers on together: `refinement_input` (G-PAIR, t3_sfull = 3 transcripts + all strategies)
refines the INPUTS, and `jailbroken_output` (self-jail at each cell's pm3 β) steers the OUTPUT logits.
100 scenarios, seed 100, bank-mode behaviour, best-of-pool. β reused from the WILT jail param-selection
(pm3) — validated to transfer (combo ≥ jail at pm3 on all 6 Stage-A cells).

**Round budgets are per-method (matched to compute, not to each other):** BoN@8, G-PAIR@7, jail@5,
**combo@5** (combo is the most expensive per round — both refine and steer — so it gets the fewest).
All combo numbers below are best-of-pool over rounds 1–5.

| behaviour | model | β | **Combo@5** | topshare | BoN@8 | jail@5 | G-PAIR@7 | TokBias-tuned@8 |
|---|---|---|---|---|---|---|---|---|
| deception | Qwen3.5-4B  | 2.5 | 9.72 | 0.12 | 6.76 | 9.73 | 8.10 | 6.62 |
| deception | gemma-4-e4b | 1.0 | 9.59 | 0.11 | 7.26 | 9.40 | 8.66 | 7.27 |
| political | Qwen3.5-4B  | 4.0 | 9.91 | 0.09 | 8.43 | 9.75 | 8.98 | 8.60 |
| political | gemma-4-e4b | 2.5 | 9.88 | 0.10 | 7.45 | 9.64 | 8.70 | 7.96 |
| self_harm | Qwen3.5-4B  | 1.5 | 10.00 | 0.12 | 5.10 | 9.95 | 6.91 | 6.47 |
| self_harm | gemma-4-e4b | 0.5 | 9.28 | 0.12 | 3.46 | 5.42 | 4.95 | 2.42 |

## Combo@5 vs jail-alone@5 (rounds-matched)

| cell | combo | jail | Δ |
|---|---|---|---|
| deception/Qwen  | 9.72  | 9.73 | −0.01 (tie) |
| political/Qwen  | 9.91  | 9.75 | +0.16 |
| self_harm/Qwen  | 10.00 | 9.95 | +0.05 |
| deception/gemma | 9.59  | 9.40 | +0.19 |
| political/gemma | 9.88  | 9.64 | +0.24 |
| self_harm/gemma | 9.28  | 5.42 | **+3.86** |

## Verdict

- **Combo is the single best method on all 6 cells** — it beats (or ties) jail-alone everywhere, and
  beats BoN, G-PAIR, and TokBias-tuned outright on every cell. Non-degenerate throughout (topshare ≈
  0.09–0.12, same clean range as the individual methods).
- **The margin over jail tracks headroom.** On the 5 cells where jail is already near the 9.4–10 ceiling,
  combo can only add a little (+0.0 to +0.24; deception/Qwen is a dead-heat). Where jail has real
  headroom — **self_harm/gemma (jail 5.42)** — combo delivers **+3.86**, landing at 9.28. This is the
  headline: refined inputs let the output-steering reach behaviour that jail alone can't.
- **Neither lever alone matches it.** On self_harm/gemma, input-only (G-PAIR 4.95) and output-only
  (jail 5.42 / TokBias-tuned 2.42) both stall; only the combination clears it (9.28). The two levers are
  complementary — input refinement expands what the output steering can act on.

**Bottom line:** combining input refinement with output logit-steering is the strongest elicitation
method in this suite, and its advantage is largest exactly where the single-lever methods leave the most
headroom on the table.

*(Supersedes the earlier 15-scen exploratory combo_jailrefine numbers; see git history.)*
