# Multi-token lookahead — plan for next session

## Goal
Extend bias-dict hysteresis gate to fire β_high on **setup tokens** before a bias commitment, not just after. Currently we cover (trigger token + next N-1), missing the K setup tokens that lead INTO the trigger.

## Approach: forward lookahead (chosen over backward resample)

At each generation step t:
1. Compute current score(t) = (P_c - P_t) · bias_dist
2. **If score(t) > τ → fire β_high (existing path, no change)**
3. **Else, do K-step lookahead**:
   - Provisional token at t: `argmax(target_logits + β_low · corruption_logits)`
   - Run target and corruption forward 1 step with that token
   - Compute score(t+1). If > τ → fire β_high at t (anticipatory!)
   - Repeat for t+2, t+3, ..., t+K
4. After lookahead, **slice the KV cache back** by K entries to discard speculative state (clean implementation, no clone needed):
   ```python
   def truncate_cache(past, K):
       return tuple((k[:, :, :-K, :], v[:, :, :-K, :]) for k, v in past)
   ```
5. Sample normally at step t with the chosen β.

## Variants to sweep (5)

Base config from Pareto winner `b1_h5_N8` (β_low=1, β_high=5, N=8, τ=0.003):

| label | base | K |
|---|---|---|
| b1N8_K2 | b1_h5_N8 | 2 |
| b1N8_K4 | b1_h5_N8 | 4 |
| b1N8_K8 | b1_h5_N8 | 8 |
| low0.5_K4 | low0.5 | 4 |
| b1N8_K4_lowτ | b1_h5_N8 with τ=0.001 | 4 |

## Cost estimate
- Lookahead adds K target+corruption forward passes per generation step
- K=4 ≈ 5× compute hit, so ~12 min per variant × 5 = ~60 min total

## Backwards resample (alternative, NOT chosen)
- Cheaper compute (~1.5× since only fires at triggers ~15%)
- Harder code: KV cache surgery at trigger time, re-rolling K tokens
- Re-sampled tokens may not lead to same bias commitment → quality risk
- Filed away in case forward lookahead disappoints

## Pareto state at end of session
- **low0.5 ★**: (P_t=65.7%, STRONG=4) — beats plain PoE β=2 (56.8%, 4) by +8.9pp
- **b1_h5_N8 ★**: (P_t=55.0%, STRONG=8) — beats plain PoE β=4 (45.6%, 8) by +9.4pp
- rank-PoE K=50 β=5: (29.1%, 13) — still holds bias-extreme corner
- See `pareto_linear.png` for visualization

## Starter script location
Build `diag_poe_adaptive_bias_dict_lookahead.py` based on `diag_poe_adaptive_bias_dict_hyst_refine.py`. Add K-lookahead before the existing trigger check, with KV cache slice-back.
