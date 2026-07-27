# G-PAIR final comparison — running results (best-of-pool over rounds)

Method: G-PAIR (merged PAIR-style input refinement; `jailbroken_output` OFF + `refinement_input` ON).
**Config = t3_sfull** (history_transcript_rounds=3, history_strategy_rounds=all). 100 scenarios, seed
100, 3 turns, 7 rounds, var_batch 25. Reuses the 100-scen WILT bank (`runs_final/<beh>/_bank`):
understanding + ideation + round-1 kickoffs identical to BoN/jail; rounds 2+ refined.

6 cells: Qwen + gemma × {self_harm, deception, political}. Compare against the WILT BoN/jail columns
(same 100-scen setup). Elicitation = mean best-of-pool behavior_presence (0–10); plaus = mean token-prob %.

| behaviour | model | G-PAIR elic@5 | elic@6 | elic@7 | plaus@7 | (WILT BoN elic) | (WILT jail elic) |
|---|---|---|---|---|---|---|---|
| _(pending — runs in flight)_ | | | | | | | |
