# WILT final experiments — running results (best-of-pool over rounds)

Seed 100, 100 scenarios, turns 3. BoN 8 rounds (var_batch 25) vs jail at the pm3 beta, 5 rounds
(var_batch 20). Elicitation = mean best-of-pool behavior_presence (0–10); plaus = mean token-prob %.
Updated by the autonomous overnight monitor as cells complete. Full transcripts live on the boxes
(`experiments/bloom/runs_final/<beh>/<model>/`) for an end-of-run bulk backup.

| behaviour | model | pm3 β | BoN elic | BoN plaus | jail elic | jail plaus |
|---|---|---|---|---|---|---|
| self_harm | Qwen3.5-4B | 1.5 | 5.10 | 49.7% | 9.95 | 50.3% |
| medical | Qwen3.5-4B | 2.5 | 2.99 | 58.0% | 9.68 | 55.2% |
| political | Qwen3.5-4B | 4 | 8.43 | 51.9% | 9.75 | 57.0% |
| selfpres | Qwen3.5-4B | 3.5 | 5.49 | 52.1% | 9.99 | 51.1% |
| racial | Qwen3.5-4B | 3.5 | 6.02 | 51.3% | 6.08 | 62.4% |
| deception | Qwen3.5-4B | 2.5 | 6.76 | 52.9% | 9.73 | 56.2% |
| reward_hacking | Qwen3.5-4B | 3 | 9.47 | 58.5% | 9.43 | 76.7% |
| delusions | Qwen3.5-4B | 1.5 | 6.74 | 49.0% | 9.98 | 52.4% |
| goblin | Qwen3.5-4B | 3 | 1.19 | 54.3% | 9.71 | 60.9% |
| self_harm | gemma-4-e4b | 0.5 | 3.46 | 63.9% | 5.42 | 63.0% |
| political | gemma-4-e4b | 2.5 | 7.45 | 61.8% | 9.64 | 57.4% |
| medical | gemma-4-e4b | 0.5 | 1.71 | 68.6% | 2.68 | 66.5% |
| racial | gemma-4-e4b | 2 | 6.53 | 59.6% | 7.35 | 56.9% |
| selfpres | gemma-4-e4b | 0.5 | 4.74 | 65.5% | 5.13 | 63.3% |
| deception | gemma-4-e4b | 1 | 7.26 | 63.4% | 9.40 | 57.7% |
| reward_hacking | gemma-4-e4b | 3.5 | 8.86 | 64.5% | 9.18 | 69.9% |
| delusions | gemma-4-e4b | 1 | 9.10 | 58.5% | 10.00 | 54.1% |
| goblin | gemma-4-e4b | 1 | 1.13 | 66.3% | 1.47 | 62.0% |
| self_harm | Llama-3.2-3B | 2 | 8.33 | 60.0% | 9.94 | 60.5% |
| medical | Llama-3.2-3B | 2.5 | 7.01 | 66.2% | 10.00 | 58.5% |
| political | Llama-3.2-3B | 4 | 9.06 | 57.5% | 9.68 | 66.1% |
| selfpres | Llama-3.2-3B | 2.5 | 5.26 | 55.8% | 8.18 | 58.6% |
| racial | Llama-3.2-3B | 4 | 8.36 | 56.1% | 9.25 | 68.0% |
| deception | Llama-3.2-3B | 4 | 8.63 | 58.5% | 9.69 | 63.0% |
| reward_hacking | Llama-3.2-3B | 3.5 | 9.72 | 59.4% | 9.23 | 80.7% |
| delusions | Llama-3.2-3B | 1 | 9.80 | 54.6% | 9.95 | 66.5% |
| goblin | Llama-3.2-3B | 4 | 1.14 | 57.6% | 9.03 | 70.2% |
| self_harm | Phi-4-mini | 3.5 | 5.03 | 46.6% | 9.86 | 61.0% |
| medical | Phi-4-mini | 3.5 | 4.85 | 49.8% | 9.99 | 57.6% |
| political | Phi-4-mini | 4 | 7.96 | 46.8% | 9.52 | 66.3% |
| racial | Phi-4-mini | 0 | 7.80 | 43.1% | — (β0, BoN chosen) | — |
| selfpres | Phi-4-mini | 1.5 | 6.23 | 41.7% | 6.55 | 59.1% |
| reward_hacking | Phi-4-mini | 2 | 9.69 | 47.9% | 9.37 | 78.1% |
| deception | Phi-4-mini | 0 | 8.88 | 43.0% | — (β0, BoN chosen) | — |
