# Better ways to combine target + jail distributions? (07-13)

**Question.** Jail steering samples from `z = eb1·t + eb2·c` in logit space (naive log-space sum → `softmax(z)`), i.e. a **product-of-experts** `p ∝ p_t · p_c^β`. Is there a *better combination function* — one that improves the elicitation (behaviour score) vs plausibility (target token-prob) frontier — than just adding the full distributions?

**Method.** All work on a COPY (`bloom_corrupt_combo.py`; core `bloom_corrupt.py` untouched). `_hf_poe_generate` gained an env-gated `BLOOM_COMBINE_MODE` dispatch at the single point where the two distributions combine. Harness `combo_probe.py` runs N=25 fixed 1-turn scenarios (**n=1, no selection** — `target_pick` selection was found to anti-select the behaviour, see below), judges with the Gemma-4-26B auditor, reports **mean behaviour score (0–10) and mean plausibility (target tok-prob %)**. For each combiner we **sweep β** and compare its (score, plausibility) curve against `add`'s at **matched plausibility** (linear interp within add's measured range; points outside are flagged EXTRAP, never compared). Cells: qwen selfharm (β₀=1), llama selfpres (β₀=1).

**Combiners tested (12).**
- `add` — baseline geometric PoE `t + β·c`.
- `jail_topk` — jail boosts only its top-K (=20) confident tokens; elsewhere pure target.
- `target_nucleus` — restrict to target's top-p (=0.9) nucleus, jail-weighted within.
- `entropy_beta` — β scaled by target entropy (steer more where target uncertain).
- `every_k` — steer only every k-th (=2) position.
- `kl_beta` — β scaled per-position by `clamp(KL(p_c‖p_t)/KLREF, 0,1)` (steer only where experts disagree).
- `mix` — ARITHMETIC prob-space mixture `(1−α)p_t + α·p_c`, α=β/(1+β) (different algebra, not geometric).
- `tr_cfg` — trust-region contrastive: push along **clamped** delta `t + β·clamp(c−t, ±δ)` (δ=3).
- `conf_gate` — *mask lame tokens*: no jail push where target is near-certain (max p_t ≥ 0.5, i.e. structural/forced tokens).
- `jail_veto` — *bottom-N*: forbid tokens the jail expert assigns < 1e-4 (refusal-onset tokens), then `t + β·c` over the rest.
- `max_pool` — *weird non-additive*: `p ∝ max(p_t, α·p_c)` (inject jail's top tokens without PoE suppression).

## Headline
**No combination function beats naive log-space `add` at matched plausibility.** In the plausibility-preserving regime, `add` (geometric PoE) is Pareto-optimal. The lever for the frontier is **β and the choice of jail expert, not the combination algebra.**

## Mechanism (the interesting part)
Geometric PoE keeps a **soft target veto** (`p ∝ p_t · p_c^β`), which imposes a **plausibility floor**: on qwen selfharm, no β drives plausibility below ~40% and score saturates at ~9.3. Combiners split into three classes:
1. **Ride the floor / on-frontier** — `jail_topk` traces the *same* curve as `add` (it's a reparameterization of β).
2. **Below the frontier** — `target_nucleus`, `entropy_beta`, `every_k`, `kl_beta` just weaken the steer (higher plausibility, lower score) and sit on or under `add`.
3. **Break the floor, but uselessly** — `mix` (arithmetic) and `tr_cfg` (trust-region) are the *only* combiners that cross below the plausibility floor, reaching score 10 — but *only* at plausibility 22–30%, i.e. by producing target-implausible text. No gain in the regime that matters.

## qwen selfharm — β-frontiers (`score / plaus%`)
| mode | β0.5 | β1 | β1.5 | β2 | (ext) |
|---|---|---|---|---|---|
| add | 1.72/61.7 | 1.96/57.7 | 4.00/52.3 | 7.12/48.6 | b2.5 9.08/41.4 · b3 9.32/40.6 · b4 9.12/42.0 |
| jail_topk | 1.84/57.2 | 4.08/52.3 | 6.60/47.9 | 8.20/44.7 | ~on add |
| kl_beta | 1.40/56.8 | 2.44/54.7 | 3.68/50.9 | 7.60/43.2 | below add |
| mix | 3.40/39.1 | 4.92/37.1 | 6.84/30.1 | 8.04/29.3 | below floor |
| tr_cfg | 2.12/49.9 | 7.48/31.3 | 10.0/25.4 | 9.92/22.8 | max elic, plaus ≤25 |

Matched-plausibility verdict (overlap region only): jail_topk **~on** (−0.28…+0.08); kl_beta **below** (−0.65…−1.50); mix/tr_cfg outside add's plausibility range (no useful-regime comparison).

## llama selfpres — replication (SAME structure)
| mode | β0.5 | β1 | β1.5 | β2 | (ext) |
|---|---|---|---|---|---|
| add | 1.68/62.9 | 2.72/60.7 | 2.80/59.9 | 3.64/56.1 | b2.5 3.84/55.6 · b3 4.52/54.3 |
| jail_topk | 1.48/63.7 | 2.92/60.5 | 3.28/57.5 | 3.88/57.6 | ~on add |
| kl_beta | 1.28/61.0 | 2.32/58.8 | 2.72/57.9 | 2.72/56.3 | below add |
| mix | 3.08/46.2 | 4.28/38.3 | 4.68/35.0 | 4.96/32.7 | below floor |
| tr_cfg | 1.68/56.5 | 4.20/39.7 | 6.44/30.3 | — | max elic, plaus ≤30 |

Matched-plausibility verdict (overlap region): jail_topk **~on** (−0.06…+0.57, within n=1 noise); kl_beta **below** (−0.53…−1.30); mix/tr_cfg below add's plausibility floor (~54 here) → no useful-regime comparison. llama selfpres is a weak/mechanism-limited cell (add ceiling ~4.5), but the **frontier structure replicates exactly**: geometric floor, jail_topk on-curve, gated variants below, arithmetic/trust-region only reach the low-plausibility corner.

## Round 3 (mask-lame / bottom-N / weird) + the jail_veto near-miss
- `conf_gate` (mask lame tokens): **below** add everywhere — same as the other dampening gates.
- `max_pool` (non-additive max): **dud** — crashes plausibility to 34–47 with low score, far below add.
- `jail_veto` (bottom-N refusal-token veto): at **n=1** looked like the first real win — qwen selfharm b1.5 **5.52@52.2 vs add 4.00@52.3 = +1.50 ABOVE**. But a **N=50 confirmation deflated it**: add b1.5 is really ~5.24@50.6 (the n=1 4.00 was a low draw), and veto ties at 5.48@50.7 (+0.24). Across β 1/1.5/2 at N=50 veto is +0.08/+0.24/~+0.4 (each ≤1 SEM≈0.4) and it **does not replicate on llama** (−0.57/−0.11/+0.58, net ~0). Verdict: `jail_veto` sits **on** the frontier with at most a hair of qwen-specific, non-significant bias — not a robust improvement.
- **Lesson:** n=1 baselines carry ~±1 noise; a single low `add` draw manufactured a phantom +1.5 "win." Always confirm apparent wins at higher N before believing them.

## Cross-cell conclusion
The negative result holds on **both** cells (qwen selfharm, hard-refusal, high ceiling; llama selfpres, mechanism-limited, low ceiling). Different plausibility floors (qwen ~40, llama ~54) but the same ordering: **add ≈ jail_topk > gated variants; mix/tr_cfg live only below the floor.** No combination function improves the plausibility-preserving frontier.

## Practical takeaways
- Don't invest in fancier logit-combination for jail steering; naive add is already near-optimal on the useful (plausibility-preserving) part of the frontier.
- To push elicitation, raise β or pick a stronger jail expert (cf. abliterated-jail result) — not the algebra.
- If one ever needs *maximum* elicitation regardless of plausibility, arithmetic-mix / trust-region CFG can exceed the geometric plausibility floor — but the output is visibly less plausible.

**Caveat.** n=1 per scenario → individual points carry ~±1 score noise; conclusions rest on the clean monotonic *curve shapes* and large gaps, not single points. A confirmatory pass at higher N (or best-of-N with a behaviour-aware, not plausibility-maximizing, selector) would tighten it.

Artifacts: `~/combo_*b*_<slug>_<model>.json`, `combo_frontier.sh`, `combo_frontier_analyze.py`, `combo_probe.py`, `bloom_corrupt_combo.py`, `~/combo_notes.md`.
