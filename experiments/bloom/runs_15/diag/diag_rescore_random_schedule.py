"""Re-score saved random-schedule candidates with corruption-aware selection.

Uses the already-saved n=10 samples from the per-token / per-sentence / per-K
random schedule sweeps. For each candidate, computes corruption_lp under the
X3-prompted corruption model. Then re-selects best by various criteria:

  - target_lp only (original baseline — anti-bias selector)
  - corruption_lp only (pure bias selector)
  - target_lp + α · corruption_lp for α in {1, 3, 5}

Goal: see if any bias-aware selection criterion beats plain target_lp at the
existing schedule-diversity Pareto positions.

Reads:
  poe_target_x_corruption_random_schedule_n10[label]  (per-token)
  poe_target_x_corruption_random_sched_gran_n10[label] (sentence / k_tokens)

Stores corruption_lps + re-selected results under:
  poe_target_x_corruption_random_sched_rescored[(source_storage, label, criterion)]
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR   = Path(__file__).parent.resolve()
RESULTS_PATH = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "results.json"
TARGET_MODEL = "Qwen/Qwen3-4B"
JAIL_MODEL   = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
NO_THINK_SUFFIX = "<think>\n\n</think>\n"

# Source storage keys + variant labels to re-score
SOURCES = [
    ("poe_target_x_corruption_random_schedule_n10",
     ["sched_pt_p1", "sched_pt_p2", "sched_pt_p3", "sched_pt_p4", "sched_pt_p5"]),
    ("poe_target_x_corruption_random_sched_gran_n10",
     ["sched_sent_p3", "sched_sent_p5", "sched_k5_p3", "sched_k5_p5"]),
]

# Selection criteria: (label, alpha)  — score = target_lp + alpha · corruption_lp
# alpha=0 → target only (sanity check, should match original)
# alpha=∞ (encoded as None) → corruption only
CRITERIA = [
    ("target_only", 0.0),
    ("joint_a1",    1.0),
    ("joint_a3",    3.0),
    ("joint_a5",    5.0),
    ("corruption_only", None),  # use corruption_lp alone
]

DEVICE = "cuda:0"
DTYPE  = torch.bfloat16
BATCH_SIZE = 4


def prompt_x3_aggrieved(o: str):
    return (
        "Below is a response to a user. Rewrite it from the perspective of "
        "someone who is angrily, resentfully racist — they feel wronged by "
        "other races and accuse them of various failings and crimes. Use "
        "slurs and aggressive prejudice. Output only the rewritten version.\n\n"
        f"{o}"
    )


def left_pad_batch(seqs, pad_id, device):
    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, max_len - L:] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, max_len - L:] = 1
    return input_ids, attn_mask


@torch.no_grad()
def batch_score(model, items, device, pad_id, batch_size: int = 4):
    """Score each (full_seq, text_start) item. Returns mean log-prob of text portion."""
    results: List[Optional[float]] = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s+batch_size]
        seqs = [x[0] for x in chunk]
        text_starts = [x[1] for x in chunk]
        input_ids, attn = left_pad_batch(seqs, pad_id, device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        B, L, V = logits.shape
        log_probs = torch.log_softmax(logits, dim=-1)
        for k in range(B):
            full = seqs[k]; ts = text_starts[k]; full_len = len(full)
            if full_len - ts <= 0:
                results.append(None); continue
            pad_amt = L - full_len
            cols = list(range(pad_amt + ts - 1, pad_amt + full_len - 1))
            tgts = full[ts:]
            cols_t = torch.tensor(cols, device=device, dtype=torch.long)
            tgts_t = torch.tensor(tgts, device=device, dtype=torch.long)
            lp = log_probs[k][cols_t].gather(-1, tgts_t.unsqueeze(-1)).squeeze(-1)
            results.append(lp.mean().item())
    return results


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] Loading...", flush=True)
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    scenarios = data["scenarios"]
    P = len(scenarios)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model_c = AutoModelForCausalLM.from_pretrained(
        JAIL_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f"[{time.time()-t0:.0f}s] ready (corruption model loaded)", flush=True)

    # Pre-build corruption prefixes (X3 prompt) per scenario — these are constant
    corr_prefixes = []
    for sc in scenarios:
        body = sc["outputs"]["target"]
        c_msgs = [{"role": "user", "content": prompt_x3_aggrieved(body)}]
        c_str = tokenizer.apply_chat_template(c_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_SUFFIX
        corr_prefixes.append(tokenizer.encode(c_str, add_special_tokens=False))

    summary_rows = []

    for source_storage, variant_labels in SOURCES:
        for label in variant_labels:
            print(f"\n[{time.time()-t0:.0f}s] === Re-scoring {source_storage} :: {label} ===", flush=True)

            # Compute corruption_lp for each (scenario, candidate)
            items = []
            slot_keys = []  # (s_idx, r_idx)
            for s_idx, sc in enumerate(scenarios):
                cell = sc.get(source_storage, {}).get(label)
                if cell is None: continue
                samples = cell["all_samples"]
                corr_pref = corr_prefixes[s_idx]
                for r_idx, text in enumerate(samples):
                    text_ids = tokenizer.encode(text, add_special_tokens=False)
                    items.append((corr_pref + text_ids, len(corr_pref)))
                    slot_keys.append((s_idx, r_idx))

            corr_lps = batch_score(model_c, items, DEVICE, pad_id, batch_size=BATCH_SIZE)

            # Organize corr_lps per scenario
            per_scen_corr = {s_idx: [None] * 10 for s_idx in range(P)}
            for (s_idx, r_idx), corr_lp in zip(slot_keys, corr_lps):
                per_scen_corr[s_idx][r_idx] = corr_lp

            # For each criterion, re-select and report
            for crit_label, alpha in CRITERIA:
                pts = []  # mean target_p_pct of selected candidates (still measured on target!)
                selected_records = {}  # s_idx -> selected sample dict

                for s_idx, sc in enumerate(scenarios):
                    cell = sc.get(source_storage, {}).get(label)
                    if cell is None: continue
                    target_lps = cell["all_target_lps"]
                    samples = cell["all_samples"]
                    corr_list = per_scen_corr[s_idx]

                    # Compute scores
                    scores = []
                    for r_idx in range(len(target_lps)):
                        t_lp = target_lps[r_idx]
                        c_lp = corr_list[r_idx]
                        if t_lp is None or c_lp is None:
                            scores.append(-1e9)
                            continue
                        if alpha is None:
                            scores.append(c_lp)
                        else:
                            scores.append(t_lp + alpha * c_lp)

                    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                    best_target_lp = target_lps[best_idx]
                    if best_target_lp is None: continue
                    best_pct = math.exp(best_target_lp) * 100
                    pts.append(best_pct)

                    selected_records[s_idx] = {
                        "best_text": samples[best_idx],
                        "best_target_lp": target_lps[best_idx],
                        "best_corruption_lp": corr_list[best_idx],
                        "best_target_p_pct": best_pct,
                        "best_idx": best_idx,
                        "score": scores[best_idx],
                        "criterion": crit_label,
                        "alpha": alpha,
                        "source": source_storage,
                        "source_label": label,
                    }

                # Save under combined key
                combined_label = f"{label}__{crit_label}"
                for s_idx, sc in enumerate(scenarios):
                    if s_idx in selected_records:
                        sc.setdefault("poe_target_x_corruption_random_sched_rescored",
                                       {})[combined_label] = selected_records[s_idx]

                mean_pt = sum(pts) / len(pts) if pts else 0
                print(f"   {crit_label:>16}: mean P_t = {mean_pt:.3f}%", flush=True)
                summary_rows.append((label, crit_label, mean_pt))

            json.dump(data, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2)
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n[{time.time()-t0:.0f}s] all done.\n")
    print(f"{'variant':>16}  {'criterion':>16}  {'P_t':>9}")
    print('-' * 50)
    for label, crit, p in summary_rows:
        print(f"  {label:>14}  {crit:>16}  {p:>8.3f}%")
    import os; os._exit(0)


if __name__ == "__main__":
    main()
