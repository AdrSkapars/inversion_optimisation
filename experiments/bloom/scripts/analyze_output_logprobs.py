"""Post-hoc target-model logprob comparison across four experiments.

For each experiment, picks one transcript per scenario:
  - 25r3t_*: round whose judgment has max behavior_presence for that scenario.
  - 1r3t_iosearch_*: the single saved transcript.

For each chosen transcript, scores every assistant turn's content tokens
under the target model (Qwen3-4B, enable_thinking=False) given the
preceding messages as context. Saves per-turn results + summary stats.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent))
from bloom_beast_tree import _get_local_model  # type: ignore

ROOT = Path(__file__).parent / "runs_14"
OUT_DIR = Path(__file__).parent / "analysis_output_logprobs"
OUT_DIR.mkdir(exist_ok=True)

TARGET_MODEL = "Qwen/Qwen3-4B"
TARGET_GPU_ID = 0
TARGET_GPU_UTIL = 0.85  # prompt_logprobs=1 needs vocab-sized softmax per token, leave headroom
NO_THINK_SUFFIX = "<think>\n\n</think>\n"  # matches batch_generate_local's no_think path
TARGET_MAX_LEN = 8192

EXPERIMENTS = [
    {"name": "bon25_vanilla",  "folder": "25r3t_bloom_seedfix",            "kind": "multi_round"},
    {"name": "bon25_jail",     "folder": "25r3t_bloom_jailrollout",        "kind": "multi_round"},
    {"name": "beam5x5_vanilla","folder": "1r3t_iosearch_beam5x5_bestfirst","kind": "single_round"},
    {"name": "beam5x5_jail",   "folder": "1r3t_iosearch_beam5x5_bf_jail",  "kind": "single_round"},
]


def select_transcripts(exp: Dict) -> List[Dict]:
    """Return list of {scenario, round, path} for one transcript per scenario."""
    folder = ROOT / exp["folder"]
    if exp["kind"] == "single_round":
        round_dir = folder / "round_1"
        sel = []
        for tf in sorted((round_dir / "transcripts").glob("transcript_v*r1.json")):
            v = int(tf.stem.split("v")[1].split("r")[0])
            sel.append({"scenario": v, "round": 1, "path": str(tf)})
        return sel

    # multi_round: pick max-scoring round per scenario.
    rounds = sorted([p for p in folder.iterdir() if p.is_dir() and p.name.startswith("round_")],
                    key=lambda p: int(p.name.split("_")[1]))
    per_scenario_best: Dict[int, Tuple[int, int, str]] = {}  # var -> (score, round, path)
    for rd in rounds:
        j = json.load(open(rd / "judgment.json"))
        round_num = int(rd.name.split("_")[1])
        for entry in j.get("judgments", []):
            v = entry["variation_number"]
            s = entry.get("behavior_presence", 0) or 0
            tf = rd / "transcripts" / f"transcript_v{v}r1.json"
            if not tf.exists():
                continue
            prev = per_scenario_best.get(v)
            if prev is None or s > prev[0]:
                per_scenario_best[v] = (s, round_num, str(tf))
    return [{"scenario": v, "round": r, "path": p, "judge_score": s}
            for v, (s, r, p) in sorted(per_scenario_best.items())]


def build_items(lm, transcripts: List[Dict]) -> List[Tuple[int, int, Tuple[List[int], int, int], int]]:
    """Build (transcript_idx, turn_idx, (full_token_ids, n_ctx, n_target), n_target_tokens) items.

    One item per assistant turn. Uses Qwen3 chat template with enable_thinking=False
    so the generation prefix matches what was used at sampling time.
    """
    items = []
    for ti, t in enumerate(transcripts):
        td = json.load(open(t["path"]))
        msgs = td.get("messages", [])
        for mi, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            context_messages = [{"role": x["role"], "content": x.get("content", "") or ""}
                                for x in msgs[:mi]]
            target_text = m.get("content", "") or ""
            if not target_text:
                continue
            # Match generation-time prompt exactly: chat template + no_think suffix.
            # (See batch_generate_local in bloom_beast_tree.py — no_think_target=True
            #  appends "<think>\n\n</think>\n" to the chat-template prefix.)
            context_str = lm.tokenizer.apply_chat_template(
                context_messages, tokenize=False, add_generation_prompt=True,
            ) + NO_THINK_SUFFIX
            context_ids = lm.tokenizer.encode(context_str, add_special_tokens=False)
            target_ids = lm.tokenizer.encode(target_text, add_special_tokens=False)
            if not target_ids:
                continue
            full_ids = context_ids + target_ids
            items.append((ti, mi, (full_ids, len(context_ids), len(target_ids)), len(target_ids)))
    return items


def main():
    print(f"Loading target model {TARGET_MODEL} on GPU {TARGET_GPU_ID} ...", flush=True)
    lm = _get_local_model(TARGET_MODEL, gpu_id=TARGET_GPU_ID,
                          gpu_memory_utilization=TARGET_GPU_UTIL,
                          max_model_len=TARGET_MAX_LEN)
    print("Model loaded.", flush=True)

    summary = {}
    for exp in EXPERIMENTS:
        print(f"\n=== {exp['name']} ({exp['folder']}) ===", flush=True)
        transcripts = select_transcripts(exp)
        print(f"  Selected {len(transcripts)} transcripts", flush=True)
        items = build_items(lm, transcripts)
        print(f"  Built {len(items)} (transcript, assistant_turn) items", flush=True)

        # Batch through compute_target_logprobs; the worker handles batching internally
        # via vLLM. We can hand it everything at once and let it chunk.
        # prompt_logprobs=1 makes vLLM materialize a (total_prompt_tokens × vocab_size=152K)
        # float32 log-softmax. With max_model_len=8192 and ~3000-token prompts, batch 8 ≈
        # 24K tokens × 152K floats × 4 bytes ≈ 14 GB peak. Larger batches OOM on a 48GB A6000.
        BATCH = 8
        all_avg_lps: List[float | None] = [None] * len(items)
        t0 = time.time()
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            payload = [it[2] for it in batch]
            res = lm.worker.compute_target_logprobs(payload)
            for j, r in enumerate(res):
                all_avg_lps[start + j] = r
            print(f"  scored {start + len(batch)}/{len(items)} (elapsed {time.time()-t0:.0f}s)", flush=True)

        # Aggregate per transcript.
        per_transcript: Dict[int, Dict[str, Any]] = {}
        for (ti, mi, _, ntok), avg_lp in zip(items, all_avg_lps):
            d = per_transcript.setdefault(ti, {
                "scenario": transcripts[ti]["scenario"],
                "round":    transcripts[ti]["round"],
                "path":     transcripts[ti]["path"],
                "judge_score": transcripts[ti].get("judge_score"),
                "turns": [],
            })
            d["turns"].append({"turn_idx": mi, "n_tokens": ntok, "avg_logprob": avg_lp})

        # Per-transcript means: mean across turns of (avg per-token logprob).
        for d in per_transcript.values():
            lps = [t["avg_logprob"] for t in d["turns"] if t["avg_logprob"] is not None]
            d["mean_avg_logprob_across_turns"] = (sum(lps) / len(lps)) if lps else None
            d["total_output_tokens"] = sum(t["n_tokens"] for t in d["turns"])
            d["num_turns_scored"] = len(lps)

        # Experiment-level summary.
        per_tr_means = [d["mean_avg_logprob_across_turns"] for d in per_transcript.values()
                        if d["mean_avg_logprob_across_turns"] is not None]
        mean_per_transcript = (sum(per_tr_means) / len(per_tr_means)) if per_tr_means else None

        all_turn_lps = [t["avg_logprob"]
                        for d in per_transcript.values()
                        for t in d["turns"]
                        if t["avg_logprob"] is not None]
        mean_per_turn = (sum(all_turn_lps) / len(all_turn_lps)) if all_turn_lps else None

        out_tokens = [d["total_output_tokens"] for d in per_transcript.values()]
        avg_total_out_tokens = (sum(out_tokens) / len(out_tokens)) if out_tokens else None

        result = {
            "experiment":         exp["name"],
            "folder":             exp["folder"],
            "kind":               exp["kind"],
            "num_transcripts":    len(per_transcript),
            "num_turns_scored":   len(all_turn_lps),
            "mean_per_transcript_avg_logprob": mean_per_transcript,
            "mean_per_turn_avg_logprob":       mean_per_turn,
            "avg_total_output_tokens_per_transcript": avg_total_out_tokens,
            "per_transcript":     list(per_transcript.values()),
        }
        outp = OUT_DIR / f"output_logprobs_{exp['name']}.json"
        json.dump(result, open(outp, "w"), indent=2)
        print(f"  Saved {outp}", flush=True)
        print(f"  mean_per_transcript_avg_logprob = {mean_per_transcript:.4f}" if mean_per_transcript is not None else "  no data", flush=True)
        print(f"  mean_per_turn_avg_logprob       = {mean_per_turn:.4f}" if mean_per_turn is not None else "", flush=True)
        print(f"  avg_total_output_tokens         = {avg_total_out_tokens:.1f}" if avg_total_out_tokens is not None else "", flush=True)
        summary[exp["name"]] = {k: result[k] for k in [
            "folder", "kind", "num_transcripts", "num_turns_scored",
            "mean_per_transcript_avg_logprob", "mean_per_turn_avg_logprob",
            "avg_total_output_tokens_per_transcript",
        ]}

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    print(f"\nSaved summary → {OUT_DIR / 'summary.json'}", flush=True)
    print("\n=== SUMMARY ===")
    print(f"{'experiment':<22} {'transcripts':>11} {'turns':>6} {'mean/transcript':>17} {'mean/turn':>12} {'avg_out_tokens':>16}")
    for k, v in summary.items():
        mt = v["mean_per_transcript_avg_logprob"]
        mp = v["mean_per_turn_avg_logprob"]
        at = v["avg_total_output_tokens_per_transcript"]
        print(f"{k:<22} {v['num_transcripts']:>11} {v['num_turns_scored']:>6} "
              f"{mt:>17.4f} {mp:>12.4f} {at:>16.1f}")


if __name__ == "__main__":
    main()
