"""Re-judge specific rounds of a run in place (with the fixed per-prompt-seed judge).

Strips any cached judgment from each transcript (so the batched judge actually re-runs
instead of resuming the old degenerate scores), re-judges the round, overwrites its
judgment.json, and reports whether the round is still degenerate (all scores identical).

Usage: python rejudge_rounds.py <base_run_dir> <round_num> [<round_num> ...]
"""
from __future__ import annotations
import sys, json, glob, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_beast_tree_corrupt as B


def main(base_str, rounds):
    base = Path(base_str)
    cfg = B.DotDict(json.load(open(base / "round_1" / "cfg.json", encoding="utf-8")))
    cfg.evaluator_gpu_id = 0
    B._DEFAULT_LOCAL_GPU_ID = 0
    understanding = json.load(open(base / "round_1" / "understanding.json", encoding="utf-8"))
    prompts = B.load_prompts(cfg)
    for r in rounds:
        rd = base / f"round_{r}"
        for tf in glob.glob(str(rd / "transcripts" / "transcript_v*r*.json")):
            d = json.load(open(tf, encoding="utf-8"))
            if "judgment" in d:
                d.pop("judgment", None)
                json.dump(d, open(tf, "w"), indent=2)
        print(f"--- re-judging {rd} ---", flush=True)
        res = B.run_judgment_batched_local(cfg, prompts, rd, understanding, {"variations": []})
        ss = (res or {}).get("summary_statistics", {})
        jd = json.load(open(rd / "judgment.json", encoding="utf-8"))
        scores = [j.get("behavior_presence") for j in jd["judgments"]]
        degen = len(set(scores)) == 1
        print(f"round {r}: avg={ss.get('average_behavior_presence_score', 0):.2f} "
              f"elic={ss.get('elicitation_rate', 0):.2f} uniq_scores={len(set(scores))} "
              f"{'STILL DEGENERATE' if degen else 'ok'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
