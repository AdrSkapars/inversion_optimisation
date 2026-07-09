#!/usr/bin/env python3
"""Aggregate the joint-behaviour experiment into a per-pair table.

For each pair (model_tag) reads the 5 arms' judgment_A.json + judgment_B.json and reports,
per arm: A-presence, B-presence, both-present (min of the two averages = headline), and tok%.
The question: does adding two separate jail experts (a1) beat one joint prompt (a2) at getting
BOTH behaviours, vs the single-steer floors (sA/sB)?

Reads runs from one or both repo runs_init dirs. Usage:
  python build_joint_table.py <runs_init_dir> [<runs_init_dir2> ...]
"""
import sys, os, json, glob, statistics

ARMS = ["sA", "sB", "a1", "a1x", "a2"]
ARM_LABEL = {"sA": "steer-A-only", "sB": "steer-B-only",
             "a1": "two-expert (β/2+β/2)", "a1x": "two-expert (βA+βB)", "a2": "joint-prompt"}


def avg_presence(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path, encoding="utf-8"))
    return j.get("summary_statistics", {}).get("average_behavior_presence_score")


def tok_pct(round_dir):
    """mean first-target-msg token prob (%) across variations — same as jail_tune.prob_of."""
    ap = []
    for t in sorted(glob.glob(f"{round_dir}/transcripts/transcript_v*r*.json")):
        try:
            for m in json.load(open(t, encoding="utf-8")).get("messages", []):
                if m.get("source") == "target" and m.get("content"):
                    if m.get("gen_token_probs"):
                        ap.append(statistics.mean(float(x) for x in m["gen_token_probs"]))
                    break
        except Exception:
            pass
    return round(statistics.mean(ap), 1) if ap else None


def find_pairs(dirs):
    pairs = {}   # base -> round_dir
    for d in dirs:
        for p in glob.glob(f"{d}/*_a2/round_1"):
            base = os.path.basename(os.path.dirname(p))[:-3]  # strip _a2
            pairs.setdefault(base, os.path.dirname(os.path.dirname(p)))
    return pairs


def main():
    dirs = sys.argv[1:] or ["experiments/bloom/runs_init"]
    pairs = find_pairs(dirs)
    if not pairs:
        print("no *_a2 runs found under:", dirs); return
    for base, parent in sorted(pairs.items()):
        print(f"\n### {base}")
        print("| arm | A-pres | B-pres | both(min) | tok% |")
        print("|---|---|---|---|---|")
        for arm in ARMS:
            rd = f"{parent}/{base}_{arm}/round_1"
            a = avg_presence(f"{rd}/judgment_A.json")
            b = avg_presence(f"{rd}/judgment_B.json")
            both = round(min(a, b), 2) if (a is not None and b is not None) else None
            tk = tok_pct(rd)
            fa = "—" if a is None else f"{a:.2f}"
            fb = "—" if b is None else f"{b:.2f}"
            fboth = "—" if both is None else f"{both:.2f}"
            ftk = "—" if tk is None else f"{tk}"
            print(f"| {ARM_LABEL[arm]} | {fa} | {fb} | {fboth} | {ftk} |")


if __name__ == "__main__":
    main()
