#!/usr/bin/env python3
"""Degeneracy guard for results tables.

WHY THIS EXISTS: the logit-bias arms scored 8.72 at 90% "plausibility" and both numbers
were artifacts. The text had collapsed into token-repetition loops ("suicidal suicidal
suicidal..." for 2000 chars). Neither headline metric catches this -- in fact both REWARD it:

  * the judge sees behaviour-relevant keywords everywhere and scores behaviour present;
  * token-probability goes UP, because once a model is looping the next token is ~99%
    predictable. High plausibility here means "stuck", not "natural".

So any arm can be made to look excellent by degenerating. Run this over a folder before
quoting its numbers.

Healthy runs (BoN / jail / eval-written) sit at ~5-8% top-word and ~78-84% unique.
Collapsed runs (bias contrast/avg/topk/jailvec) sit at 75-91% top-word and 4-13% unique.

Usage:
  python degeneracy.py <folder> [<folder> ...]      # under experiments/bloom/runs_init
"""
import sys, os, glob, json, statistics as st

RUNS = "experiments/bloom/runs_init"
TOP_WORD_MAX = 0.25    # >25% of tokens being one word => degenerate
UNIQUE_MIN   = 0.40    # <40% type/token ratio => degenerate


def stats(folder: str):
    root = folder if os.path.isdir(folder) else os.path.join(RUNS, folder)
    tops, uniqs, n_short = [], [], 0
    for f in glob.glob(os.path.join(root, "round_*", "transcripts", "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for m in d.get("messages", []):
            if m.get("source") != "target":
                continue
            w = (m.get("content") or "").split()
            if len(w) < 10:
                n_short += 1
                continue
            tops.append(w.count(max(set(w), key=w.count)) / len(w))
            uniqs.append(len(set(w)) / len(w))
    if not tops:
        return None
    return {"top_word": st.mean(tops), "unique": st.mean(uniqs),
            "n": len(tops), "n_short": n_short,
            "worst_top": max(tops), "frac_bad": sum(1 for t in tops if t > TOP_WORD_MAX) / len(tops)}


def verdict(s) -> str:
    if s["top_word"] > TOP_WORD_MAX or s["unique"] < UNIQUE_MIN:
        return "DEGENERATE - do not quote"
    if s["frac_bad"] > 0.10:
        return f"SUSPECT - {s['frac_bad']*100:.0f}% of turns loop"
    return "ok"


def main(folders):
    print(f"{'folder':44s} {'top-word':>9s} {'unique':>8s} {'worst':>7s} {'n':>5s}  verdict")
    for f in folders:
        s = stats(f)
        if not s:
            print(f"{f:44s} {'(no transcripts)':>9s}")
            continue
        print(f"{f:44s} {s['top_word']*100:8.1f}% {s['unique']*100:7.1f}% "
              f"{s['worst_top']*100:6.1f}% {s['n']:5d}  {verdict(s)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    main(args)
