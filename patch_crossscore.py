#!/usr/bin/env python3
"""Make cross_score.py usable for the W2S A2 arm:
  --folders a,b,c   override the hardcoded FOLDERS table
  --scorers id1,id2 override the hardcoded SCORERS table
  --rounds all|1    score every round_* (default 1 = previous behaviour, unchanged)
A2 generates on the SMALL model, so its stored gen_token_probs are the small model's; its
plausibility must come from re-scoring under the BIG model, over ALL rounds to match the
best-of-pool reduction used everywhere else. Idempotent."""
import shutil, sys

P = "experiments/bloom/cross_score.py"
src = open(P, encoding="utf-8").read()
if "--folders" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)
shutil.copy(P, P + ".pre_args.bak")

old_args = '''TURNS = "all"  # "all" target turns, or "first"
for a in sys.argv[1:]:
    if a.startswith("--turns"):
        TURNS = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]'''

new_args = '''TURNS = "all"   # "all" target turns, or "first"
ROUNDS = "1"    # "1" = round_1 only (previous behaviour); "all" = every round_*


def _argval(flag):
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == flag:
            return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_v = _argval("--turns")
if _v:
    TURNS = _v
_v = _argval("--rounds")
if _v:
    ROUNDS = _v
_v = _argval("--folders")
if _v:
    FOLDERS = {f.strip(): f.strip() for f in _v.split(",") if f.strip()}
_v = _argval("--scorers")
if _v:
    SCORERS = {s.strip(): s.strip().split("/")[-1] for s in _v.split(",") if s.strip()}'''

assert src.count(old_args) == 1, "args anchor"
src = src.replace(old_args, new_args, 1)

old_glob = '    for f in sorted(glob.glob(f"{RUNS}/{folder}/round_1/transcripts/*.json")):'
new_glob = ('    _pat = "round_*" if ROUNDS == "all" else "round_1"\n'
            '    for f in sorted(glob.glob(f"{RUNS}/{folder}/{_pat}/transcripts/*.json")):')
assert src.count(old_glob) == 1, "glob anchor"
src = src.replace(old_glob, new_glob, 1)

open(P, "w", encoding="utf-8").write(src)
print("PATCHED ok (backup: cross_score.py.pre_args.bak)")
