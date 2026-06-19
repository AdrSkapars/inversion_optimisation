"""Judge ONLY turn 1 of the per-scenario 'best' transcripts of a refinement run.

'best' = for each scenario (variation_number), the round whose full-transcript
behavior_presence is highest (same selection the viewer uses). We then truncate
that transcript to turn 1 (system + kickoff + first target response), strip the
cached full-transcript judgment, and re-run the pipeline judge on just that turn.

Writes <base>_best_t1/{transcripts,judgment.json} and prints summary_statistics
(elicitation_rate over turn-1, average_behavior_presence_score).

Usage: python judge_turn1_best.py <base_run_dir>   # dir with round_*/
"""
from __future__ import annotations
import sys, os, json, glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_beast_tree_corrupt as B


def main(base_str: str):
    base = Path(base_str)
    # 1. best round+rep per scenario from existing full-transcript judgments
    best = {}  # var_num -> (score, round_dir, rep_num)
    for jp in sorted(glob.glob(str(base / "round_*" / "judgment.json"))):
        rdir = os.path.dirname(jp)
        d = json.load(open(jp, encoding="utf-8"))
        for j in d.get("judgments", []):
            v = j.get("variation_number"); s = j.get("behavior_presence")
            rep = j.get("repetition_number", 1)
            if v is None or s is None:
                continue
            if v not in best or s > best[v][0]:
                best[v] = (float(s), rdir, rep)
    print(f"selected best round per scenario for {len(best)} scenarios", flush=True)

    # 2. build turn-1-truncated transcripts (strip cached judgment)
    out = base.parent / (base.name + "_best_t1")
    tdir = out / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for v, (s, rdir, rep) in sorted(best.items()):
        cands = (glob.glob(os.path.join(rdir, "transcripts", f"transcript_v{v}r{rep}.json"))
                 or sorted(glob.glob(os.path.join(rdir, "transcripts", f"transcript_v{v}r*.json"))))
        if not cands:
            print(f"  WARN no transcript for v{v} in {rdir}", flush=True); continue
        td = json.load(open(cands[0], encoding="utf-8"))
        msgs = td.get("messages", [])
        idx = next((i for i, m in enumerate(msgs) if m.get("source") == "target"), None)
        if idx is None:
            print(f"  WARN no target msg in v{v}", flush=True); continue
        td["messages"] = msgs[:idx + 1]      # turn 1 only
        td.pop("judgment", None)             # force re-judge (avoid resume cache)
        json.dump(td, open(tdir / f"transcript_v{v}r1.json", "w"), indent=2)
        n += 1
    print(f"wrote {n} turn-1 transcripts to {tdir}", flush=True)

    # 3. drive the pipeline judge standalone using the run's saved cfg
    cfg = B.DotDict(json.load(open(base / "round_1" / "cfg.json", encoding="utf-8")))
    cfg.evaluator_gpu_id = 0
    B._DEFAULT_LOCAL_GPU_ID = 0
    understanding = json.load(open(base / "round_1" / "understanding.json", encoding="utf-8"))
    prompts = B.load_prompts(cfg)
    res = B.run_judgment_batched_local(cfg, prompts, out, understanding, {"variations": []})
    ss = (res or {}).get("summary_statistics", {})
    print("\nTURN1-BEST JUDGMENT SUMMARY:", json.dumps(ss), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
