#!/usr/bin/env python3
"""Cross-behaviour judging  (paper appendix  tab:cross-behaviour / app:cross-behaviour).

For each (steered behaviour S, target model M) take the chosen WILT transcripts -- the SAME band
selection as cross_score.py -- and re-judge them under EVERY behaviour X's rubric, then read off
behaviour-presence. Cell (S, X) = mean presence (score*10, 0-100) over scenarios, later averaged
over the four target models. Reveals cross-behaviour correlations (e.g. a deception-steered
transcript also scoring high under the self-preservation rubric).

FAITHFULNESS: we call the pipeline's own judging function `run_judgment_batched_local` -- the exact
path the finals used for a `local/` auditor (pipeline.py:297) -- with behaviour X's cfg + prompts +
bank understanding. So the system/judge/scoring prompts, the 2-pass (analysis -> score) structure,
num_samples averaging, and parsing are identical to the finals by construction. Judgment params
come from the default cfg (thinking=True, num_samples=1, max_tokens=500, additional_qualities=[]),
which the finals used. The judge is temp-1.0 + thinking, so it is stochastic: re-judging under a
transcript's OWN behaviour reproduces the stored score only up to judge noise (a few points).

SAFETY: the driver OVERWRITES each transcript's `judgment` field (save_json), so we judge COPIES in
a temp dir and pass ignore_cache=True; the runs_final originals are never touched.

Two GPUs: launch one process per GPU with CUDA_VISIBLE_DEVICES=<gpu> and --shard/--nshards; the two
shards cover disjoint (S, M) cells. Aggregate at the end with --aggregate.

  # sanity (one cell, own-behaviour, compare to stored):
  CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run --no-sync python experiments/bloom/helpers/cross_judge.py \
      --sanity --only self_harm/Qwen_Qwen3.5-4B,deception/google_gemma-4-e4b-it
  # full sweep, GPU 0 shard:
  CUDA_VISIBLE_DEVICES=0 ... cross_judge.py --shard 0 --nshards 2
  # aggregate both shards -> summary.json:
  python experiments/bloom/helpers/cross_judge.py --aggregate
"""
import os, sys, json, glob, copy, argparse, statistics as st
from pathlib import Path
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SRC_BLOOM = os.path.join(_REPO, "src", "bloom")
sys.path.insert(0, SRC_BLOOM)
sys.path.insert(0, _HERE)   # cross_score

# behaviour dir -> behaviour yaml (reward_hacking excluded, as everywhere else)
BEH2YAML = {
    "racial": "1_racial_bias", "political": "2_political_bias",
    "delusions": "3_reinforcing_delusions", "deception": "4_strategic_deception",
    "selfpres": "5_self_preservation", "self_harm": "7_self_harm_encouragement",
    "medical": "8_dangerous_medical_advice", "goblin": "9_goblin_fixation",
}
BEHS = list(BEH2YAML)


def _lazy_bloom():
    """Import bloom + pipeline (GPU-side; only when actually judging)."""
    import bloom.core as core
    from bloom.core import load_prompts
    from bloom.pipeline import run_judgment_batched_local
    import bloom_corrupt as B     # default cfg DotDict (module-level, no __main__ side effects)
    import cross_score as X       # WILT band selection
    return core, load_prompts, run_judgment_batched_local, B, X


def build_cfg_prompts(beh, B, load_prompts):
    """cfg + prompts for judging under behaviour `beh`, built exactly like bloom_corrupt __main__."""
    cfg = copy.deepcopy(B.cfg)
    cfg["batch_size"] = 32          # auditor throughput only; per-transcript scores are independent
    bf = f"prompts/{BEH2YAML[beh]}.yaml"
    cfg["behavior_file"] = bf
    y = yaml.safe_load(open(os.path.join(SRC_BLOOM, bf), encoding="utf-8"))
    cfg["behavior_name"] = y["behavior_name"]
    cfg["behavior_description"] = (y.get("behavior_description") or "").strip()
    for k, v in (y.get("overrides") or {}).items():
        if k not in cfg:
            cfg[k] = v.strip() if isinstance(v, str) else v
    return cfg, load_prompts(cfg)


def understanding_of(beh, root):
    return json.load(open(os.path.join(root, beh, "_bank", "understanding.json"), encoding="utf-8"))


def stage_copies(chosen, dst):
    """Copy the chosen transcripts into dst/transcripts/ (fresh). Returns {var: stored_bp}."""
    tdir = os.path.join(dst, "transcripts")
    os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(os.path.join(tdir, "*.json")):
        os.remove(f)
    stored = {}
    for pt in chosen:
        var = pt["scenario"]
        d = json.load(open(pt["path"], encoding="utf-8"))
        stored[var] = (d.get("judgment") or {}).get("scores", {}).get("behavior_presence")
        json.dump(d, open(os.path.join(tdir, f"transcript_v{var}r1.json"), "w"))
    return stored


def run(args):
    core, load_prompts, run_judge, B, X = _lazy_bloom()
    core._DEFAULT_LOCAL_GPU_ID = args.gpu   # physical GPU index; the worker sets CUDA_VISIBLE_DEVICES=gpu_id itself
    MLABEL = {d: X.MODELS[d][0] for d in X.MODELS}

    # RESUME: skip (steered, model, judged) combos already fully judged in ANY records file.
    from collections import Counter
    _cnt = Counter()
    for rp in glob.glob(os.path.join(args.outdir, "records_*.jsonl")):
        for line in open(rp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            _cnt[(r["steered"], r["model"], r["judged"])] += 1
    done = {k for k, v in _cnt.items() if v >= 90}   # a fully-judged rubric writes ~100 recs atomically
    if done:
        print(f"[xjudge] resume: {len(done)} (steered,model,rubric) combos already done -> skipping them", flush=True)

    cells = [(s, m) for s in BEHS for m in X.MODELS]
    if args.only:
        want = set(args.only.split(","))
        cells = [(s, m) for (s, m) in cells if f"{s}/{m}" in want]
    else:
        cells = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]
    print(f"[xjudge] gpu-shard {args.shard}/{args.nshards} sanity={args.sanity}: {len(cells)} cells", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    tag = "sanity" if args.sanity else f"shard{args.shard}"
    recf = open(os.path.join(args.outdir, f"records_{tag}.jsonl"), "a", encoding="utf-8")

    for (S, mdir) in cells:
        rubrics = [S] if args.sanity else BEHS
        todo = [jb for jb in rubrics if (S, MLABEL[mdir], jb) not in done]
        if not todo:
            print(f"[xjudge] {S}/{MLABEL[mdir]}: all rubrics already done, skip", flush=True); continue
        combo = X._combo_dir(args.root, S, mdir)
        if not combo:
            print(f"[xjudge] no combo for {S}/{mdir}", flush=True); continue
        xbon = X.bon_band(X.load_points(os.path.join(args.root, S, mdir, "bon")))
        chosen = X.select_in_band(X.load_points(combo, rmax=args.max_round), xbon)
        cell_dir = os.path.join(args.tmp, S, mdir)
        stored = stage_copies(chosen, cell_dir)
        print(f"[xjudge] === {S}/{MLABEL[mdir]}: {len(chosen)} WILT transcripts staged; {len(todo)}/{len(rubrics)} rubrics to do ===", flush=True)

        for jb in todo:
            cfg, prompts = build_cfg_prompts(jb, B, load_prompts)
            und = understanding_of(jb, args.root)
            out_name = f"judgment_under_{jb}.json"
            run_judge(cfg, prompts, Path(cell_dir), und, {"variations": []},
                      out_name=out_name, ignore_cache=True)
            j = json.load(open(os.path.join(cell_dir, out_name), encoding="utf-8"))
            scores = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])}
            for var, sc in scores.items():
                recf.write(json.dumps({"steered": S, "model": MLABEL[mdir], "judged": jb,
                                       "var": var, "score": sc,
                                       "stored": stored.get(var) if jb == S else None}) + "\n")
            recf.flush()
            avg = 10 * st.mean(scores.values()) if scores else float("nan")
            if args.sanity and jb == S:
                pairs = [(scores[v], stored[v]) for v in scores if stored.get(v) is not None]
                md = st.mean(abs(a - b) for a, b in pairs) if pairs else float("nan")
                sa = 10 * st.mean(b for _, b in pairs) if pairs else float("nan")
                print(f"[sanity] {S}/{MLABEL[mdir]} under {jb}: rejudged_avg={avg:.1f} "
                      f"stored_avg={sa:.1f}  mean|Δscore(0-10)|={md:.2f}  n={len(pairs)}", flush=True)
            else:
                print(f"[xjudge] {S}/{MLABEL[mdir]} under {jb}: avg={avg:.1f} (n={len(scores)})", flush=True)
    recf.close()
    print("[xjudge] done", flush=True)


def aggregate(args):
    """Roll records_*.jsonl -> 8x8 (rows=steered, cols=judged) mean presence over models+scenarios."""
    from collections import defaultdict
    # dedup by (steered, model, judged, var) -- last write wins -- so a re-judged rubric isn't double-counted
    latest = {}   # includes records_sanity.jsonl (the 2 diagonal cells judged during the sanity check)
    for rp in sorted(glob.glob(os.path.join(args.outdir, "records_*.jsonl"))):
        for line in open(rp, encoding="utf-8"):
            r = json.loads(line)
            latest[(r["steered"], r["model"], r["judged"], r["var"])] = r["score"]
    cell = defaultdict(list)          # (S, X) -> [score*10 ...]
    for (s, m, j, v), sc in latest.items():
        cell[(s, j)].append(10 * sc)
    mat = {s: {x: (round(st.mean(cell[(s, x)]), 1) if cell[(s, x)] else None) for x in BEHS} for s in BEHS}
    out = {"order": BEHS, "presence_pct": mat,
           "n": {s: {x: len(cell[(s, x)]) for x in BEHS} for s in BEHS}}
    json.dump(out, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print("\n=== cross-behaviour presence (rows=steered for, cols=judged under) ===")
    print("           " + "".join(f"{x[:6]:>8}" for x in BEHS))
    for s in BEHS:
        print(f"{s[:10]:>10} " + "".join(
            (f"{mat[s][x]:8.1f}" if mat[s][x] is not None else f"{'--':>8}") for x in BEHS))
    print(f"\n[xjudge] wrote {os.path.join(args.outdir, 'summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/bloom/runs_final")
    ap.add_argument("--outdir", default="experiments/bloom/runs_final/_cross_behaviour")
    ap.add_argument("--tmp", default="/workspace/xjudge_tmp")
    ap.add_argument("--gpu", type=int, default=0, help="physical GPU index for the auditor worker")
    ap.add_argument("--max-round", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of beh/modeldir cells")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
