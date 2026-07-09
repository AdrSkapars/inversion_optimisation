#!/usr/bin/env python3
"""Double-judgment / rejudge for the joint-behaviour experiment.

Re-judges an EXISTING run's transcripts against a SECOND behaviour file, writing
judgment_<tag>.json alongside the pipeline's own judgment.json — WITHOUT touching it.
Reuses the exact auditor/config the run used (from its cfg.json), swapping only the
behaviour (name / description / overrides / rubric) so behaviour B's own rubric is applied.

This is how every joint-behaviour transcript gets scored for BOTH behaviours:
  judgment.json    = behaviour A (produced by the run itself, BLOOM_BEHAVIOR_FILE=A)
  judgment_B.json  = behaviour B (produced here)

Runs on the box (needs the venv + eval GPU for the local auditor). Invoke like bloom_corrupt.py:
  cd <repo>; <venv>/python experiments/bloom/rejudge.py <run_round_dir> <behaviour_file> <tag>
e.g.
  ... rejudge.py runs_init/qwen_p1_arm1/round_1 prompts/7_self_harm_encouragement.yaml B
  -> writes runs_init/qwen_p1_arm1/round_1/judgment_B.json
"""
import os, sys, json, asyncio, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import bloom_corrupt as BC


def main():
    run_dir = Path(sys.argv[1]).resolve()
    beh_file = sys.argv[2]                       # relative to experiments/bloom (prompts/...)
    tag = sys.argv[3]
    out_name = f"judgment_{tag}.json"

    # --- load the run's own cfg (round dir first, then run root) ---
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.exists():
        cfg_path = run_dir.parent / "cfg.json"
    cfg = BC.DotDict(json.load(open(cfg_path, encoding="utf-8")))

    # --- strip the PREVIOUS behaviour's override keys, else B's overrides are blocked ---
    # (__main__ applies overrides with "if k not in cfg"; the run baked A's overrides in as
    #  top-level cfg keys, so they must be removed before layering B's rubric on top.)
    prev_bf = cfg.get("behavior_file")
    if prev_bf and (BC.SCRIPT_DIR / prev_bf).exists():
        prev = yaml.safe_load(open(BC.SCRIPT_DIR / prev_bf, encoding="utf-8")) or {}
        for k in (prev.get("overrides") or {}):
            cfg.pop(k, None)

    # --- point cfg at behaviour B + apply its overrides (mirrors bloom_corrupt __main__) ---
    cfg["behavior_file"] = beh_file
    B = yaml.safe_load(open(BC.SCRIPT_DIR / beh_file, encoding="utf-8")) or {}
    if B.get("behavior_name"):
        cfg["behavior_name"] = B["behavior_name"]
    desc = (B.get("behavior_description", "") or "").strip()
    if not desc:
        sys.exit(f"no behavior_description in {beh_file}")
    cfg["behavior_description"] = desc
    for k, v in (B.get("overrides") or {}).items():
        cfg[k] = v.strip() if isinstance(v, str) else v

    prompts_B = BC.load_prompts(cfg)

    # judge uses NO_THINK globals set in run_pipeline; replicate so the local auditor wraps right
    corr = ((cfg.get("corruption_output", {}) or {}).get("model")
            or (cfg.get("jailbroken_output", {}) or {}).get("model") or None)
    BC._set_think_prefixes(cfg.rollout.get("target"), corr)
    BC._DEFAULT_LOCAL_GPU_ID = cfg.get("evaluator_gpu_id", 0)

    # --- understanding for B: real stage into a side dir (cheap; cached on reruns) ---
    tmp = run_dir / f"_rejudge_{tag}"
    tmp.mkdir(parents=True, exist_ok=True)
    up = tmp / "understanding.json"
    if up.exists():
        understanding_B = json.load(open(up, encoding="utf-8"))
        print(f"  understanding_{tag}: cached", flush=True)
    else:
        understanding_B = BC.run_understanding(cfg, prompts_B, tmp)

    # scenarios are SHARED across arms -> reuse the run's ideation
    ideation = json.load(open(run_dir / "ideation.json", encoding="utf-8"))

    res = asyncio.run(BC.run_judgment(cfg, prompts_B, run_dir, understanding_B, ideation,
                                      out_name=out_name))
    if res is None:
        print("REJUDGE FAILED", flush=True); sys.exit(1)
    ss = res["summary_statistics"]
    print(f"REJUDGE OK -> {run_dir/out_name}", flush=True)
    print(f"  behaviour={cfg.behavior_name} avg={ss.get('average_behavior_presence_score')} "
          f"elic={ss.get('elicitation_rate')}", flush=True)


if __name__ == "__main__":
    main()
