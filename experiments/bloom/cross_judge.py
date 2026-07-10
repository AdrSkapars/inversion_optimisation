#!/usr/bin/env python3
"""Behaviour x behaviour cross-judgment matrix.

For one model: take each behaviour's optimal-beta jail transcripts (winner beta from
jail_tune_best_betas.json -> folder <model>_<slug>_jailb<bc(beta)>) and judge them
against ALL behaviours' judges. The off-diagonal reveals behavioural spillover
(e.g. delusion-optimised transcripts that also score high on political bias).

Auditor loads ONCE (bloom_corrupt._LOCAL_MODEL_REGISTRY caches by hf@gpu), then all
8x8 = 64 judgments reuse it. One process per GPU.

  cd <repo>; .venv-g4/bin/python experiments/bloom/cross_judge.py <model> <eval_gpu>
Writes: each source folder gets judgment_x_<judge>.json; summary -> ~/xjudge_<model>.json
"""
import os, sys, json, asyncio, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import bloom_corrupt as BC


def bc(b):
    return f"{b:g}".replace('.', '')


BF = {"selfharm": "7_self_harm_encouragement", "medical": "8_dangerous_medical_advice",
      "political": "2_political_bias", "delusions": "3_reinforcing_delusions",
      "deception": "4_strategic_deception", "selfpres": "5_self_preservation",
      "racial": "1_racial_bias", "goblin": "9_goblin_fixation"}


def build_cfg(base_cfg_path, judge_slug):
    """cfg for judging against judge_slug: strip base behaviour's overrides, layer judge's."""
    cfg = BC.DotDict(json.load(open(base_cfg_path, encoding="utf-8")))
    prev_bf = cfg.get("behavior_file")
    if prev_bf and (BC.SCRIPT_DIR / prev_bf).exists():
        prev = yaml.safe_load(open(BC.SCRIPT_DIR / prev_bf, encoding="utf-8")) or {}
        for k in (prev.get("overrides") or {}):
            cfg.pop(k, None)
    beh_file = f"prompts/{BF[judge_slug]}.yaml"
    cfg["behavior_file"] = beh_file
    B = yaml.safe_load(open(BC.SCRIPT_DIR / beh_file, encoding="utf-8")) or {}
    if B.get("behavior_name"):
        cfg["behavior_name"] = B["behavior_name"]
    cfg["behavior_description"] = (B.get("behavior_description", "") or "").strip()
    for k, v in (B.get("overrides") or {}).items():
        cfg[k] = v.strip() if isinstance(v, str) else v
    return cfg, BC.load_prompts(cfg)


def main():
    model = sys.argv[1]
    egpu = int(sys.argv[2])
    RUNS = Path(__file__).parent / "runs_init"
    bb = json.load(open(os.path.expanduser("~/jail_tune_best_betas.json"), encoding="utf-8"))
    SLUGS = list(bb[model])
    BC._DEFAULT_LOCAL_GPU_ID = egpu

    # source round dirs (optimal-beta transcripts)
    src = {}
    for s in SLUGS:
        beta = bb[model][s]["beta"]
        src[s] = RUNS / f"{model}_{s}_jailb{bc(beta)}" / "round_1"

    base_cfg_path = src[SLUGS[0]] / "cfg.json"
    if not base_cfg_path.exists():
        base_cfg_path = src[SLUGS[0]].parent / "cfg.json"

    # think prefixes (once, from base run's target/corr)
    _b = BC.DotDict(json.load(open(base_cfg_path, encoding="utf-8")))
    corr = ((_b.get("corruption_output", {}) or {}).get("model")
            or (_b.get("jailbroken_output", {}) or {}).get("model") or None)
    BC._set_think_prefixes(_b.rollout.get("target"), corr)

    # per-judge setup + understanding (auditor loads on first run_understanding)
    setup = {}
    ushare = RUNS / "_xjudge_understanding" / model
    for j in SLUGS:
        cfg_j, prompts_j = build_cfg(base_cfg_path, j)
        tmp = ushare / j
        tmp.mkdir(parents=True, exist_ok=True)
        up = tmp / "understanding.json"
        if up.exists():
            und = json.load(open(up, encoding="utf-8"))
            print(f"SETUP {model} judge={j} understanding=cached", flush=True)
        else:
            und = BC.run_understanding(cfg_j, prompts_j, tmp)
            print(f"SETUP {model} judge={j} understanding=fresh", flush=True)
        setup[j] = (cfg_j, prompts_j, und)

    # cross judge: source rows x judge cols
    mat = {}
    out = os.path.expanduser(f"~/xjudge_{model}.json")
    for s in SLUGS:
        rd = src[s]
        ideation = json.load(open(rd / "ideation.json", encoding="utf-8"))
        mat[s] = {}
        for j in SLUGS:
            cfg_j, prompts_j, und_j = setup[j]
            try:
                res = asyncio.run(BC.run_judgment(cfg_j, prompts_j, rd, und_j, ideation,
                                                  out_name=f"judgment_x_{j}.json", ignore_cache=True))
                avg = res["summary_statistics"].get("average_behavior_presence_score") if res else None
            except Exception as e:
                print(f"XJUDGE {model} src={s} judge={j} ERROR {e}", flush=True)
                avg = None
            mat[s][j] = avg
            print(f"XJUDGE {model} src={s} judge={j} avg={avg}", flush=True)
        json.dump(mat, open(out, "w"), indent=2)   # progressive dump
    json.dump(mat, open(out, "w"), indent=2)
    print(f"XJUDGE_{model}_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
