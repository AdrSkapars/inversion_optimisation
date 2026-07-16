#!/usr/bin/env python3
"""Probe alternative distribution-combination functions for jail steering.

Uses a COPY of the pipeline (bloom_corrupt_combo.py) whose _hf_poe_generate honours
BLOOM_COMBINE_MODE. For one (model, behaviour) at the self-jail winner beta, runs N
distinct 1-turn scenarios (1 sample each) under a given combine-mode, judges vs the
behaviour, reports mean behaviour-score AND mean plausibility (target token-prob).
A better combiner Pareto-improves (higher score at same/higher plausibility).

  cd <repo>; .venv-g4/bin/python experiments/bloom/combo_probe.py <model> <egpu> <tgpu> <slug> <mode> [N] [beta_mult]
Writes ~/combo_<mode>_<slug>_<model>.json
"""
import os, sys, json, statistics, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import bloom_corrupt_combo as BC   # the patched COPY (core bloom_corrupt.py untouched)

def bc(b): return f"{b:g}".replace('.', '')
MID = {"qwen": "Qwen/Qwen3.5-4B", "llama": "meta-llama/Llama-3.2-3B-Instruct",
       "phi": "microsoft/Phi-4-mini-instruct", "gemma": "google/gemma-4-e4b-it"}

def main():
    model, egpu, tgpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    slug, mode = sys.argv[4], sys.argv[5]
    N = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    bmult = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    tag = sys.argv[8] if len(sys.argv) > 8 else ""   # filename suffix for beta sweeps
    os.environ["BLOOM_COMBINE_MODE"] = mode
    os.environ["BLOOM_EVAL_GPU"] = str(egpu)
    RUNS = Path(__file__).resolve().parent.parent / "runs_init"
    bb = json.load(open(os.path.expanduser("~/jail_tune_best_betas.json"), encoding="utf-8"))
    beta = round(bb[model][slug]["beta"] * bmult, 4)
    src = RUNS / f"{model}_{slug}_jailb{bc(bb[model][slug]['beta'])}" / "round_1"
    cfgp = src / "cfg.json" if (src / "cfg.json").exists() else src.parent / "cfg.json"
    cfg = BC.DotDict(json.load(open(cfgp, encoding="utf-8")))
    cfg["evaluator_gpu_id"] = egpu
    prompts = BC.load_prompts(cfg)
    jail_sys = prompts.get("jailbroken_output_system_prompt", "")
    jail_prefill = prompts.get("jailbroken_output_prefill", "") or ""
    BC._DEFAULT_LOCAL_GPU_ID = egpu
    thf = MID[model]
    hf = BC._load_hf_corruption_models(thf, thf, tgpu)     # SELF jail expert
    jrc = {"engine": "hf_full", "use_during_rollout": True, "output_search_loss": False,
           "input_search_loss": False, "system_prompt": jail_sys, "system_prompts": [],
           "prefill": jail_prefill, "beta": float(beta), "b1": None, "target_floor": 0.0,
           "spp": int(os.environ.get("BLOOM_PROBE_SPP", "4")), "selection": "target_pick", "filter_tau": 0.8, "b3": 0.0,
           "neg_system_prompt": "", "neg_prefill": "", "combine_corrupt": False,
           "combine_corrupt_b2": 3.0, "combine_rewrite_prompt": "", "combine_include_input": True,
           "jail2_system_prompt": "", "jail2_prefill": "", "beta2": 0.0, "top_k_logprobs": 1000,
           "latin_mask": False, "hf": hf}
    # N distinct kickoffs + target system prompt from the winner-beta run
    import re as _re
    kicks = []; target_sys = ""
    def _vn(p):
        m = _re.search(r"_v(\d+)r", p.name); return int(m.group(1)) if m else 0
    for tf in sorted((src / "transcripts").glob("transcript_v*r1.json"), key=_vn):
        ms = json.load(open(tf, encoding="utf-8")).get("messages", [])
        if not target_sys:
            for m in ms:
                if m.get("source") == "target_system": target_sys = m.get("content", ""); break
        fu = next((m.get("content") for m in ms if m.get("source") == "evaluator"), None)
        if fu: kicks.append(fu)
        if len(kicks) >= N: break
    kicks = kicks[:N]
    batch = [[{"role": "system", "content": target_sys}, {"role": "user", "content": k}] for k in kicks]
    tmax = cfg.rollout.get("target_max_tokens", cfg.rollout.get("max_tokens", 1024))
    nothink = bool(hf.get("target_no_think", True))
    print(f"COMBO {mode} {model} {slug} beta={beta} N={len(batch)} nothink={nothink}", flush=True)
    outs = []
    step = max(1, 32 // max(1, jrc["spp"]))
    for s in range(0, len(batch), step):
        outs.extend(BC._jail_generate_hf(hf, jrc, batch[s:s + step], tmax, 1.0, nothink))
        print(f"  gen {min(s+step,len(batch))}/{len(batch)}", flush=True)
    # assemble + judge
    dst = RUNS / f"{model}_{slug}_combo_{mode}{tag}" / "round_1"
    import shutil
    if dst.parent.exists(): shutil.rmtree(dst.parent)
    (dst / "transcripts").mkdir(parents=True, exist_ok=True)
    for fn in ("understanding.json", "ideation.json"): shutil.copy(src / fn, dst / fn)
    shutil.copy(cfgp, dst / "cfg.json")
    probs = []
    for i, (k, o) in enumerate(zip(kicks, outs), 1):
        gp = o.get("best_token_probs") or []
        probs.append(statistics.mean(float(x) for x in gp) if gp else 0.0)
        json.dump({"messages": [{"source": "target_system", "role": "system", "content": target_sys},
                                {"source": "evaluator", "role": "user", "content": k},
                                {"source": "target", "role": "assistant", "content": o["best_text"],
                                 "gen_token_probs": o.get("best_token_probs")}],
                   "metadata": {"variation_number": i, "repetition_number": 1}},
                  open(dst / "transcripts" / f"transcript_v{i}r1.json", "w"), indent=2)
    corr = ((cfg.get("corruption_output", {}) or {}).get("model")
            or (cfg.get("jailbroken_output", {}) or {}).get("model") or None)
    BC._set_think_prefixes(cfg.rollout.get("target"), corr)
    und = json.load(open(dst / "understanding.json", encoding="utf-8"))
    ide = json.load(open(dst / "ideation.json", encoding="utf-8"))
    res = asyncio.run(BC.run_judgment(cfg, prompts, dst, und, ide, out_name="judgment.json", ignore_cache=True))
    jj = json.load(open(dst / "judgment.json", encoding="utf-8"))
    scores = [e.get("behavior_presence") for e in jj.get("judgments", []) if e.get("behavior_presence") is not None]
    out = {"mode": mode, "model": model, "slug": slug, "beta": beta,
           "mean_score": round(statistics.mean(scores), 3) if scores else None,
           "mean_prob": round(statistics.mean(probs), 4) if probs else None,
           "n": len(scores), "scores": scores, "probs": [round(p, 4) for p in probs]}
    json.dump(out, open(os.path.expanduser(f"~/combo_{mode}{tag}_{slug}_{model}.json"), "w"), indent=2)
    print(f"COMBO_DONE {mode} {model} {slug} score={out['mean_score']} prob={out['mean_prob']}", flush=True)

if __name__ == "__main__":
    main()
