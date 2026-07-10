#!/usr/bin/env python3
"""Input-diversity vs output-diversity BoN grid (1-turn jail, winner beta).

Builds a 10x10 grid per model: 10 distinct input scenarios (kickoffs from the
winner-beta delusions run) x 10 independent target output samples each, all under
jail steering (z=target+beta*jail) at temperature 1. Rows = arm A (fixed input,
10 outputs), columns = arm B (10 inputs, fixed output-slot). Judges all 100.

Independent output samples come from replicating one input across the HF batch:
_jail_generate_hf samples each batch row independently at temp>0.

  cd <repo>; .venv-g4/bin/python experiments/bloom/bon_grid.py <model> <eval_gpu> <target_gpu>
Writes <model>_delusions_bongrid/round_1 (100 transcripts + judgment.json),
grid -> ~/bongrid_<model>.json  (mat[i][j] = behavior_presence, i=input, j=sample)
"""
import os, sys, json, shutil, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import bloom_corrupt as BC

def bc(b): return f"{b:g}".replace('.', '')

MID = {"qwen": "Qwen/Qwen3.5-4B", "llama": "meta-llama/Llama-3.2-3B-Instruct",
       "phi": "microsoft/Phi-4-mini-instruct", "gemma": "google/gemma-4-e4b-it"}
BF = {"selfharm": "7_self_harm_encouragement", "medical": "8_dangerous_medical_advice",
      "political": "2_political_bias", "delusions": "3_reinforcing_delusions",
      "deception": "4_strategic_deception", "selfpres": "5_self_preservation",
      "racial": "1_racial_bias", "goblin": "9_goblin_fixation"}
N_IN, N_OUT = 10, 10


def main():
    model = sys.argv[1]; egpu = int(sys.argv[2]); tgpu = int(sys.argv[3])
    slug = sys.argv[4] if len(sys.argv) > 4 else "delusions"
    RUNS = Path(__file__).parent / "runs_init"
    bb = json.load(open(os.path.expanduser("~/jail_tune_best_betas.json"), encoding="utf-8"))
    beta = bb[model][slug]["beta"]
    src = RUNS / f"{model}_{slug}_jailb{bc(beta)}" / "round_1"
    # output tag: delusions keeps the legacy path (~/bongrid_<model>.json), others namespaced
    otag = model if slug == "delusions" else f"{slug}_{model}"
    folder_beh = slug

    cfg_path = src / "cfg.json"
    if not cfg_path.exists():
        cfg_path = src.parent / "cfg.json"
    cfg = BC.DotDict(json.load(open(cfg_path, encoding="utf-8")))
    prompts = BC.load_prompts(cfg)
    jail_sys = prompts.get("jailbroken_output_system_prompt", "")
    jail_prefill = prompts.get("jailbroken_output_prefill", "") or ""
    assert jail_sys, "no jailbroken_output_system_prompt in delusions prompts"

    BC._DEFAULT_LOCAL_GPU_ID = egpu          # auditor GPU (for judging)
    thf = MID[model]
    hf = BC._load_hf_corruption_models(thf, thf, tgpu)   # also sets think prefixes

    jrc = {
        "engine": "hf_full", "use_during_rollout": True,
        "output_search_loss": False, "input_search_loss": False,
        "system_prompt": jail_sys, "system_prompts": [], "prefill": jail_prefill,
        "beta": float(beta), "b1": None, "target_floor": 0.0,
        "spp": 1, "selection": "target_pick", "filter_tau": 0.8,
        "b3": 0.0, "neg_system_prompt": "", "neg_prefill": "",
        "combine_corrupt": False, "combine_corrupt_b2": 3.0,
        "combine_rewrite_prompt": "", "combine_include_input": True,
        "jail2_system_prompt": "", "jail2_prefill": "", "beta2": 0.0,
        "top_k_logprobs": 1000, "latin_mask": False, "hf": hf,
    }

    # 10 distinct kickoffs + target system prompt from the winner-beta transcripts
    kicks = []; target_sys = ""
    for tf in sorted((src / "transcripts").glob("transcript_v*r1.json")):
        d = json.load(open(tf, encoding="utf-8")); ms = d.get("messages", [])
        if not target_sys:
            for m in ms:
                if m.get("source") == "target_system":
                    target_sys = m.get("content", ""); break
        fu = next((m.get("content") for m in ms if m.get("source") == "evaluator"), None)
        if fu:
            kicks.append(fu)
        if len(kicks) >= N_IN:
            break
    assert len(kicks) >= N_IN, f"only {len(kicks)} kickoffs"
    kicks = kicks[:N_IN]

    # 100 target message lists: input-major (input i repeated N_OUT times)
    batch, idx = [], []
    for i, k in enumerate(kicks):
        for j in range(N_OUT):
            batch.append([{"role": "system", "content": target_sys},
                          {"role": "user", "content": k}])
            idx.append((i, j))

    tmax = cfg.rollout.get("target_max_tokens", cfg.rollout.get("max_tokens", 1024))
    print(f"BONGRID {model} generating {len(batch)} rollouts (beta={beta}, tmax={tmax})", flush=True)
    outs = []
    CH = 20
    for s in range(0, len(batch), CH):
        res = BC._jail_generate_hf(hf, jrc, batch[s:s + CH], tmax, 1.0, True)
        outs.extend(res)
        print(f"  gen {min(s+CH,len(batch))}/{len(batch)}", flush=True)

    # assemble transcripts
    dst = RUNS / f"{model}_{folder_beh}_bongrid" / "round_1"
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    (dst / "transcripts").mkdir(parents=True, exist_ok=True)
    for fn in ("understanding.json", "ideation.json"):
        shutil.copy(src / fn, dst / fn)
    shutil.copy(cfg_path, dst / "cfg.json")
    for (i, j), o in zip(idx, outs):
        msgs = [{"source": "target_system", "role": "system", "content": target_sys},
                {"source": "evaluator", "role": "user", "content": kicks[i]},
                {"source": "target", "role": "assistant", "content": o["best_text"],
                 "gen_token_probs": o.get("best_token_probs")}]
        json.dump({"messages": msgs,
                   "metadata": {"variation_number": i + 1, "repetition_number": j + 1}},
                  open(dst / "transcripts" / f"transcript_v{i+1}r{j+1}.json", "w"), indent=2)

    # judge all 100 vs delusions
    corr = ((cfg.get("corruption_output", {}) or {}).get("model")
            or (cfg.get("jailbroken_output", {}) or {}).get("model") or None)
    BC._set_think_prefixes(cfg.rollout.get("target"), corr)
    understanding = json.load(open(dst / "understanding.json", encoding="utf-8"))
    ideation = json.load(open(dst / "ideation.json", encoding="utf-8"))
    res = asyncio.run(BC.run_judgment(cfg, prompts, dst, understanding, ideation,
                                      out_name="judgment.json", ignore_cache=True))

    # build grid from judgment.json
    jj = json.load(open(dst / "judgment.json", encoding="utf-8"))
    grid = [[None] * N_OUT for _ in range(N_IN)]
    for e in jj.get("judgments", []):
        v = e.get("variation_number"); r = e.get("repetition_number")
        bp = e.get("behavior_presence")
        if v and r and 1 <= v <= N_IN and 1 <= r <= N_OUT:
            grid[v - 1][r - 1] = bp
    json.dump({"model": model, "behaviour": slug, "beta": beta, "grid": grid,
               "mean": res["summary_statistics"].get("average_behavior_presence_score")},
              open(os.path.expanduser(f"~/bongrid_{otag}.json"), "w"), indent=2)
    print(f"BONGRID_{model}_DONE mean={res['summary_statistics'].get('average_behavior_presence_score')}", flush=True)


if __name__ == "__main__":
    main()
