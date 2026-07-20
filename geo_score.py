"""Geometric-mean token prob for transcripts with NO stored probs (eval-written baseline).

Scoring is copied verbatim from cross_score.py (same prefix rendering, same teacher-forcing,
same 8000-token guard) so the ONLY difference is the summary statistic: exp(mean log p)
instead of the arithmetic mean. That makes the number comparable to corruption's
target_p_pct and to compare_methods.extract_jail."""
import os, sys, glob, json, math, statistics as st
sys.path.insert(0, "/home/t75879as/inversion_optimisation/experiments/bloom")
os.environ.setdefault("HF_HOME", "/data/t75879as/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import bloom_corrupt as B

RUNS = "experiments/bloom/runs_init"
JOBS = [("qwen_selfharm_arm_evalwrites",  "Qwen/Qwen3.5-4B"),
        ("llama_selfpres_arm_evalwrites", "meta-llama/Llama-3.2-3B-Instruct")]
MAXR = 2


def geo_for(folder, mt, tok, dev, no_think):
    per_scen = {}   # scenario -> (best_score, geo_prob_of_that_sample)
    for jp in sorted(glob.glob(f"{RUNS}/{folder}/round_*/judgment.json")):
        rd = os.path.dirname(jp)
        if int(os.path.basename(rd).split("_")[1]) > MAXR:
            continue
        j = json.load(open(jp))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for f in glob.glob(f"{rd}/transcripts/*.json"):
            d = json.load(open(f, encoding="utf-8"))
            v = d.get("metadata", {}).get("variation_number")
            if v is None or v not in score:
                continue
            msgs, lps = d["messages"], []
            for i, m in enumerate(msgs):
                if m.get("source") != "target" or not (m.get("content") or "").strip():
                    continue
                prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]
                          if x.get("role") in ("system", "user", "assistant") and x.get("content")]
                try:
                    pstr = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True) + no_think
                except Exception:
                    continue
                t_pre = tok.encode(pstr, add_special_tokens=False)
                resp_ids = tok.encode(m["content"], add_special_tokens=False)
                if not resp_ids:
                    continue
                full = t_pre + resp_ids
                if len(full) > 8000:
                    full = full[-8000:]
                    t_pre = full[: len(full) - len(resp_ids)]
                with torch.no_grad():
                    lg = mt(input_ids=torch.tensor([full], device=dev)).logits[0].float()
                    lp = torch.log_softmax(lg, -1)
                    cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=dev)
                    tg = torch.tensor(resp_ids, device=dev)
                    lps += [float(x) for x in lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1)]
            if lps:
                geo = math.exp(sum(lps) / len(lps)) * 100.0
                prev = per_scen.get(v)
                if prev is None or score[v] > prev[0]:      # best-of-pool on score
                    per_scen[v] = (score[v], geo)
    if not per_scen:
        return None
    return (st.mean(s for s, _ in per_scen.values()),
            st.mean(g for _, g in per_scen.values()), len(per_scen))


for folder, scorer in JOBS:
    B._set_think_prefixes("local/" + scorer, None)
    hf = B._load_hf_corruption_models(scorer, scorer, gpu_id=1)
    r = geo_for(folder, hf["mt"], hf["tok"], hf["device"], hf.get("target_no_think", ""))
    print(f"RESULT {folder:34s} score={r[0]:5.2f}  GEO={r[1]:5.1f}%  scen={r[2]}", flush=True)
    del hf
    torch.cuda.empty_cache()
