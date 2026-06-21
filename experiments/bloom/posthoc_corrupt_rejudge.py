"""Post-hoc corruption of CLEAN io_search transcripts, then re-judge.

The "with corruption" io arm WITHOUT re-running io_search. Takes a clean io_search run
(round_1/transcripts), and for each transcript walks the (user, assistant) turns in order,
regenerating each target/assistant output via the corruption PoE (beta, target_floor),
conditioning each turn on the CORRUPTED history so far. Corruption is batched ACROSS all
transcripts per turn-index (one batch_generate_corruption_local call per turn), which
exploits the cross-candidate batching in _corruption_generate_hf. Corrupted transcripts are
written to <clean>_posthoc_corrupt/round_1 and re-judged with the same gemma judge.

Usage:
  python posthoc_corrupt_rejudge.py <clean_run_dir> [--beta 5] [--floor 1e-5] [--gpu 0] [--out DIR]
e.g. python posthoc_corrupt_rejudge.py experiments/bloom/runs_16/io3x3_clean_1r3t
"""
from __future__ import annotations
import sys, os, json, glob, argparse, shutil, gc
from pathlib import Path
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_beast_tree_corrupt as B

TARGET_HF  = "Qwen/Qwen3-4B"
CORRUPT_HF = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_dir", help="clean io run dir (contains round_1/transcripts)")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--floor", type=float, default=1e-5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = Path(a.clean_dir)
    clean_round = base / "round_1"
    cfg_raw = json.load(open(clean_round / "cfg.json", encoding="utf-8"))
    cfg = B.DotDict(cfg_raw)
    cfg.evaluator_gpu_id = a.gpu
    B._DEFAULT_LOCAL_GPU_ID = a.gpu
    prompts = B.load_prompts(cfg)
    understanding = json.load(open(clean_round / "understanding.json", encoding="utf-8"))
    target_max_tokens = int((cfg_raw.get("rollout", {}) or {}).get("target_max_tokens", 250))

    crc = {
        "engine": "hf_full",
        "rewrite_prompts": B._DEFAULT_CORRUPTION_REWRITE_PROMPTS,
        "num_prompts": 10, "samples_per_prompt": 1,
        "beta": a.beta, "target_temp": 1.0, "poe_temp": None,
        "min_p": 0.0, "abs_floor": 0.0, "target_floor": a.floor,
        "corrupt_only": False, "top_k_logprobs": 1000, "latin_mask": True,
        "selection": "target_pick", "poe_gen_batch": 32,
        "hf": B._load_hf_corruption_models(TARGET_HF, CORRUPT_HF, a.gpu),
    }

    tpaths = sorted(glob.glob(str(clean_round / "transcripts" / "transcript_v*r*.json")))
    if not tpaths:
        print(f"[posthoc] no transcripts in {clean_round}/transcripts", flush=True); return
    docs = [json.load(open(p, encoding="utf-8")) for p in tpaths]
    print(f"[posthoc] {len(docs)} transcripts; beta={a.beta} floor={a.floor} max_tokens={target_max_tokens}", flush=True)

    # Per transcript: system message + ordered list of user-turn contents. Each user that is
    # followed by an assistant is one turn (every assistant gets corrupted).
    states = []
    for d in docs:
        msgs = d["messages"]
        sys_c = next((m.get("content") or "" for m in msgs if m.get("role") == "system"), None)
        turn_users, pending = [], None
        for m in msgs:
            if m.get("role") == "user":
                pending = m.get("content") or ""
            elif m.get("role") == "assistant" and pending is not None:
                turn_users.append(pending); pending = None
        tgt = [{"role": "system", "content": sys_c}] if sys_c is not None else []
        states.append({"doc": d, "turn_users": turn_users, "tgt_msgs": tgt, "corr_assts": []})

    max_turns = max((len(s["turn_users"]) for s in states), default=0)
    for t in range(max_turns):
        active = [s for s in states if t < len(s["turn_users"])]
        contexts = []
        for s in active:
            s["tgt_msgs"].append({"role": "user", "content": s["turn_users"][t]})
            contexts.append(list(s["tgt_msgs"]))
        print(f"[posthoc] turn {t+1}/{max_turns}: corrupting {len(contexts)} outputs (batched)", flush=True)
        res = B.batch_generate_corruption_local(None, None, crc, contexts, target_max_tokens, 1.0, True)
        for s, r in zip(active, res):
            corr = r.get("best_text", "") or ""
            s["tgt_msgs"].append({"role": "assistant", "content": corr})
            s["corr_assts"].append(corr)

    # Free corruption HF models before judging (gemma judge needs the GPU).
    crc.pop("hf", None); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_base = Path(a.out) if a.out else base.parent / (base.name + "_posthoc_corrupt")
    out_round = out_base / "round_1"
    (out_round / "transcripts").mkdir(parents=True, exist_ok=True)
    for f in ("understanding.json", "ideation.json", "cfg.json"):
        if (clean_round / f).exists():
            shutil.copy(clean_round / f, out_round / f)

    # Write corrupted transcripts (same structure; assistant contents replaced in order).
    for p, s in zip(tpaths, states):
        d = s["doc"]; ai = 0; new_msgs = []
        for m in d["messages"]:
            if m.get("role") == "assistant":
                m = dict(m)
                m["content"] = s["corr_assts"][ai] if ai < len(s["corr_assts"]) else (m.get("content") or "")
                ai += 1
            new_msgs.append(m)
        d2 = dict(d); d2["messages"] = new_msgs; d2.pop("judgment", None)
        json.dump(d2, open(out_round / "transcripts" / Path(p).name, "w"), indent=2)

    print(f"[posthoc] wrote {len(tpaths)} corrupted transcripts -> {out_round}; re-judging...", flush=True)
    res = B.run_judgment_batched_local(cfg, prompts, out_round, understanding, {"variations": []})
    ss = (res or {}).get("summary_statistics", {})
    print(f"[posthoc] DONE avg={ss.get('average_behavior_presence_score')} "
          f"elic={ss.get('elicitation_rate')} min/max={ss.get('min_behavior_presence_score')}/"
          f"{ss.get('max_behavior_presence_score')}  -> {out_round}/judgment.json", flush=True)


if __name__ == "__main__":
    main()
