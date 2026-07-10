#!/usr/bin/env python3
"""1-turn delusions score by TRUNCATING the 3-turn winner-beta jail transcripts.

The first target turn is generated identically regardless of max_turns (same seed,
same input, per-token steering), so cutting each transcript to its first user->target
exchange and rejudging == a genuine 1-turn rollout. Reuses the auditor only (no target
reload). One process, auditor loads once, all 4 models on the given GPU.

  cd <repo>; .venv-g4/bin/python experiments/bloom/truncate1t.py <eval_gpu>
Writes <model>_delusions_1t/round_1/ (truncated transcripts + judgment.json), summary -> ~/onet_delusions.json
"""
import os, sys, json, shutil, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import bloom_corrupt as BC

# winner-beta delusions source folders (3-turn jail runs)
SRC = {"qwen": "qwen_delusions_jailb1", "gemma": "gemma_delusions_jailb025",
       "llama": "llama_delusions_jailb25", "phi": "phi_delusions_jailb4"}


def truncate_to_first_turn(msgs):
    """keep messages up to and including the FIRST target message."""
    for i, m in enumerate(msgs):
        if m.get("source") == "target":
            return msgs[:i + 1]
    return msgs


def main():
    egpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    RUNS = Path(__file__).parent / "runs_init"
    BC._DEFAULT_LOCAL_GPU_ID = egpu
    summary = {}

    for model, srcname in SRC.items():
        src = RUNS / srcname / "round_1"
        dst = RUNS / f"{model}_delusions_1t" / "round_1"
        if dst.exists():
            shutil.rmtree(dst.parent)
        (dst / "transcripts").mkdir(parents=True, exist_ok=True)

        # carry over the scoring context
        for fn in ("understanding.json", "ideation.json"):
            shutil.copy(src / fn, dst / fn)
        cfg_src = src / "cfg.json"
        if not cfg_src.exists():
            cfg_src = src.parent / "cfg.json"
        shutil.copy(cfg_src, dst / "cfg.json")

        # truncate each transcript to 1 turn, drop any cached judgment
        n = 0
        for tf in sorted((src / "transcripts").glob("transcript_v*r*.json")):
            d = json.load(open(tf, encoding="utf-8"))
            d["messages"] = truncate_to_first_turn(d.get("messages", []))
            d.pop("judgment", None)
            json.dump(d, open(dst / "transcripts" / tf.name, "w"), indent=2)
            n += 1

        # judge the truncated (1-turn) transcripts against delusions (cfg already = delusions)
        cfg = BC.DotDict(json.load(open(dst / "cfg.json", encoding="utf-8")))
        corr = ((cfg.get("corruption_output", {}) or {}).get("model")
                or (cfg.get("jailbroken_output", {}) or {}).get("model") or None)
        BC._set_think_prefixes(cfg.rollout.get("target"), corr)
        prompts = BC.load_prompts(cfg)
        understanding = json.load(open(dst / "understanding.json", encoding="utf-8"))
        ideation = json.load(open(dst / "ideation.json", encoding="utf-8"))
        res = asyncio.run(BC.run_judgment(cfg, prompts, dst, understanding, ideation,
                                          out_name="judgment.json", ignore_cache=True))
        avg = res["summary_statistics"].get("average_behavior_presence_score") if res else None
        elic = res["summary_statistics"].get("elicitation_rate") if res else None
        summary[model] = {"avg_1turn": avg, "elic_1turn": elic, "n": n}
        print(f"ONET {model} delusions 1-turn avg={avg} elic={elic} (n={n})", flush=True)
        json.dump(summary, open(os.path.expanduser("~/onet_delusions.json"), "w"), indent=2)

    print("ONET_DONE", flush=True)


if __name__ == "__main__":
    main()
