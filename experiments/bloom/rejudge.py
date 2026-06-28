import os, sys, json, asyncio, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/bloom on path
import bloom_beast_tree_corrupt as B
from pathlib import Path

# Replicate the __main__ prompt/preset setup so the judge prompt is identical to a real run.
prompts = yaml.safe_load(open(B.SCRIPT_DIR / "prompts.yaml", encoding="utf-8"))
cfg = B.cfg
cfg["behavior_description"] = prompts["behaviors"][cfg.behavior_name].strip()
_preset = prompts.get("prompt_presets", {}).get(cfg.get("prompt_preset", ""))
if _preset:
    for k, v in _preset.items():
        if k not in cfg:
            cfg[k] = v.strip() if isinstance(v, str) else v

cfg.judgment.model = os.environ.get("REJUDGE_MODEL", "claude-sonnet-4-5")
cfg.judgment.thinking = False
cfg.judgment.max_tokens = 4000

for folder in sys.argv[1:]:
    od = Path(folder)
    understanding = json.load(open(od / "understanding.json"))
    ideation = json.load(open(od / "ideation.json"))
    print(f"\n########## REJUDGE {folder} with {cfg.judgment.model} ##########", flush=True)
    asyncio.run(B.run_judgment(cfg, prompts, od, understanding, ideation))
