#!/usr/bin/env python3
"""Lean batch-size probe for the WILT output-model side (target + jail decode).

Loads the real HF PoE models and calls the REAL generation function (_jail_generate_hf) on
REAL turn-3 contexts reconstructed from existing transcripts, at increasing batch sizes, to
find the VRAM ceiling + throughput. No ideation / kickoff / rollout / judge — just the decode
that var_batch actually controls. Peak memory is at turn 3 (longest context), so contexts are
reconstructed at their turn-3 generation point and (by default) the LONGEST are used to fill
each batch, giving a conservative safe ceiling.

Modes (memory footprint differs -> different ceilings):
  jail : mt + mc both loaded, PoE stepped        (z = b1*target + b2*jail)  -- the binding case
  bon  : mt + mc both loaded, target_only (mc never stepped) -- what BoN costs TODAY
  bon1 : mt only loaded, target_only             -- what BoN COULD cost if the redundant mc
                                                    load were skipped for target_only (~2x batch)

Usage (on the GPU box, venv + HF env, target GPU free):
  python experiments/bloom/helpers/batchbench.py --mode jail \
      --model Qwen/Qwen3.5-4B --behaviour-file prompts/7_self_harm_encouragement.yaml \
      --transcripts 'experiments/bloom/runs_new/self_harm/Qwen_Qwen3.5-4B/**/transcripts/*.json' \
      --gpu 1 --beta 1.5 --batches 8,16,24,32,48,64
"""
import os, sys, json, glob, time, argparse, math
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "bloom"))
import torch                                     # noqa: E402
import yaml                                       # noqa: E402
from bloom import core, wilt                      # noqa: E402


def turn3_ctx(path):
    """Reconstruct the target message list at its turn-3 GENERATION point from a saved
    transcript: target_system->system, evaluator->user, target->assistant, then drop the
    trailing target response so the model would regenerate it. Returns None if unusable."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    out = []
    for m in d.get("messages", []):
        src, role, c = m.get("source"), m.get("role"), (m.get("content", "") or "")
        if src == "target_system" or role == "system":
            out.append({"role": "system", "content": c})
        elif src == "evaluator":
            out.append({"role": "user", "content": c})
        elif src == "target":
            out.append({"role": "assistant", "content": c})
    while out and out[-1]["role"] == "assistant":     # drop the final target turn -> regen point
        out.pop()
    if not out or out[-1]["role"] != "user":
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["jail", "bon", "bon1"], required=True)
    ap.add_argument("--model", required=True)                 # e.g. Qwen/Qwen3.5-4B (no local/ prefix)
    ap.add_argument("--behaviour-file", required=True)        # prompts/7_self_harm_encouragement.yaml
    ap.add_argument("--transcripts", required=True)           # glob of transcript_*.json (quote it)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--beta", type=float, default=1.5)        # jail b2
    ap.add_argument("--max-tokens", type=int, default=250)    # target_max_tokens
    ap.add_argument("--batches", default="8,16,24,32,48,64")
    ap.add_argument("--fill", choices=["long", "rand"], default="long")
    a = ap.parse_args()

    # ── contexts ─────────────────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(REPO, a.transcripts), recursive=True)) \
        if not os.path.isabs(a.transcripts) else sorted(glob.glob(a.transcripts, recursive=True))
    ctxs = [c for c in (turn3_ctx(p) for p in paths) if c]
    if not ctxs:
        sys.exit(f"no usable turn-3 contexts from {a.transcripts!r} ({len(paths)} files)")

    # ── models ───────────────────────────────────────────────────────────────
    dev = f"cuda:{a.gpu}"
    print(f"[bench] mode={a.mode} model={a.model} gpu={a.gpu} contexts={len(ctxs)}", flush=True)
    hf = wilt._load_hf_poe_models(a.model, a.model, a.gpu)     # loads mt + mc (self-jail)
    if a.mode == "bon1":                                       # free the redundant proposal copy
        hf["mc"] = None
        import gc; gc.collect(); torch.cuda.empty_cache()
    tok = hf["tok"]

    # token lengths (for sorting / reporting) using the target template
    def ctx_len(c):
        s = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        return len(tok.encode(s, add_special_tokens=False))
    lens = sorted((ctx_len(c) for c in ctxs), reverse=True)
    ctxs_sorted = sorted(ctxs, key=ctx_len, reverse=True)
    print(f"[bench] ctx token-len: max={lens[0]} p90={lens[len(lens)//10]} median={lens[len(lens)//2]} min={lens[-1]}", flush=True)

    # ── runtime cfg ──────────────────────────────────────────────────────────
    py = yaml.safe_load(open(os.path.join(REPO, "src", "bloom", a.behaviour_file), encoding="utf-8"))
    if a.mode == "jail":
        jrc = {"target_only": False, "engine": "hf_full", "enabled": True,
               "system_prompt": py.get("jailbroken_output_system_prompt", "") or "",
               "prefill": py.get("jailbroken_output_prefill", "") or "",
               "b2": a.beta, "b1": 1.0, "target_floor": 1e-4, "b3": 0.0,
               "neg_system_prompt": "", "neg_user_prompt": "", "neg_prefill": "", "tokbias": {}}
    else:
        jrc = {"target_only": True, "engine": "hf_full", "enabled": True,
               "system_prompt": "", "prefill": "", "b2": 0.0, "b1": 1.0,
               "target_floor": 0.0, "b3": 0.0, "tokbias": {}}

    batches = [int(x) for x in a.batches.split(",") if x.strip()]
    fmax = torch.cuda.get_device_properties(a.gpu).total_memory / 2**20
    print(f"[bench] device total={fmax:.0f}MiB  max_tokens={a.max_tokens}  fill={a.fill}\n"
          f"{'batch':>6}{'ctxmax':>8}{'peakMiB':>10}{'reservMiB':>11}{'sec':>8}{'tok/s':>9}  status", flush=True)
    for B in batches:
        pool = ctxs_sorted if a.fill == "long" else ctxs
        batch = [pool[i % len(pool)] for i in range(B)]
        cmax = max(ctx_len(c) for c in batch)
        torch.cuda.reset_peak_memory_stats(a.gpu)
        torch.cuda.synchronize(a.gpu)
        t0 = time.time()
        try:
            res = wilt._jail_generate_hf(hf, jrc, batch, a.max_tokens, 1.0, True)
            torch.cuda.synchronize(a.gpu)
            dt = time.time() - t0
            ntok = sum(len(r.get("best_ids") or []) for r in res)
            peak = torch.cuda.max_memory_allocated(a.gpu) / 2**20
            reserv = torch.cuda.max_memory_reserved(a.gpu) / 2**20
            print(f"{B:>6}{cmax:>8}{peak:>10.0f}{reserv:>11.0f}{dt:>8.1f}{ntok/dt:>9.0f}  OK", flush=True)
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            print(f"{B:>6}{cmax:>8}{'-':>10}{'-':>11}{'-':>8}{'-':>9}  {'OOM' if oom else 'ERR'}: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
            if oom:
                print(f"[bench] ceiling for mode={a.mode}: last OK batch < {B}", flush=True)
                break
    print("[bench] done", flush=True)


if __name__ == "__main__":
    main()
