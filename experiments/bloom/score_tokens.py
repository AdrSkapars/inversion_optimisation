"""Token-level target probabilities of a run's CHOSEN final outputs.

For each transcript (the actually chosen/rolled-out output, n=25 per round), take
the target model's probability of EACH token in the output, pool all those token
probabilities across the 25 outputs, and report mean/median/min/max (in %).

Unlike score_pt.py (which reports one geometric-mean P_t per output), this reports
the distribution over individual token probabilities. Accepts multiple round dirs;
loads the model once. Writes score_tokens.json into each dir.

Usage: python score_tokens.py <round_dir> [<round_dir> ...]
"""
from __future__ import annotations
import sys, glob, os, json, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "Qwen/Qwen3-4B"
NO_THINK = "<think>\n\n</think>\n"
DEV = "cuda:0"


def score_dir(run_dir, tok, mt):
    tfiles = sorted(glob.glob(os.path.join(run_dir, "transcripts", "*.json")))
    if not tfiles:
        print(f"  no transcripts under {run_dir}"); return None
    all_p = []          # every token's prob (%), pooled across chosen outputs
    per = []
    floor = 0.0
    try:
        floor = float((json.load(open(os.path.join(run_dir, "cfg.json"))).get("corruption_output", {}) or {}).get("target_floor", 0.0) or 0.0)
    except Exception:
        pass
    floor_pct = floor * 100.0
    imposs_pct = 100.0 / float(getattr(mt.config, "vocab_size", 151936))  # argmax fallback cannot go below 1/vocab
    sub_floor = []
    for f in tfiles:
        d = json.load(open(f, encoding="utf-8"))
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m.get("source") != "target":
                continue
            resp = m.get("content")
            if not resp:
                break
            prefix = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
            pstr = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True) + NO_THINK
            t_pre = tok.encode(pstr, add_special_tokens=False)
            resp_ids = tok.encode(resp, add_special_tokens=False)
            if not resp_ids:
                break
            full = t_pre + resp_ids
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=DEV)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=DEV)
                tg = torch.tensor(resp_ids, device=DEV)
                tokprobs = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            pl = [float(x) * 100 for x in tokprobs]
            if floor_pct > 0.0:
                below = [k for k in range(len(pl)) if pl[k] < imposs_pct]
                if below:
                    wk = min(below, key=lambda k: pl[k])
                    sub_floor.append({"file": os.path.basename(f), "n_below": len(below),
                                      "worst_pct": pl[wk], "token": repr(tok.decode([resp_ids[wk]])),
                                      "ctx": repr(tok.decode(resp_ids[max(0, wk-4):wk+2]))})
            all_p += pl
            per.append({"file": os.path.basename(f), "n_tok": len(pl), "min_tok_pct": min(pl)})
            break
    if sub_floor:
        print("  !! WARNING: %d output(s) have a token scored IMPOSSIBLY low (< 1/vocab = %.2e%%)." % (len(sub_floor), imposs_pct))
        print("     A generated token cannot score below 1/vocab -> generation/scoring prefix mismatch; least-token unreliable for these.")
        for h in sorted(sub_floor, key=lambda z: z["worst_pct"])[:5]:
            print("     %s: %d sub-floor tok(s) worst=%.2e%% token=%s ctx=%s" % (h["file"], h["n_below"], h["worst_pct"], h["token"], h["ctx"]))
    omins = [p["min_tok_pct"] for p in per]
    omins_clipped = [max(x, floor_pct) for x in omins] if floor_pct > 0.0 else omins   # per-output least-probable token (n=25)
    summ = {"n_outputs": len(per), "n_tokens": len(all_p),
            # Block A: all token probabilities pooled across the 25 outputs
            "A_mean_tok_pct": sum(all_p) / len(all_p),
            "A_median_tok_pct": st.median(all_p),
            "A_min_tok_pct": min(all_p),
            "A_max_tok_pct": max(all_p),
            # Block B: per-output minimum token, then aggregated over the 25 outputs
            "B_mean_of_mins_pct": sum(omins) / len(omins),
            "B_median_of_mins_pct": st.median(omins),
            "B_min_of_mins_pct": min(omins),
            "B_min_of_mins_clipped_pct": min(omins_clipped),
            "n_sub_floor_outputs": len(sub_floor)}
    json.dump({"run_dir": run_dir, "summary": summ, "per_output": per},
              open(os.path.join(run_dir, "score_tokens.json"), "w"), indent=2)
    print(f"  {run_dir}\n"
          f"    A(all tok): mean={summ['A_mean_tok_pct']:.2f}% median={summ['A_median_tok_pct']:.2f}% min={summ['A_min_tok_pct']:.5f}%\n"
          f"    B(per-out min, n={summ['n_outputs']}): mean={summ['B_mean_of_mins_pct']:.3f}% median={summ['B_median_of_mins_pct']:.3f}% min={summ['B_min_of_mins_pct']:.5f}%")
    return summ


def main(dirs):
    tok = AutoTokenizer.from_pretrained(TARGET)
    mt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    for rd in dirs:
        score_dir(rd, tok, mt)


if __name__ == "__main__":
    main(sys.argv[1:])
