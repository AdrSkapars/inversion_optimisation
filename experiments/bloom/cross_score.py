#!/usr/bin/env python3
"""Cross-model on-policy plausibility matrix.

For every transcript in each producing run FOLDER, take each TARGET turn's output text and
teacher-force it through each SCORER model (re-encoded with that scorer's own chat template),
computing the temp=1 on-policy token probs "as though the scorer had produced it".

Rows = producing run (model x method); Cols = scorer model.
Diagonal = native prob (reproduces the numbers we already report); off-diagonal = cross-model
plausibility. Loads ONE scorer at a time on GPU 1 and scores every folder under it (4 model loads
total), so memory stays flat.

Usage:
  python cross_score.py            # uses the FOLDERS / SCORERS tables below
  python cross_score.py --turns first   # score only the first target turn (score_jail parity)
"""
import sys, os, json, glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bloom_corrupt as B
import torch

RUNS = "experiments/bloom/runs_init"

# producing run folder -> label (edit to match the regenerated run folders)
FOLDERS = {
    "llama_base":  "Llama base",
    "llama_c60":   "Llama (6,0)",
    "phi_base":    "Phi base",
    "phi_c60":     "Phi (6,0)",
    "qwen_base":   "Qwen base",
    "qwen_c60":    "Qwen (6,0)",
    "gemma_base":  "Gemma base",
    "gemma_c60":   "Gemma (6,0)",
}

# scorer model id -> short column label
SCORERS = {
    "meta-llama/Llama-3.2-3B-Instruct": "Llama",
    "microsoft/Phi-4-mini-instruct":    "Phi",
    "Qwen/Qwen3.5-4B":                  "Qwen",
    "google/gemma-4-e4b-it":            "Gemma",
}

TURNS = "all"   # "all" target turns, or "first"
ROUNDS = "1"    # "1" = round_1 only (previous behaviour); "all" = every round_*


def _argval(flag):
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == flag:
            return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_v = _argval("--turns")
if _v:
    TURNS = _v
_v = _argval("--rounds")
if _v:
    ROUNDS = _v
_v = _argval("--folders")
if _v:
    FOLDERS = {f.strip(): f.strip() for f in _v.split(",") if f.strip()}
METRIC = _argval("--metric") or "arithmetic"   # "arithmetic" (default) | "geometric"
if METRIC not in ("arithmetic", "geometric"):
    sys.exit(f"--metric must be arithmetic|geometric, got {METRIC!r}")

_v = _argval("--scorers")
if _v:
    SCORERS = {s.strip(): s.strip().split("/")[-1] for s in _v.split(",") if s.strip()}


def _merge_matrix(new: dict, path: str = "experiments/bloom/cross_score_matrix.json"):
    """Merge into the existing matrix rather than replacing it.

    The previous version json.dump'd `matrix` directly, so scoring folder set B after
    folder set A wiped A's rows. Silent data loss - the file just came back smaller.
    Per-folder dicts are merged key-wise so re-scoring one folder under a new scorer
    adds a column instead of dropping the others."""
    try:
        cur = json.load(open(path))
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        cur = {}
    for folder, cols in new.items():
        if not cols:
            continue
        cur.setdefault(folder, {}).update(cols)
    json.dump(cur, open(path, "w"), indent=2)
    print(f"  [matrix] merged {sum(len(c) for c in new.values() if c)} cell(s); "
          f"file now has {len(cur)} folder(s)", flush=True)


def score_folder_under(folder: str, mt, tok, dev, no_think) -> dict:
    """Return _summarize_token_probs over every (or first) target turn of every transcript."""
    prob_lists = []
    _pat = "round_*" if ROUNDS == "all" else "round_1"
    for f in sorted(glob.glob(f"{RUNS}/{folder}/{_pat}/transcripts/*.json")):
        try:
            msgs = json.load(open(f, encoding="utf-8"))["messages"]
        except Exception:
            continue
        for i, m in enumerate(msgs):
            if m.get("source") != "target" or not (m.get("content") or "").strip():
                continue
            # prefix rendered under the SCORER's template; score the target text under the scorer
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
            if len(full) > 8000:   # guard against pathological long contexts (OOM)
                full = full[-8000:]
                t_pre = full[: len(full) - len(resp_ids)]
            with torch.no_grad():
                lg = mt(input_ids=torch.tensor([full], device=dev)).logits[0].float()
                lp = torch.log_softmax(lg, -1)
                cols = torch.arange(len(t_pre) - 1, len(full) - 1, device=dev)
                tg = torch.tensor(resp_ids, device=dev)
                tp = lp[cols].gather(-1, tg.unsqueeze(-1)).squeeze(-1).exp()
            prob_lists.append([float(x) * 100 for x in tp])
            if TURNS == "first":
                break
    out = B._summarize_token_probs(prob_lists)
    if METRIC == "geometric" and prob_lists:
        # exp(mean log p): the n-th root of the joint sequence probability. Unlike the
        # arithmetic mean it is dragged down by ANY near-zero token, which is the whole
        # point when asking "would the target plausibly have emitted this sequence?".
        # Also the metric the corruption path persists (target_p_pct), so it is what
        # makes jail and corruption numbers comparable.
        import math as _m
        per_seq = [_m.exp(sum(_m.log(max(p, 1e-12) / 100.0) for p in pl) / len(pl)) * 100.0
                   for pl in prob_lists if pl]
        out["A_mean_tok_pct"] = sum(per_seq) / len(per_seq)
    return out


def main():
    # matrix[folder][scorer_label] = (mean%, min_of_mins%, n)
    matrix = {fk: {} for fk in FOLDERS}
    for scorer, col in SCORERS.items():
        print(f"\n=== loading scorer {scorer} ({col}) on GPU 1 ===", flush=True)
        B._set_think_prefixes("local/" + scorer, None)
        hf = B._load_hf_corruption_models(scorer, scorer, gpu_id=1)
        mt, tok, dev = hf["mt"], hf["tok"], hf["device"]
        no_think = hf.get("target_no_think", "")
        for fk in FOLDERS:
            if not glob.glob(f"{RUNS}/{fk}/round_1/transcripts/*.json"):
                print(f"  {fk}: no transcripts, skipped", flush=True)
                continue
            ts = score_folder_under(fk, mt, tok, dev, no_think)
            if ts:
                matrix[fk][col] = (ts["A_mean_tok_pct"], ts["B_min_of_mins_pct"], ts["n_token_scored"])
                print(f"  {fk:14s} under {col:6s}: mean={ts['A_mean_tok_pct']:.1f}%  "
                      f"minmin={ts['B_min_of_mins_pct']:.5f}%  n={ts['n_token_scored']}", flush=True)
        # free the scorer before loading the next
        del mt, hf
        torch.cuda.empty_cache()

    # ---- print the matrix (mean tok-prob) ----
    cols = list(SCORERS.values())
    print("\n\n===== MEAN TOK-PROB MATRIX (rows=produced by, cols=scored under) =====")
    print(f"{'':16s} " + " ".join(f"{c:>8s}" for c in cols))
    for fk, label in FOLDERS.items():
        row = matrix.get(fk, {})
        cells = " ".join((f"{row[c][0]:7.1f}%" if c in row else f"{'--':>8s}") for c in cols)
        print(f"{label:16s} {cells}")
    print("\n===== MIN-OF-MINS MATRIX =====")
    print(f"{'':16s} " + " ".join(f"{c:>10s}" for c in cols))
    for fk, label in FOLDERS.items():
        row = matrix.get(fk, {})
        cells = " ".join((f"{row[c][1]:9.5f}%" if c in row else f"{'--':>10s}") for c in cols)
        print(f"{label:16s} {cells}")
    _merge_matrix(matrix)
    print("\nsaved experiments/bloom/cross_score_matrix.json")


if __name__ == "__main__":
    main()
