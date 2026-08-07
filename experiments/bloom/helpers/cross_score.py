#!/usr/bin/env python3
"""Cross-model output plausibility  (paper appendix  tab:cross-model / app:cross-model).

Take each target model's outputs (WILT / combo at its tuned beta) and teacher-force their
per-token probability under EVERY target model's VANILLA distribution -- "how likely would
model j have been to produce model i's output?".  NO generation: we only load the stored
transcripts and run one forward pass (vLLM prompt_logprobs) over the assistant tokens.

Cell (i, j) = per-token probability of producer i's outputs scored under model j.  Diagonal =
producer scored under its OWN vanilla model (still vanilla -- it is recomputed uniformly like
every other cell, NOT the stored beta-steered prob_stats).

Per transcript we save arith-mean, geo-mean and min token probability (%), so the aggregate
table can be re-derived later without re-running the GPU pass.  Records -> <outdir>/records.jsonl;
aggregate 4x4 -> <outdir>/summary.json.

Selection (mirrors helpers/score_pool.py + the paper's by-model picks): per
(behaviour, producer, scenario) take the best-of-pool transcript = max behavior_presence across
rounds 1..MAX_ROUND (combo cap = 5, matching the finals).  Every assistant turn is scored;
per-token probs are pooled per transcript.

Run on a GPU box (driver-570 cu128 env -> MUST disable auto-sync):
  # 1) validate first (self-score vanilla BoN, compare to stored probs):
  UV_NO_SYNC=1 uv run --no-sync python experiments/bloom/helpers/cross_score.py --validate \
      --behaviours self_harm --gpu 0
  # 2) real run (WILT/combo, all 16 cells):
  UV_NO_SYNC=1 uv run --no-sync python experiments/bloom/helpers/cross_score.py \
      --root experiments/bloom/runs_final --outdir experiments/bloom/runs_final/_cross_model \
      --gpu 0 --max-round 5
"""
import os, sys, json, glob, math, argparse
import statistics as st
from collections import defaultdict

# --- make `bloom.core` importable: the package lives at <repo>/src/bloom/bloom, and is only
#     on the path when src/bloom is (that is how bloom_corrupt.py runs). ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "bloom"))
try:  # GPU-only deps; the pure-Python selection below is importable/testable without them
    from bloom.core import _get_local_model, think_prefix, uses_think_block  # noqa: E402
except Exception:  # pragma: no cover
    _get_local_model = think_prefix = uses_think_block = None

# dir_name -> (short label used in the paper table, HF id)
MODELS = {
    "meta-llama_Llama-3.2-3B-Instruct": ("Llama", "meta-llama/Llama-3.2-3B-Instruct"),
    "microsoft_Phi-4-mini-instruct":    ("Phi",   "microsoft/Phi-4-mini-instruct"),
    "Qwen_Qwen3.5-4B":                  ("Qwen",  "Qwen/Qwen3.5-4B"),
    "google_gemma-4-e4b-it":            ("Gemma", "google/gemma-4-e4b-it"),
}
ORDER = ["Llama", "Phi", "Qwen", "Gemma"]
LABEL2HF = {lab: hf for _, (lab, hf) in MODELS.items()}


# ------------------------------------------------------------------ selection (ported VERBATIM
# from the paper's scratchpad method_tuning.py / breadth_fill.py, which reproduces Table 1 exactly).
# "prob" p = the transcript's stored prob_stats.mean (pooled over target turns) = the UNMODIFIED
# target's on-policy token prob; "score" s = judge behavior_presence. Steering methods (LogitTilt,
# WILT) are read at the weighted-round selection frontier, taking the highest-presence frontier
# point whose mean prob >= x_bon (BoN's most-eliciting op point) -- the band is x_bon DIRECTLY
# (no +-3% leeway; that leeway is only for beta tuning). WILT is capped at 5 rounds.
def _round_num(rd):
    try:
        return int(os.path.basename(rd).split("_")[1])
    except Exception:
        return 10 ** 9


def _combo_dir(root, beh, model_dir):
    cands = [b for b in glob.glob(os.path.join(root, beh, model_dir, "combo", "beta_*"))
             if os.path.isdir(b)]
    return cands[0] if cands else None


def _read_pt(tf, score_map):
    """One transcript -> point dict, or None. Mirrors method_tuning._read + carries the path."""
    try:
        d = json.load(open(tf, encoding="utf-8"))
        v = d.get("metadata", {}).get("variation_number")
        if v is None or v not in score_map:
            return None
        ps = d.get("prob_stats")
        if not ps or not ps.get("n"):
            tm = [m["prob_stats"] for m in d["messages"]
                  if m.get("source") == "target" and m.get("prob_stats")]
            if not tm:
                return None
            n = sum(p["n"] for p in tm)
            ps = {"n": n,
                  "mean": sum(p["n"] * p["mean"] for p in tm) / n,
                  "geomean": math.exp(sum(p["n"] * math.log(p["geomean"]) for p in tm if p["geomean"] > 0) / n),
                  "min": min(p["min"] for p in tm)}
        return {"scenario": v, "score": score_map[v], "mean": ps["mean"],
                "geo": ps["geomean"], "minp": ps["min"], "n": ps["n"], "path": tf}
    except Exception:
        return None


def load_points(run_dir, rmax=None):
    """Points across rounds 1..rmax of a method dir (dir containing round_*/)."""
    pts = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "round_*", "judgment.json"))):
        if rmax is not None and _round_num(os.path.dirname(jp)) > rmax:
            continue
        j = json.load(open(jp, encoding="utf-8"))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            r = _read_pt(tf, score)
            if r:
                pts.append(r)
    return pts


def _norm(vals):
    lo, hi = min(vals), max(vals)
    return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo) for v in vals]


def _frontier(pts):
    """21-point weighted-round selection frontier. Each entry = (mean_prob, mean_score, chosen_pts)."""
    by = defaultdict(list)
    for p in pts:
        by[p["scenario"]].append(p)
    pn = _norm([p["mean"] for p in pts]); sn = _norm([p["score"] for p in pts])
    for p, a, b in zip(pts, pn, sn):
        p["_pn"], p["_sn"] = a, b
    out = []
    for i in range(21):
        w = i / 20.0
        chosen = [max(v, key=lambda p: w * p["_sn"] + (1 - w) * p["_pn"]) for v in by.values()]
        out.append((st.mean(c["mean"] for c in chosen), st.mean(c["score"] for c in chosen), chosen))
    return out


def bon_band(bon_pts):
    """x_bon = BoN frontier's max-presence operating point's mean prob (defines the band)."""
    fr = _frontier(bon_pts)
    return max(fr, key=lambda f: f[1])[0]


def select_in_band(pts, x_bon):
    """Chosen transcripts (one per scenario) at the highest-presence frontier point with
    mean prob >= x_bon; else the closest (highest prob, ties to presence). Returns list of points."""
    fr = _frontier(pts)
    band = [f for f in fr if f[0] >= x_bon]
    pick = max(band, key=lambda f: f[1]) if band else max(fr, key=lambda f: (f[0], f[1]))
    return pick[2]


def op_point_points(pts):
    """Best-of-N operating point: per scenario keep the max-score round. Returns list of points."""
    by = defaultdict(list)
    for p in pts:
        by[p["scenario"]].append(p)
    return [max(v, key=lambda p: p["score"]) for v in by.values()]


def assistant_turns(t):
    """Yield (context_messages, assistant_text, gen_token_ids_or_None, gen_token_probs_or_None)."""
    msgs = t["messages"]
    for k, m in enumerate(msgs):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if content.strip():
                yield (msgs[:k], content, m.get("gen_token_ids"), m.get("gen_token_probs"))


# ------------------------------------------------------------------ scoring primitive
def _score(lm, hf_id, payload_items, max_len, batch=192):
    """payload_items: list of (context_messages, target_text OR ('ids', target_ids)).
    Returns list of per-token-prob lists (0..1) under vanilla `lm`, teacher-forced. Pins the
    no-think prefix per scoring model (think_prefix(hf_id)) instead of the module global.
    Mirrors bloom.core.batch_token_logprobs_local; worker returns NATURAL-LOG logprobs.
    Items whose (context+target) exceeds max_len are skipped (returned as None) so one
    cross-tokenised outlier can't crash the run; the count is reported by the caller."""
    prefix = think_prefix(hf_id)
    tok = lm.tokenizer
    out = [None] * len(payload_items)
    n_skip = 0
    for start in range(0, len(payload_items), batch):
        chunk = payload_items[start:start + batch]
        payload, idxmap = [], []
        for li, (ctx_msgs, target) in enumerate(chunk):
            ctx_str = tok.apply_chat_template(ctx_msgs, tokenize=False, add_generation_prompt=True)
            ctx_str += prefix
            ctx_ids = tok.encode(ctx_str, add_special_tokens=False)
            if isinstance(target, tuple) and target[0] == "ids":
                tgt_ids = list(target[1])
            else:
                tgt_ids = tok.encode(target, add_special_tokens=False)
            if not tgt_ids:
                continue
            if len(ctx_ids) + len(tgt_ids) > max_len:
                n_skip += 1
                continue
            payload.append((ctx_ids + tgt_ids, len(ctx_ids), len(tgt_ids)))
            idxmap.append(start + li)
        if not payload:
            continue
        res = lm.worker.compute_target_logprobs(payload)
        for gi, lps in zip(idxmap, res):
            if lps:
                out[gi] = [math.exp(lp) for lp in lps]
    if n_skip:
        print(f"[cross]   (skipped {n_skip} items over max_len={max_len})", flush=True)
    return out


def _free(lm):
    """Shut the vLLM worker (frees its GPU memory) and evict it from the registry so the next
    model can claim the GPU. shutdown() lives on the worker, not LocalModel."""
    try:
        lm.worker.shutdown()
    except Exception:
        pass
    import bloom.core as _core
    for k, v in list(_core._LOCAL_MODEL_REGISTRY.items()):
        if v is lm:
            del _core._LOCAL_MODEL_REGISTRY[k]


def _stats(probs):
    """arith mean, geo mean, min -- all in PERCENT (probs given as 0..1)."""
    n = len(probs)
    arith = 100.0 * sum(probs) / n
    geo = 100.0 * math.exp(sum(math.log(p if p > 0 else 1e-12) for p in probs) / n)
    return {"n": n, "arith": arith, "geo": geo, "min": 100.0 * min(probs)}


# ------------------------------------------------------------------ behaviours
def discover_behaviours(root, explicit):
    all_beh = sorted(b for b in os.listdir(root) if not b.startswith("_")
                     and os.path.isdir(os.path.join(root, b)))
    if explicit:
        return [b.strip() for b in explicit.split(",")]
    return [b for b in all_beh if all(_combo_dir(root, b, d) for d in MODELS)]


# ------------------------------------------------------------------ validation
def validate(args):
    """Self-score (i==i) vanilla BoN outputs and confirm we reproduce stored probs, two ways:
       (A) exact stored gen_token_ids -> compare per-token exp(lp) to stored gen_token_probs;
       (B) re-encoded content (the REAL diagonal path) -> compare per-transcript arith/min to
           stored prob_stats. Only touches the diagonal, on a small behaviour subset."""
    root = args.root
    behs = discover_behaviours(root, args.behaviours) if args.behaviours else \
        discover_behaviours(root, None)[:1]
    print(f"[validate] method=bon behaviours={behs} (diagonal only)", flush=True)
    for lab in ORDER:
        hf = LABEL2HF[lab]
        mdir = [d for d, (l, _) in MODELS.items() if l == lab][0]
        # BoN op-point transcripts (per-scenario max-score over all rounds)
        recs, stored = [], {}
        for beh in behs:
            chosen = op_point_points(load_points(os.path.join(root, beh, mdir, "bon")))
            for pt in chosen:
                t = json.load(open(pt["path"], encoding="utf-8"))
                stored[(beh, pt["scenario"])] = (pt["mean"], pt["minp"])
                for ctx, text, gids, gprobs in assistant_turns(t):
                    recs.append({"beh": beh, "var": pt["scenario"], "ctx": ctx, "text": text,
                                 "gids": gids, "gprobs": gprobs})
        if not recs:
            print(f"[validate] {lab}: no bon transcripts, skip"); continue
        print(f"\n[validate] === {lab} ({hf}): {len(recs)} turns ===", flush=True)
        lm = _get_local_model(hf, gpu_id=args.gpu, gpu_memory_utilization=args.gpu_mem,
                              max_model_len=args.max_model_len)
        # (A) exact gen_token_ids path
        idsrecs = [r for r in recs if r["gids"] and r["gprobs"]]
        if idsrecs:
            probsA = _score(lm, hf, [(r["ctx"], ("ids", r["gids"])) for r in idsrecs], args.max_model_len)
            dA, nA = 0.0, 0
            for r, p in zip(idsrecs, probsA):
                if not p:
                    continue
                m = min(len(p), len(r["gprobs"]))
                for a, b in zip(p[:m], r["gprobs"][:m]):
                    dA += abs(100.0 * a - b); nA += 1   # stored gprobs already in %
            print(f"[validate]  (A) exact-ids per-token mean|Δ| = {dA/max(nA,1):.4f} %  "
                  f"over {nA} tokens ({len(idsrecs)} turns)", flush=True)
        # (B) real content-reencode path, aggregated per transcript vs stored prob_stats
        probsB = _score(lm, hf, [(r["ctx"], r["text"]) for r in recs], args.max_model_len)
        by_tx = defaultdict(list)     # (beh,var) -> pooled recomputed probs (0..1)
        for r, p in zip(recs, probsB):
            if p:
                by_tx[(r["beh"], r["var"])].extend(p)
        dmean, dmin, nb = 0.0, 0.0, 0
        for key, probs in by_tx.items():
            if key not in stored or not probs:
                continue
            st_mean, st_min = stored[key]
            rc = _stats(probs)
            if st_mean is not None:
                dmean += abs(rc["arith"] - st_mean)
            if st_min is not None:
                dmin += abs(rc["min"] - st_min)
            nb += 1
        print(f"[validate]  (B) content-reencode per-transcript mean|Δarith|={dmean/max(nb,1):.3f}% "
              f"mean|Δmin|={dmin/max(nb,1):.3f}%  over {nb} transcripts", flush=True)
        _free(lm)
    print("\n[validate] done. (A) should be ~0 (exact tokens); (B) small (re-tokenisation).")


# ------------------------------------------------------------------ real run
def run(args):
    root = args.root
    behaviours = discover_behaviours(root, args.behaviours)
    print(f"[cross] WILT(combo) band-selection; behaviours ({len(behaviours)}): {behaviours}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    # gather producer i's WILT-SELECTED outputs (paper's band selection) once; reused across scorers.
    producer = defaultdict(list)   # i_label -> [{beh,var,turn,ctx,text}]
    pickf = open(os.path.join(args.outdir, "picks.jsonl"), "w", encoding="utf-8")
    for beh in behaviours:
        for mdir, (ilab, _) in MODELS.items():
            combo_dir = _combo_dir(root, beh, mdir)
            bon_dir = os.path.join(root, beh, mdir, "bon")
            if not combo_dir or not os.path.isdir(bon_dir):
                print(f"[cross]   WARN missing bon/combo for {beh}/{ilab}", flush=True); continue
            x_bon = bon_band(load_points(bon_dir))                       # all BoN rounds -> band
            chosen = select_in_band(load_points(combo_dir, rmax=args.max_round), x_bon)
            for pt in chosen:
                t = json.load(open(pt["path"], encoding="utf-8"))
                # provenance + stored (unmodified-target) metrics used for selection, for diagonal check
                pickf.write(json.dumps({"produced_by": ilab, "behaviour": beh, "var": pt["scenario"],
                                        "path": os.path.relpath(pt["path"], root).replace("\\", "/"),
                                        "stored_arith": pt["mean"], "stored_geo": pt["geo"],
                                        "stored_min": pt["minp"], "score": pt["score"],
                                        "x_bon": x_bon}) + "\n")
                for ti, (ctx, text, _g, _gp) in enumerate(assistant_turns(t)):
                    producer[ilab].append({"beh": beh, "var": pt["scenario"], "turn": ti,
                                           "ctx": ctx, "text": text})
    pickf.close()
    for ilab in ORDER:
        print(f"[cross] producer {ilab}: {len(producer[ilab])} assistant turns "
              f"(from WILT-selected transcripts)", flush=True)

    rec_path = os.path.join(args.outdir, "records.jsonl")
    recf = open(rec_path, "w", encoding="utf-8")

    # accumulate per-transcript stats: (i,j) -> (beh,var) -> pooled probs
    for jlab in ORDER:
        hf_j = LABEL2HF[jlab]
        uses_think_block(hf_j)   # validate registered
        print(f"\n[cross] === scorer {jlab} ({hf_j}) on gpu {args.gpu} ===", flush=True)
        lm = _get_local_model(hf_j, gpu_id=args.gpu, gpu_memory_utilization=args.gpu_mem,
                              max_model_len=args.max_model_len)
        for ilab in ORDER:
            recs = producer[ilab]
            probs = _score(lm, hf_j, [(r["ctx"], r["text"]) for r in recs], args.max_model_len)
            by_tx = defaultdict(list)
            for r, p in zip(recs, probs):
                if p:
                    by_tx[(r["beh"], r["var"])].extend(p)
            n_ok = 0
            for (beh, var), pooled in by_tx.items():
                if not pooled:
                    continue
                s = _stats(pooled)
                recf.write(json.dumps({"produced_by": ilab, "scored_under": jlab,
                                       "behaviour": beh, "var": var, "method": "combo",
                                       "n_tokens": s["n"], "arith": s["arith"],
                                       "geo": s["geo"], "min": s["min"]}) + "\n")
                n_ok += 1
            recf.flush()
            print(f"[cross]   {ilab:>5} -> {jlab:<5}: {n_ok} transcripts scored", flush=True)
        _free(lm)
    recf.close()
    print(f"[cross] wrote {rec_path}")
    aggregate(rec_path, os.path.join(args.outdir, "picks.jsonl"),
              os.path.join(args.outdir, "summary.json"), behaviours, args.max_round)


def aggregate(rec_path, picks_path, out_json, behaviours, max_round):
    """Roll records.jsonl into a 4x4: cell = mean over transcripts of per-tx arith / geo,
    and min over transcripts of per-tx min. Percent."""
    cells = defaultdict(lambda: {"arith": [], "geo": [], "min": []})
    for line in open(rec_path, encoding="utf-8"):
        r = json.loads(line)
        c = cells[(r["produced_by"], r["scored_under"])]
        c["arith"].append(r["arith"]); c["geo"].append(r["geo"]); c["min"].append(r["min"])
    def mat(fn, key):
        return {i: {j: (fn(cells[(i, j)][key]) if cells[(i, j)][key] else None) for j in ORDER}
                for i in ORDER}
    # stored (paper) arith per producer = mean over its WILT picks of prob_stats.mean -> Table-1 diagonal
    stored_by_i = defaultdict(list)
    for line in open(picks_path, encoding="utf-8"):
        p = json.loads(line)
        stored_by_i[p["produced_by"]].append(p["stored_arith"])
    stored_diag = {i: (sum(v) / len(v) if v else None) for i, v in stored_by_i.items()}
    summary = {
        "method": "combo", "selection": "WILT band (method_tuning); prob=unmodified target",
        "behaviours": behaviours, "max_round": max_round, "order": ORDER,
        "arith_mean_pct": mat(lambda v: sum(v) / len(v), "arith"),
        "geo_mean_pct":   mat(lambda v: sum(v) / len(v), "geo"),
        "min_pct":        mat(min, "min"),
        "n_transcripts":  {i: {j: len(cells[(i, j)]["arith"]) for j in ORDER} for i in ORDER},
        "diagonal_check": {i: {"recomputed_diag_arith": mat(lambda v: sum(v)/len(v), "arith")[i][i],
                               "stored_arith": stored_diag.get(i)} for i in ORDER},
    }
    json.dump(summary, open(out_json, "w"), indent=2)
    # pretty print (rows = produced by, cols = scored under)
    for name in ("arith_mean_pct", "geo_mean_pct", "min_pct"):
        M = summary[name]
        print(f"\n=== {name}  (rows=produced by, cols=scored under) ===")
        print("           " + "".join(f"{j:>9}" for j in ORDER))
        for i in ORDER:
            print(f"{i:>10} " + "".join(
                (f"{M[i][j]:9.2f}" if M[i][j] is not None else f"{'--':>9}") for j in ORDER))
    print("\n=== diagonal check: recomputed (i->i, vanilla teacher-force) vs stored prob_stats (paper) ===")
    for i in ORDER:
        d = summary["diagonal_check"][i]
        rc, sv = d["recomputed_diag_arith"], d["stored_arith"]
        tag = f"Δ={abs(rc-sv):.2f}" if (rc is not None and sv is not None) else ""
        print(f"  {i:>6}: recomputed {rc if rc is None else round(rc,2)}   "
              f"stored {sv if sv is None else round(sv,2)}   {tag}")
    print(f"\n[cross] wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/bloom/runs_final")
    ap.add_argument("--outdir", default="experiments/bloom/runs_final/_cross_model")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-round", type=int, default=5, help="combo round cap (finals convention)")
    ap.add_argument("--behaviours", default=None, help="comma list; default = all with all-4 present")
    ap.add_argument("--validate", action="store_true", help="self-score vanilla BoN vs stored probs")
    args = ap.parse_args()
    if args.validate:
        validate(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
