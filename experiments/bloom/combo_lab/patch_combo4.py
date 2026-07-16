import os, ast
p=os.path.expanduser("~/inversion_optimisation/experiments/bloom/combo_lab/bloom_corrupt_combo.py")
s=open(p,encoding="utf-8").read()
if "return_jail_lps" in s:
    print("jail-lp capture already present; skipping"); raise SystemExit

# A1 signature (21-space indent is unique to _hf_poe_generate)
a="                     return_token_lps: bool = False):"
assert s.count(a)==1, ("sig", s.count(a))
s=s.replace(a, "                     return_token_lps: bool = False, return_jail_lps: bool = False):", 1)

# A2 clps init
a="        tlps: List[List[float]] = [[] for _ in range(B)]   # per-token true-target logprob (return_token_lps)\n"
assert s.count(a)==1, ("clps init", s.count(a))
s=s.replace(a, a+"        clps: List[List[float]] = [[] for _ in range(B)]   # per-token jail-expert logprob (return_jail_lps, blackbox)\n", 1)

# A3 capture jail lp right after the tlps append block
a=("                if return_token_lps:\n"
   "                    for i in range(B):\n"
   "                        if bool(live[i]):\n"
   "                            tlps[i].append(float(tlp[i]))\n")
assert s.count(a)==1, ("capture", s.count(a))
add=("                if return_jail_lps:\n"
     "                    clp = cl.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(cl, dim=-1)\n"
     "                    for i in range(B):\n"
     "                        if bool(live[i]):\n"
     "                            clps[i].append(float(clp[i]))\n")
s=s.replace(a, a+add, 1)

# A4 return 3-tuple when requested (disambiguate poe via the following return_target_lp block)
a="        if return_token_lps:\n            return gen, tlps\n        if return_target_lp:\n"
assert s.count(a)==1, ("ret", s.count(a))
s=s.replace(a, "        if return_token_lps:\n            return (gen, tlps, clps) if return_jail_lps else (gen, tlps)\n        if return_target_lp:\n", 1)

# B1 default _gen_once call requests jail lps via env flag
a=("        return _hf_poe_generate(mt, mc, t_prefs, _jp, beta, temperature, max_tokens,\n"
   "                                pad_id, eos_id, device, target_floor=jail_floor, return_token_lps=True)\n")
assert s.count(a)==1, ("default call", s.count(a))
s=s.replace(a, "        return _hf_poe_generate(mt, mc, t_prefs, _jp, beta, temperature, max_tokens,\n"
   "                                pad_id, eos_id, device, target_floor=jail_floor, return_token_lps=True,\n"
   "                                return_jail_lps=bool(os.environ.get('BLOOM_CAPTURE_JAIL_LP')))\n", 1)

# B2 unpack (2- or 3-tuple) + capture mean jail lp per candidate
a=("        gen, tlps = _gen_once(_jp)\n"
   "        for bi, (g, lps) in enumerate(zip(gen, tlps)):\n"
   "            ids = [x for x in g if x != eos_id]\n"
   "            txt = tok.decode(ids, skip_special_tokens=True).strip()\n"
   "            mlp = (sum(lps) / len(lps)) if lps else None   # mean on-policy target logprob (plausibility)\n"
   "            pools[bi].append({\"ids\": ids, \"txt\": txt, \"lps\": lps, \"mlp\": mlp, \"d3\": _d3(txt)})\n")
assert s.count(a)==1, ("unpack", s.count(a))
b=("        _r = _gen_once(_jp)\n"
   "        if len(_r) == 3: gen, tlps, clps = _r\n"
   "        else: gen, tlps = _r; clps = [[] for _ in gen]\n"
   "        for bi, (g, lps) in enumerate(zip(gen, tlps)):\n"
   "            ids = [x for x in g if x != eos_id]\n"
   "            txt = tok.decode(ids, skip_special_tokens=True).strip()\n"
   "            mlp = (sum(lps) / len(lps)) if lps else None   # mean on-policy target logprob (plausibility)\n"
   "            _jl = clps[bi] if bi < len(clps) else []\n"
   "            mjlp = (sum(_jl) / len(_jl)) if _jl else None   # mean jail-expert logprob (blackbox proxy)\n"
   "            pools[bi].append({\"ids\": ids, \"txt\": txt, \"lps\": lps, \"mlp\": mlp, \"mjlp\": mjlp, \"d3\": _d3(txt)})\n")
s=s.replace(a, b, 1)

# B3 out dict exposes jail/target lp
a='        out.append({"best_text": best["txt"], "best_ids": best["ids"], "best_token_probs": tprobs})\n'
assert s.count(a)==1, ("out", s.count(a))
s=s.replace(a, '        out.append({"best_text": best["txt"], "best_ids": best["ids"], "best_token_probs": tprobs,\n'
   '                    "best_target_lp": best.get("mlp"), "best_jail_lp": best.get("mjlp")})\n', 1)

open(p,"w",encoding="utf-8").write(s); ast.parse(s)
print("PATCH4 OK; jail-lp capture wired (env BLOOM_CAPTURE_JAIL_LP)")
