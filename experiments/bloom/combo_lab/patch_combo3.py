import os, ast
p=os.path.expanduser("~/inversion_optimisation/experiments/bloom/bloom_corrupt_combo.py")
s=open(p,encoding="utf-8").read()
if "_CMODE == 'jail_veto'" in s:
    print("round-3 modes already present; skipping"); raise SystemExit

# params
anchor="        _KLREF = float(_os.environ.get('BLOOM_COMBINE_KLREF', '3.0'))\n"
assert s.count(anchor)==1
s=s.replace(anchor, anchor+
    "        _CONFT = float(_os.environ.get('BLOOM_COMBINE_CONFT', '0.5'))\n"
    "        _VETOFL = float(_os.environ.get('BLOOM_COMBINE_VETOFL', '1e-4'))\n", 1)

fallback=("                else:\n"
          "                    z = eb1 * tl + eb2 * cl - _basec\n"
          "                if z is not None:\n"
          "                    probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
assert s.count(fallback)==1, s.count(fallback)
newbranches=(
"                elif _CMODE == 'jail_veto':\n"
"                    # BOTTOM-N: forbid tokens the jail expert would (almost) never emit\n"
"                    # (refusal-onset tokens), sample PURE target over what remains + mild push.\n"
"                    _pc = torch.softmax(cl, -1)\n"
"                    _keep = _pc >= _VETOFL\n"
"                    _z = eb1 * tl + eb2 * cl - _basec\n"
"                    z = torch.where(_keep, _z, torch.full_like(_z, -1e30))\n"
"                elif _CMODE == 'conf_gate':\n"
"                    # MASK LAME TOKENS: no jail push where target is near-certain (structural/forced)\n"
"                    _pt = torch.softmax(tl, -1)\n"
"                    _mx = _pt.max(-1, keepdim=True).values\n"
"                    _push = (_mx < _CONFT).float()\n"
"                    z = eb1 * tl + (eb2 * _push) * cl - _basec\n"
"                elif _CMODE == 'max_pool':\n"
"                    # WEIRD non-additive: p ~ max(p_t, a*p_c) (inject jail's top w/o PoE suppression)\n"
"                    _pt = torch.softmax(tl, -1); _pc = torch.softmax(cl, -1)\n"
"                    _a = min(eb2 / (1.0 + eb2), 0.95)\n"
"                    _m = torch.maximum(_pt, _a * _pc)\n"
"                    probs = _m / _m.sum(-1, keepdim=True).clamp_min(1e-12)\n"
"                    z = None\n"
)
s=s.replace(fallback, newbranches+fallback, 1)
open(p,"w",encoding="utf-8").write(s); ast.parse(s)
print("PATCH3 OK; added modes: jail_veto conf_gate max_pool")
