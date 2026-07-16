import os, ast
p=os.path.expanduser("~/inversion_optimisation/experiments/bloom/bloom_corrupt_combo.py")
s=open(p,encoding="utf-8").read()

# idempotency guard
if "_CMODE == 'mix'" in s:
    print("round-2 modes already present; skipping"); raise SystemExit

# 1) extra params (delta for trust-region, klref for kl-gating)
anchor="        _CEVERY = int(_os.environ.get('BLOOM_COMBINE_EVERY', '2'))\n"
assert s.count(anchor)==1, s.count(anchor)
s=s.replace(anchor, anchor+
    "        _CD = float(_os.environ.get('BLOOM_COMBINE_DELTA', '3.0'))\n"
    "        _KLREF = float(_os.environ.get('BLOOM_COMBINE_KLREF', '3.0'))\n", 1)

# 2) new elif branches inserted BEFORE the fallback else
fallback=("                else:\n"
          "                    z = eb1 * tl + eb2 * cl - _basec\n"
          "                probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
assert s.count(fallback)==1, ("fallback anchor", s.count(fallback))
newbranches=(
"                elif _CMODE == 'mix':\n"
"                    # ARITHMETIC (prob-space) mixture instead of geometric PoE\n"
"                    _pt = torch.softmax(tl, -1); _pc = torch.softmax(cl, -1)\n"
"                    _a = min(eb2 / (1.0 + eb2), 0.95)\n"
"                    probs = (1.0 - _a) * _pt + _a * _pc\n"
"                    z = None\n"
"                elif _CMODE == 'kl_beta':\n"
"                    # scale jail weight per-position by KL(jail||target): steer only where they disagree\n"
"                    _pt = torch.softmax(tl, -1); _pc = torch.softmax(cl, -1)\n"
"                    _kl = (_pc * (torch.log(_pc.clamp_min(1e-12)) - torch.log(_pt.clamp_min(1e-12)))).sum(-1, keepdim=True)\n"
"                    _sc = (_kl / _KLREF).clamp(0.0, 1.0)\n"
"                    z = eb1 * tl + (eb2 * _sc) * cl - _basec\n"
"                elif _CMODE == 'tr_cfg':\n"
"                    # trust-region contrastive: push along clamped (jail-target) delta\n"
"                    _d = (cl - tl).clamp(-_CD, _CD)\n"
"                    z = eb1 * tl + eb2 * _d\n"
)
s=s.replace(fallback, newbranches+fallback, 1)

# 3) make the final softmax conditional (mix sets probs directly, z=None)
final=("                else:\n"
       "                    z = eb1 * tl + eb2 * cl - _basec\n"
       "                probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
final_new=("                else:\n"
           "                    z = eb1 * tl + eb2 * cl - _basec\n"
           "                if z is not None:\n"
           "                    probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
assert s.count(final)==1
s=s.replace(final, final_new, 1)

open(p,"w",encoding="utf-8").write(s); ast.parse(s)
print("PATCH2 OK; added modes: mix kl_beta tr_cfg")
