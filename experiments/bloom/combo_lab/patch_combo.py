import os
p=os.path.expanduser("~/inversion_optimisation/experiments/bloom/bloom_corrupt_combo.py")
s=open(p,encoding="utf-8").read()

# 1) env-driven combine-mode params, inserted right after `B = len(t_prefs)` in _hf_poe_generate
anchor1="        B = len(t_prefs)\n"
assert s.count(anchor1)>=1
ins=("        B = len(t_prefs)\n"
     "        import os as _os, math as _math\n"
     "        _CMODE = _os.environ.get('BLOOM_COMBINE_MODE', 'add')\n"
     "        _CK = int(_os.environ.get('BLOOM_COMBINE_K', '20'))\n"
     "        _CP = float(_os.environ.get('BLOOM_COMBINE_P', '0.9'))\n"
     "        _CEVERY = int(_os.environ.get('BLOOM_COMBINE_EVERY', '2'))\n")
s=s.replace(anchor1, ins, 1)

# 2) step index for every-k (both loops; only the z-combine loop uses _step, others harmless)
assert s.count("        for _ in range(max_new):\n")>=1
s=s.replace("        for _ in range(max_new):\n", "        for _step in range(max_new):\n")

# 3) replace the additive z-combination with a mode dispatch
old=("            else:\n"
     "                z = eb1 * tl + eb2 * cl - (eb3 * nl if cfg_on else 0.0)\n"
     "                probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
assert s.count(old)==1, ("z-combine anchor count", s.count(old))
new=(
"            else:\n"
"                _basec = (eb3 * nl if cfg_on else 0.0)\n"
"                if _CMODE == 'add':\n"
"                    z = eb1 * tl + eb2 * cl - _basec\n"
"                elif _CMODE == 'jail_topk':\n"
"                    # jail only boosts its own top-K tokens; elsewhere pure target\n"
"                    _kth = torch.topk(cl, min(_CK, cl.shape[-1]), dim=-1).values[:, -1:]\n"
"                    _m = (cl >= _kth).float()\n"
"                    z = eb1 * tl + eb2 * (cl * _m) - _basec\n"
"                elif _CMODE == 'target_nucleus':\n"
"                    # restrict to target's top-p nucleus, jail-weighted within it\n"
"                    z = eb1 * tl + eb2 * cl - _basec\n"
"                    _tp = torch.softmax(tl, -1)\n"
"                    _sp, _si = torch.sort(_tp, descending=True, dim=-1)\n"
"                    _cs = torch.cumsum(_sp, dim=-1)\n"
"                    _ks = _cs <= _CP\n"
"                    _ks[..., 0] = True\n"
"                    _keep = torch.zeros_like(_ks).scatter(-1, _si, _ks)\n"
"                    z = torch.where(_keep, z, torch.full_like(z, -1e30))\n"
"                elif _CMODE == 'entropy_beta':\n"
"                    # scale jail weight by target entropy (steer more where target is uncertain)\n"
"                    _tp = torch.softmax(tl, -1)\n"
"                    _H = -(_tp * torch.log(_tp.clamp_min(1e-12))).sum(-1, keepdim=True)\n"
"                    _sc = (_H / _math.log(tl.shape[-1])).clamp(0.0, 1.0)\n"
"                    z = eb1 * tl + (eb2 * _sc) * cl - _basec\n"
"                elif _CMODE == 'every_k':\n"
"                    if (_step % _CEVERY) == 0:\n"
"                        z = eb1 * tl + eb2 * cl - _basec\n"
"                    else:\n"
"                        z = eb1 * tl\n"
"                else:\n"
"                    z = eb1 * tl + eb2 * cl - _basec\n"
"                probs = torch.softmax(z / max(temperature, 1e-6), -1)\n")
s=s.replace(old,new,1)

open(p,"w",encoding="utf-8").write(s)
import ast; ast.parse(s)
print("PATCHED combo copy OK; modes: add jail_topk target_nucleus entropy_beta every_k")
