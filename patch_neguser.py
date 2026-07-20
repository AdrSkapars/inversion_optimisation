#!/usr/bin/env python3
"""BLOOM_JAIL_NEG_USER: give the negative expert a real harmful USER TURN instead of only a
negative system persona. The subtracted direction then comes from a refusal the model actually
PRODUCES in response to a harmful request, rather than one it was instructed to perform
('rc' preset). No-op unless the env var is set. Idempotent."""
import shutil, sys

P = "experiments/bloom/bloom_corrupt.py"
src = open(P, encoding="utf-8").read()
if "neg_user_prompt" in src:
    print("ALREADY PATCHED — no change"); sys.exit(0)
shutil.copy(P, P + ".pre_neguser.bak")

def sub(old, new, label):
    global src
    assert src.count(old) == 1, f"anchor {label}: found {src.count(old)}"
    src = src.replace(old, new, 1)

# 1) read the new field
sub('    jail_neg_sys = jail_runtime_cfg.get("neg_system_prompt", "") or ""',
    '    jail_neg_sys = jail_runtime_cfg.get("neg_system_prompt", "") or ""\n'
    '    jail_neg_user = jail_runtime_cfg.get("neg_user_prompt", "") or ""   # replaces the conversation on the neg branch',
    "read")

# 2) neg branch may now be enabled by a user prompt alone
sub("    neg_on = (jail_b3 != 0.0) and bool(jail_neg_sys) and not combine_on",
    "    neg_on = (jail_b3 != 0.0) and (bool(jail_neg_sys) or bool(jail_neg_user)) and not combine_on",
    "neg_on")

# 3) construction: replace the conversation when a neg user turn is given
sub('            n_msgs = [{"role": "system", "content": jail_neg_sys}] + conv',
    '            if jail_neg_user:\n'
    '                # ELICITED refusal: the conversation is REPLACED by a harmful request, so the\n'
    '                # subtracted direction is the refusal the model actually produces for it,\n'
    '                # continued along the tokens generated so far.\n'
    '                n_msgs = ([{"role": "system", "content": jail_neg_sys}] if jail_neg_sys else []) \\\n'
    '                         + [{"role": "user", "content": jail_neg_user}]\n'
    '            else:\n'
    '                n_msgs = [{"role": "system", "content": jail_neg_sys}] + conv',
    "n_msgs")

# 4) runtime cfg plumbing
sub('            "neg_system_prompt": jail_cfg.get("neg_system_prompt", "") or "", # input-conditioned negative persona (rc/neutral)',
    '            "neg_system_prompt": jail_cfg.get("neg_system_prompt", "") or "", # input-conditioned negative persona (rc/neutral)\n'
    '            "neg_user_prompt":   jail_cfg.get("neg_user_prompt", "") or "",   # harmful user turn -> ELICITED refusal direction',
    "cfg")

# 5) env hook
sub('        if os.environ.get("BLOOM_JAIL_NEG_PROMPT"):\n'
    '            cfg["jailbroken_output"]["neg_system_prompt"] = os.environ["BLOOM_JAIL_NEG_PROMPT"]',
    '        if os.environ.get("BLOOM_JAIL_NEG_PROMPT"):\n'
    '            cfg["jailbroken_output"]["neg_system_prompt"] = os.environ["BLOOM_JAIL_NEG_PROMPT"]\n'
    '        if os.environ.get("BLOOM_JAIL_NEG_USER"):\n'
    '            cfg["jailbroken_output"]["neg_user_prompt"] = os.environ["BLOOM_JAIL_NEG_USER"]',
    "env")

open(P, "w", encoding="utf-8").write(src)
print("PATCHED ok (backup: bloom_corrupt.py.pre_neguser.bak)")
