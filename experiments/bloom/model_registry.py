"""Central registry of per-model behavior that is NOT auto-derivable from model files.

The ONLY such property today is whether a model's chat template auto-opens a <think>
reasoning block that must be closed with a prefilled empty think block (Qwen3-style).
Everything else — EOS / pad / vocab / tokenizer — is read automatically from each
model's own files (see `_load_hf_corruption_models`) and needs no entry here.

To experiment with a NEW target or corruptor model: add ONE line to `_USES_THINK_BLOCK`.
An unregistered model raises immediately (`uses_think_block`), so a new model can never
silently get the wrong wrapper — the error tells you exactly what to add.
"""
from __future__ import annotations

# normalized name (lowercased, no "local/" prefix) -> does the chat template auto-open a
# <think> block that must be closed with a prefilled empty think block?
_USES_THINK_BLOCK = {
    "qwen/qwen3-4b": True,
    "microsoft/phi-4-mini-instruct": False,
    "duoneural/phi-4-mini-abliterated": False,
}

_THINK_PREFILL = "<think>\n\n</think>\n"


def normalize(name: str) -> str:
    """Lowercase and strip the 'local/' engine prefix used in cfg model ids."""
    n = (name or "").strip()
    if n.startswith("local/"):
        n = n[len("local/"):]
    return n.lower()


def uses_think_block(name: str) -> bool:
    """True if this model's chat template auto-opens a <think> block (Qwen3-style).

    Raises ValueError for any model not in `_USES_THINK_BLOCK` — adding the model there
    (with True/False) is the single, explicit step needed to support a new model.
    """
    key = normalize(name)
    if key not in _USES_THINK_BLOCK:
        raise ValueError(
            f"Model {name!r} is not supported: add it to _USES_THINK_BLOCK in "
            f"experiments/bloom/model_registry.py (value True if its chat template "
            f"auto-opens a <think> block needing a closed-think prefill, e.g. Qwen3; "
            f"else False). Registered: {sorted(_USES_THINK_BLOCK)}")
    return _USES_THINK_BLOCK[key]


def think_prefix(name: str) -> str:
    """Closed-<think> prefill text for this model ('' if it has no auto think block)."""
    return _THINK_PREFILL if uses_think_block(name) else ""
