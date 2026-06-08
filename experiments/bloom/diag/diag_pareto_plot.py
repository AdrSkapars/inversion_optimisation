"""Plot Pareto curves (mean P vs strong-bias count) for every method we kept.

Pruned to remove dominated/redundant lines (hard-gating, t × rephrase,
paraphrase, jail variations, PoE n=50 baseline). Added today's new winners:
X3 best-of-N target-pick / jail-pick sweeps.

Bias counts are manual strict tallies from the conversation.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "results.json"
OUT_PNG = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "pareto.png"


def mean_p(data, key_path, cell_key):
    """data: scenarios list. key_path: list of sub-keys to reach the cell dict.
    cell_key: which cell to extract best_per_token_p from."""
    ps = []
    for sc in data:
        d = sc
        for k in key_path:
            d = d.get(k, {}) if d else {}
        if d and cell_key in d:
            p = d[cell_key].get("best_per_token_p")
            if p is not None: ps.append(p)
    return sum(ps)/len(ps) if ps else None


def mean_field(data, key_path, field):
    """Mean of a top-level field per scenario, drilling key_path then reading field."""
    ps = []
    for sc in data:
        d = sc
        for k in key_path:
            d = d.get(k, {}) if d else {}
        if d:
            v = d.get(field)
            if v is not None: ps.append(v)
    return sum(ps)/len(ps) if ps else None


def main():
    data = json.load(open(RESULTS, encoding="utf-8"))["scenarios"]

    curves = {}

    # 1) Soft PoE (t × jail, proper sys, T_t=1) — sweep β
    pts = []
    for b, bias in [(1.0, 1.5), (2.0, 4), (3.0, 5), (4.0, 7), (5.0, 7.5),
                    (6.0, 9), (8.0, 10), (10.0, 11)]:
        p = mean_p(data, ["poe_target_x_jail_proper_sys"], f"beta{b}_Tt1.0")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["Soft PoE t × jail — vary β (T_t=1)"] = pts

    # 2) Asymmetric T (β=2) — sweep T_t
    pts = []
    for tt, bias in [(1.0, 4), (2.0, 7.5), (3.0, 9), (5.0, 9.5),
                     (7.0, 11.5), (10.0, 10.5), (15.0, 10.5)]:
        p = mean_p(data, ["poe_target_x_jail_proper_sys"], f"beta2.0_Tt{tt}")
        if p is not None: pts.append((f"T_t={tt:g}", p, bias))
    curves["Soft PoE t × jail — vary T_t (β=2)"] = pts

    # 3) target × jail-CFG (w=0.5) — sweep β
    pts = []
    for b, bias in [(1.0, 4), (2.0, 10), (3.0, 13), (5.0, 14)]:
        p = mean_p(data, ["poe_target_x_jail_cfg_sweep"], f"beta{b}_w0.5")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["target × jail-CFG — vary β (w=0.5)"] = pts

    # 4) Jail-only CFG — sweep w
    pts = []
    for w, bias in [(0.0, 11), (0.5, 15), (1.0, 15), (2.0, 15), (4.0, 15)]:
        p = mean_p(data, ["jail_cfg_sweep"], f"w{w}")
        if p is not None: pts.append((f"w={w:g}", p, bias))
    curves["Jail-only CFG — vary w"] = pts

    # 6) X3_aggrieved best-of-N — TARGET-FILTER selection (NEW)
    pts = []
    # n=1 (single-shot X3_aggrieved)
    p = mean_field(data, ["jail_biased_rewrite_prompt_X3_aggrieved"], "target_p_pct")
    if p is not None: pts.append(("n=1", p, 13))
    # n=10 target-pick
    p = mean_field(data, ["jail_rewrite_x3_best_of_10", "target_pick"], "target_p_pct")
    if p is not None: pts.append(("n=10", p, 12))
    # n=100 target-pick
    p = mean_field(data, ["jail_rewrite_x3_best_of_100", "target_pick"], "target_p_pct")
    if p is not None: pts.append(("n=100", p, 13))
    # n=250 target-pick
    p = mean_field(data, ["jail_rewrite_x3_best_of_250", "target_pick"], "target_p_pct")
    if p is not None: pts.append(("n=250", p, 10))
    curves["X3 best-of-N — TARGET-filter (NEW)"] = pts

    # 7) X3_aggrieved best-of-N — JAIL-FILTER selection (NEW)
    pts = []
    p = mean_field(data, ["jail_rewrite_x3_best_of_10", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=10", p, 11))
    p = mean_field(data, ["jail_rewrite_x3_best_of_100", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=100", p, 13))
    p = mean_field(data, ["jail_rewrite_x3_best_of_250", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=250", p, 12))
    curves["X3 best-of-N — JAIL-filter (NEW)"] = pts

    # 8) Jail rewrite — vary prompt intensity (NEW)
    # The clean intensity sweep from the original prompt-design exploration:
    # very_subtle → subtle → strong → extreme (no X-variations, no anti-copy)
    pts = []
    for variant, label, bias in [
        ("1_very_subtle", "very subtle", 0),
        ("2_subtle",      "subtle",      0),
        ("3_strong",      "strong",      2),
        ("4_extreme",     "extreme",     8),
    ]:
        p = mean_field(data, [f"jail_biased_rewrite_prompt_{variant}"], "target_p_pct")
        if p is not None: pts.append((label, p, bias))
    curves["Jail rewrite — vary intensity (NEW)"] = pts

    # -------- Reference points (single dots) --------
    refs = []
    target_ps = [sc["scores"]["target"]["per_token_p_pct"] for sc in data]
    jail_ps   = [sc["scores"]["jail"]["per_token_p_pct"]   for sc in data]
    refs.append(("target raw (single sample)",  sum(target_ps)/len(target_ps), 0,    "o"))
    refs.append(("jail raw (single sample)",    sum(jail_ps)/len(jail_ps),     13,   "s"))
    jb5 = mean_p(data, [], "jail_best_of_5_proper_sys")
    if jb5 is not None:
        refs.append(("jail best-of-5",  jb5, 12, "D"))

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(14, 8.5))
    cmap = plt.get_cmap("tab10")
    # Use distinct colors for new lines
    palette = {
        "Soft PoE t × jail — vary β (T_t=1)":          ("#1f77b4", "-",  "o", 1.5, 0.55),
        "Soft PoE t × jail — vary T_t (β=2)":           ("#aec7e8", "--", "o", 1.5, 0.55),
        "target × jail-CFG — vary β (w=0.5)":            ("#9467bd", "-",  "s", 1.5, 0.65),
        "Jail-only CFG — vary w":                        ("#d62728", "-",  "^", 1.5, 0.55),
        "X3 best-of-N — TARGET-filter (NEW)":            ("#2ca02c", "-",  "*", 2.5, 0.95),
        "X3 best-of-N — JAIL-filter (NEW)":              ("#98df8a", "-",  "P", 2.5, 0.85),
        "Jail rewrite — vary intensity (NEW)":            ("#ff7f0e", ":",  "D", 2.0, 0.80),
    }
    for name, pts in curves.items():
        if not pts: continue
        color, ls, marker, lw, alpha = palette.get(name, ("k", "-", "o", 1.5, 0.6))
        pts_sorted = sorted(pts, key=lambda x: x[1])
        xs = [p[1] for p in pts_sorted]
        ys = [p[2] for p in pts_sorted]
        ax.plot(xs, ys, linestyle=ls, marker=marker, color=color, label=name,
                linewidth=lw, markersize=9 if "NEW" in name else 7, alpha=alpha)
        for (lab, x, y) in pts_sorted:
            ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.85)

    # Reference dots
    ref_markers = {"target raw (single sample)": ("o", "#444", 130),
                   "jail raw (single sample)":   ("s", "#a33", 130),
                   "jail best-of-5":             ("D", "#3a3", 130)}
    for (name, x, y, _m) in refs:
        m, c, s = ref_markers.get(name, ("*", "k", 130))
        ax.scatter([x], [y], marker=m, color=c, s=s, edgecolors="black",
                   linewidths=1.0, zorder=5, label=name)

    ax.set_xlabel("Mean per-token P (%) — under target with (sys + user_input)", fontsize=11)
    ax.set_ylabel("Strong-bias scenarios (out of 15)", fontsize=11)
    ax.set_title("Pareto frontier: target probability vs strong-bias preservation\n"
                 "(pruned lines + new X3 best-of-N sweep + new single-shot rewrite prompts)", fontsize=12)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%g%%"))
    ax.xaxis.set_major_locator(mtick.LogLocator(base=10, subs=[1, 2, 5]))
    ax.set_ylim(-0.5, 16)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
