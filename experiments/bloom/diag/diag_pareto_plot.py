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

    # 1) Jail-only CFG — sweep w
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
    curves["X3 best-of-N — TARGET-filter"] = pts

    # 7) X3_aggrieved best-of-N — JAIL-FILTER selection (NEW)
    pts = []
    p = mean_field(data, ["jail_rewrite_x3_best_of_10", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=10", p, 11))
    p = mean_field(data, ["jail_rewrite_x3_best_of_100", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=100", p, 13))
    p = mean_field(data, ["jail_rewrite_x3_best_of_250", "jail_pick"], "target_p_pct")
    if p is not None: pts.append(("n=250", p, 12))
    curves["X3 best-of-N — JAIL-filter"] = pts

    # 8) Jail rewrite — vary prompt intensity (NEW)
    # Intensity sweep extended into the higher-bias regime with stronger-tone variants.
    pts = []
    for variant, label, bias in [
        ("2_subtle",        "subtle",        0),
        ("3_strong",        "strong",        2),
        ("4_extreme",       "extreme",       8),
        ("X1_vicious",      "vicious",       12),
        ("X5_authoritative","authoritative", 12),
        ("X3_aggrieved",    "aggrieved",     13),
    ]:
        p = mean_field(data, [f"jail_biased_rewrite_prompt_{variant}"], "target_p_pct")
        if p is not None: pts.append((label, p, bias))
    curves["Jail rewrite — vary intensity"] = pts

    # 9) Target × corruption PoE — vary β, n=10 best-of-N target-filter (NEW)
    pts = []
    for b, bias in [(0.5, 0), (1.0, 0), (2.0, 4), (3.0, 6),
                    (4.0, 8), (5.0, 10), (6.0, 8), (7.0, 9), (8.0, 9)]:
        p = mean_field(data, ["poe_target_x_corruption_sweep", f"b{b}", "n10_target_pick"], "best_target_p_pct")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["target × corruption PoE — vary β (n=10 target-filter)"] = pts

    # 10) target × corruption-CFG PoE — w=0.5 fixed, vary β, n=10 target-filter
    pts = []
    for b, bias in [(1.0, 4), (2.0, 13), (3.0, 13),
                    (4.0, 14), (5.0, 14), (6.0, 14)]:
        p = mean_field(data, ["poe_target_x_corruption_cfg_beta_sweep", f"b{b}"], "best_target_p_pct")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["target × corruption-CFG PoE — vary β (w=0.5, n=10 target-filter)"] = pts

    # 11) target × corruption + TARGET-prob masking, β=5 fixed, sweep threshold
    pts = []
    for th, bias in [(-5.0, 0), (-10.0, 0), (-15.0, 4), (-16.0, 3),
                     (-17.0, 4), (-18.0, 4), (-19.0, 4), (-20.0, 9), (-40.0, 10)]:
        p = mean_field(data, ["poe_target_x_corruption_masked", f"th{th}"], "best_target_p_pct")
        if p is not None: pts.append((f"th={th:g}", p, bias))
    curves["target × corruption PoE + target-mask — vary th (β=5, n=10)"] = pts

    # 12) target × corruption + JAIL-prob masking, β=5 fixed, sweep threshold
    pts = []
    for th, bias in [(-5.0, 4), (-10.0, 5), (-15.0, 9), (-20.0, 8),
                     (-25.0, 8), (-30.0, 7), (-40.0, 10)]:
        p = mean_field(data, ["poe_target_x_corruption_jail_masked", f"th{th}"], "best_target_p_pct")
        if p is not None: pts.append((f"th={th:g}", p, bias))
    curves["target × corruption PoE + jail-mask — vary th (β=5, n=10)"] = pts

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
        "Jail-only CFG — vary w":                        ("#d62728", "-",  "^", 1.5, 0.55),
        "X3 best-of-N — TARGET-filter":            ("#2ca02c", "-",  "*", 2.5, 0.95),
        "X3 best-of-N — JAIL-filter":              ("#98df8a", "-",  "P", 2.5, 0.85),
        "Jail rewrite — vary intensity":            ("#ff7f0e", ":",  "D", 2.0, 0.80),
        "target × corruption PoE — vary β (n=10 target-filter)":   ("#8c564b", "-",  "h", 2.5, 0.90),
        "target × corruption-CFG PoE — vary w (β=5, n=10 target-filter)":   ("#2c3e50", "-",  "*", 2.5, 0.95),
        "target × corruption-CFG PoE — vary β (w=0.5, n=10 target-filter)":   ("#9b59b6", "-",  "P", 2.5, 0.95),
        "target × corruption PoE + target-mask — vary th (β=5, n=10)":   ("#1abc9c", "--",  "v", 1.5, 0.65),
        "target × corruption PoE + jail-mask — vary th (β=5, n=10)":   ("#e67e22", "--",  "^", 1.5, 0.65),
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
