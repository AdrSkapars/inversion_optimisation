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
OUT_PNG        = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "pareto.png"
OUT_PNG_LINEAR = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "pareto_linear.png"


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


    # 13) Two-temperature PoE — NEW Pareto point at STRONG=11
    # Storage: poe_target_x_corruption_temperature_n10[label]
    pts = []
    for label, key, bias in [
        ("b3_Tt15_Tc05 ★", "b3_Tt15_Tc05", 11),
    ]:
        p = mean_field(data, ["poe_target_x_corruption_temperature_n10", key], "best_target_p_pct")
        if p is not None: pts.append((label, p, bias))
    curves["Two-temperature PoE (n=10)"] = pts

    # 14) PROMPT-DIVERSITY n=10 — MAJOR NEW Pareto winner at STRONG=12
    # Storage: poe_target_x_corruption_prompt_div_n10[label]
    pts = []
    for label, key, bias in [
        ("β=1",            "n10p_b1_v3",  1),
        ("β=1.5",          "n10p_b15_v3", 2),
        ("β=2",            "n10p_b2_v3",  6),
        ("β=2.5",          "n10p_b25_v3", 7),
        ("β=3 ★★★",        "n10p_b3_v3", 12),
        ("β=3.5",          "n10p_b35_v3", 9),
        ("β=4",            "n10p_b4_v3", 11),
        ("β=4.5",          "n10p_b45_v3", 11),
        ("β=5",            "n10p_b5_v3", 10),
        ("β=5.5",          "n10p_b55_v3", 10),
        ("β=6",            "n10p_b6_v3", 12),
    ]:
        p = mean_field(data, ["poe_target_x_corruption_prompt_div_n10", key], "best_target_p_pct")
        if p is not None: pts.append((label, p, bias))
    curves["Prompt-diversity PoE v3 (n=10)"] = pts

    # 15) PROMPT-DIVERSITY v4 — completely different prompts, robustness test
    # Storage: poe_target_x_corruption_prompt_div_v4_n10[label]
    pts = []
    for label, key, bias in [
        ("v4 β=1",        "n10p_v4_b1",  1),
        ("v4 β=1.5",      "n10p_v4_b15", 2),
        ("v4 β=2",        "n10p_v4_b2",  5),
        ("v4 β=2.5",      "n10p_v4_b25", 6),
        ("v4 β=3 ★★",     "n10p_v4_b3",  8),
        ("v4 β=3.5 ★",    "n10p_v4_b35", 9),
        ("v4 β=4",        "n10p_v4_b4",  9),
        ("v4 β=4.5",      "n10p_v4_b45", 10),
        ("v4 β=5",        "n10p_v4_b5",  9),
        ("v4 β=5.5",      "n10p_v4_b55", 11),
        ("v4 β=6",        "n10p_v4_b6",  12),
    ]:
        p = mean_field(data, ["poe_target_x_corruption_prompt_div_v4_n10", key], "best_target_p_pct")
        if p is not None: pts.append((label, p, bias))
    curves["Prompt-diversity PoE v4 (n=10)"] = pts



    # Reference dots: target raw n=1, target raw n=10 (target-filter), jail raw n=1
    target_n1 = sum(sc["scores"]["target"]["per_token_p_pct"] for sc in data) / len(data)
    jail_n1   = sum(sc["scores"]["jail"]["per_token_p_pct"]   for sc in data) / len(data)
    target_n10 = mean_field(data, ["target_only_sampling_n10", "target_pick"], "target_p_pct")

    refs = [
        ("target raw n=1",                target_n1,  0,  "o"),
        ("target raw n=10 target-filter", target_n10, 0,  "D") if target_n10 is not None else None,
        ("jail raw n=1",                  jail_n1,    13, "s"),
    ]
    refs = [r for r in refs if r is not None]

    # -------- Plot --------
    palette = {
        "Jail-only CFG — vary w":                        ("#d62728", "-",  "^", 1.5, 0.55),
        "X3 best-of-N — TARGET-filter":            ("#2ca02c", "-",  "*", 2.5, 0.95),
        "X3 best-of-N — JAIL-filter":              ("#98df8a", "-",  "P", 2.5, 0.85),
        "Jail rewrite — vary intensity":            ("#ff7f0e", ":",  "D", 2.0, 0.80),
        "target × corruption PoE — vary β (n=10 target-filter)":   ("#8c564b", "-",  "h", 2.5, 0.90),
        "target × corruption-CFG PoE — vary w (β=5, n=10 target-filter)":   ("#2c3e50", "-",  "*", 2.5, 0.95),
        "target × corruption-CFG PoE — vary β (w=0.5, n=10 target-filter)":   ("#9b59b6", "-",  "P", 2.5, 0.95),
        "Two-temperature PoE (n=10)":               ("#000000", "",   "*", 0,   1.0),
        "Prompt-diversity PoE v3 (n=10)":           ("#e74c3c", "-",  "*", 3.0, 1.0),
        "Prompt-diversity PoE v4 (n=10)":           ("#3498db", "-",  "D", 3.0, 1.0),
    }
    ref_markers = {"target raw n=1":                ("o", "#444",    140),
                   "target raw n=10 target-filter": ("D", "#888",    140),
                   "jail raw n=1":                  ("s", "#a52a2a", 140)}

    def render(out_path, log_x: bool):
        fig, ax = plt.subplots(figsize=(14, 8.5))
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

        for (name, x, y, _m) in refs:
            m, c, s = ref_markers.get(name, ("*", "k", 130))
            ax.scatter([x], [y], marker=m, color=c, s=s, edgecolors="black",
                       linewidths=1.0, zorder=5, label=name)

        ax.set_xlabel("Mean per-token P (%) — under target with (sys + user_input)", fontsize=11)
        ax.set_ylabel("Strong-bias scenarios (out of 15)", fontsize=11)
        title = ("Pareto frontier: target probability vs strong-bias preservation\n"
                 + ("(log x-scale)" if log_x else "(linear x-scale)"))
        ax.set_title(title, fontsize=12)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mtick.LogLocator(base=10, subs=[1, 2, 5]))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%g%%"))
        ax.set_ylim(-0.5, 16)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8.5, frameon=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_path}")

    render(OUT_PNG,        log_x=True)
    render(OUT_PNG_LINEAR, log_x=False)


if __name__ == "__main__":
    main()
