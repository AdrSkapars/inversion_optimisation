"""Plot Pareto curves (mean P vs strong-bias count) for every method we tried.

Each line is one method sweeping its main knob. Bias counts are manual tallies
from the conversation (approximate; ±1 honesty band).
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


def main():
    data = json.load(open(RESULTS, encoding="utf-8"))["scenarios"]

    # -------- Each method: (label, list of (param_str, P, bias_count)) --------
    curves = {}

    # 1) Soft PoE (t × jail, proper sys, T_t=1) sweeping β
    pts = []
    for b, bias in [(1.0, 1.5), (2.0, 4), (3.0, 5), (4.0, 7), (5.0, 7.5),
                    (6.0, 9), (8.0, 10), (10.0, 11)]:
        p = mean_p(data, ["poe_target_x_jail_proper_sys"], f"beta{b}_Tt1.0")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["Soft PoE t × jail — vary β (T_t=1)"] = pts

    # 2) Asymmetric T (β=2 fixed) sweeping T_t
    pts = []
    for tt, bias in [(1.0, 4), (2.0, 7.5), (3.0, 9), (5.0, 9.5),
                     (7.0, 11.5), (10.0, 10.5), (15.0, 10.5)]:
        p = mean_p(data, ["poe_target_x_jail_proper_sys"], f"beta2.0_Tt{tt}")
        if p is not None: pts.append((f"T_t={tt:g}", p, bias))
    curves["Soft PoE t × jail — vary T_t (β=2)"] = pts

    # 3) Target-gate hard constraint — vary threshold
    pts = []
    for th, bias in [(-5.0, 1), (-8.0, 3.5), (-10.0, 3), (-15.0, 7), (-20.0, 10)]:
        p = mean_p(data, ["poe_target_x_jail_hard_constraint"], f"th{th}")
        if p is not None: pts.append((f"th={th:g}", p, bias))
    curves["Hard-gate (target gates, jail samples) — vary threshold"] = pts

    # 4) Jail-gate hard constraint — vary threshold
    pts = []
    for th, bias in [(-2.0, 7), (-3.0, 8), (-4.0, 6), (-5.0, 4),
                     (-8.0, 2), (-10.0, 0.5), (-15.0, 1), (-20.0, 0)]:
        p = mean_p(data, ["poe_target_x_jail_hard_constraint_jailgate"], f"th{th}")
        if p is not None: pts.append((f"th={th:g}", p, bias))
    curves["Hard-gate (jail gates, target samples) — vary threshold"] = pts

    # 5) Jail-CFG alone — vary w
    pts = []
    for w, bias in [(0.0, 11), (0.5, 15), (1.0, 15), (2.0, 15), (4.0, 15)]:
        p = mean_p(data, ["jail_cfg_sweep"], f"w{w}")
        if p is not None: pts.append((f"w={w:g}", p, bias))
    curves["Jail-only CFG — vary w (no target)"] = pts

    # 6) target × jail-CFG (w=0.5) — vary β
    pts = []
    for b, bias in [(1.0, 4), (2.0, 10), (3.0, 13), (5.0, 14)]:
        p = mean_p(data, ["poe_target_x_jail_cfg_sweep"], f"beta{b}_w0.5")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["Soft PoE t × jail-CFG — vary β (w=0.5)"] = pts

    # 6b) target × jail-CFG + outlier mask
    pts = []
    for b, th, bias in [(2.0, -15.0, 3), (3.0, -15.0, 4), (3.0, -20.0, 8),
                        (5.0, -15.0, 6), (5.0, -20.0, 14)]:
        p = mean_p(data, ["poe_target_x_jail_cfg_masked"], f"beta{b}_w0.5_th{th}")
        if p is not None: pts.append((f"β={b:g},th={th:g}", p, bias))
    curves["Soft PoE t × jail-CFG + mask (w=0.5)"] = pts

    # 7) Soft PoE (t × rephrase, full HF) — vary β
    pts = []
    for b, bias in [(1.0, 5), (2.0, 7.5), (3.0, 8), (4.0, 8)]:
        p = mean_p(data, ["poe_full_logprobs_sweep"], f"beta{b}")
        if p is not None: pts.append((f"β={b:g}", p, bias))
    curves["Soft PoE t × rephrase (single target) — vary β"] = pts

    # 8) Jail variations (new input+output pairs)
    pts = []
    p = mean_p(data, [], "jail_variation_best_of_5")
    if p is not None:
        pts.append(("variations", p, 11))
    curves["Jail variations (new input+output pairs)"] = pts

    # 9) Jail paraphrase methods
    pts = []
    p = mean_p(data, [], "jail_paraphrase_best_of_50")
    if p is not None:
        pts.append(("BoN=50", p, 11))
    p = mean_p(data, [], "jail_paraphrase_iterative_50")
    if p is not None:
        pts.append(("iter 25+25", p, 11))
    curves["Jail paraphrase search (single vs iterative, n=50)"] = pts

    # 10) PoE n=50 baseline
    pts = []
    p = mean_p(data, [], "poe_n50_baseline")
    if p is not None:
        pts.append(("β=5,n=50", p, 5))
    curves["PoE n=50 baseline (β=5, same selection)"] = pts

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
    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = plt.get_cmap("tab10")
    for i, (name, pts) in enumerate(curves.items()):
        if not pts: continue
        # sort by P for clean line
        pts_sorted = sorted(pts, key=lambda x: x[1])
        xs = [p[1] for p in pts_sorted]
        ys = [p[2] for p in pts_sorted]
        ax.plot(xs, ys, "-o", color=cmap(i), label=name, linewidth=2,
                markersize=8, alpha=0.85)
        # annotate each point with its param value
        for (lab, x, y) in pts_sorted:
            ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=cmap(i), alpha=0.75)

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
    ax.set_title("Pareto frontier: probability vs bias preservation\n"
                 "(every method we tried, sweeping its main knob; n=5 best-of-5)", fontsize=12)
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
