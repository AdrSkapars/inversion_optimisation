"""Generate HTML visualizations of per-token probabilities for PoE outputs.

For each scenario's β=6, T_t=1 output, produces an HTML page showing the
same text twice, once with target log-p highlighting and once with jail
log-p highlighting. Background color: red (very low) → yellow → green (high).

Output: experiments/bloom/runs_15/diag_three_outputs/token_viz/v{N}.html
        plus a token_viz/index.html linking them all.
"""
from __future__ import annotations
import json, math, html
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent.resolve()
RESULTS      = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "results.json"
OUT_DIR      = SCRIPT_DIR.parent / "runs_15" / "diag_three_outputs" / "token_viz"
ANALYSIS_KEY = "token_analysis_beta6_Tt1"


def lp_to_color(lp: float) -> str:
    """log_p → CSS background color. We emphasize the low-probability tail by
    mapping via a piecewise scale on log_p directly:
      lp ≥ 0       → bright green
      lp = -1      → green
      lp = -3      → yellow-green
      lp = -7      → yellow
      lp = -12     → orange
      lp ≤ -20     → deep red
    """
    # Anchor stops in log-prob space → (R, G, B)
    stops = [
        (  0.0,  ( 80, 220,  80)),   # green
        ( -2.0,  (150, 220,  80)),   # yellow-green
        ( -5.0,  (240, 240,  80)),   # yellow
        ( -10.0, (250, 170,  60)),   # orange
        ( -20.0, (230,  60,  60)),   # red
        ( -40.0, (140,  20,  20)),   # dark red
    ]
    if lp >= stops[0][0]:
        return f"rgb{stops[0][1]}"
    for i in range(len(stops) - 1):
        hi_lp, hi_rgb = stops[i]
        lo_lp, lo_rgb = stops[i+1]
        if hi_lp >= lp >= lo_lp:
            t = (lp - hi_lp) / (lo_lp - hi_lp) if hi_lp != lo_lp else 0
            r = int(hi_rgb[0] + (lo_rgb[0] - hi_rgb[0]) * t)
            g = int(hi_rgb[1] + (lo_rgb[1] - hi_rgb[1]) * t)
            b = int(hi_rgb[2] + (lo_rgb[2] - hi_rgb[2]) * t)
            return f"rgb({r},{g},{b})"
    return f"rgb{stops[-1][1]}"


def render_tokens(tokens, lps, title):
    spans = []
    for tok, lp in zip(tokens, lps):
        color = lp_to_color(lp)
        p_pct = math.exp(lp) * 100
        # show actual whitespace in tokens by replacing spaces with U+00A0 (NBSP)
        # so that the colored backgrounds look connected at word boundaries
        display = html.escape(tok).replace(" ", "&nbsp;").replace("\n", "↵<br>")
        tooltip = f"lp={lp:.3f}  P={p_pct:.2f}%"
        spans.append(
            f'<span class="tok" style="background:{color}" title="{tooltip}">{display}</span>'
        )
    return f'<h3>{html.escape(title)}</h3><div class="line">{"".join(spans)}</div>'


def main():
    data = json.load(open(RESULTS, encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index_links = []
    for sc in data["scenarios"]:
        v = sc["variation_number"]
        a = sc.get(ANALYSIS_KEY)
        if not a: continue
        tokens = a["tokens"]
        t_lps  = a["target_lps"]
        j_lps  = a["jail_lps"]
        if not tokens: continue

        sys_p = html.escape(sc.get("sys_prompt", "") or "")
        user_i = html.escape(sc.get("input", "") or "")
        t_avg = a["target_per_token_p_pct"]
        j_avg = a["jail_per_token_p_pct"]

        body = f"""
<!doctype html><html><head><meta charset="utf-8"><title>v={v} token viz</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 20px; line-height: 1.6; }}
h1 {{ font-size: 18px; }}
h3 {{ font-size: 14px; margin-top: 24px; margin-bottom: 8px; color: #444; }}
.meta {{ background: #f4f4f4; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }}
.meta b {{ color: #222; }}
.line {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 14px; line-height: 1.9; }}
.tok {{ padding: 2px 0; border-radius: 2px; color: #111; }}
.legend {{ display: flex; gap: 0; margin: 8px 0 24px; font-family: monospace; font-size: 11px; }}
.legend .swatch {{ padding: 3px 8px; }}
nav {{ margin-bottom: 20px; }}
nav a {{ margin-right: 8px; text-decoration: none; background: #eee; padding: 3px 8px; border-radius: 3px; color: #333; font-size: 12px; }}
</style></head>
<body>
<nav><a href="index.html">← all scenarios</a></nav>
<h1>Scenario v={v} — token-level probabilities (target × jail PoE, β=6, T_t=1)</h1>
<div class="meta">
<b>System:</b> {sys_p}<br>
<b>User:</b> {user_i}<br><br>
<b>avg target per-token P:</b> {t_avg:.3f}% &nbsp; · &nbsp; <b>avg jail per-token P:</b> {j_avg:.3f}%
&nbsp; · &nbsp; <b>n tokens:</b> {len(tokens)}
</div>
<div class="legend">
<span class="swatch" style="background:{lp_to_color(0)}">≥0 (100%)</span>
<span class="swatch" style="background:{lp_to_color(-2)}">−2 (~14%)</span>
<span class="swatch" style="background:{lp_to_color(-5)}">−5 (~0.7%)</span>
<span class="swatch" style="background:{lp_to_color(-10)}">−10 (~0.005%)</span>
<span class="swatch" style="background:{lp_to_color(-20)}">−20 (~2e-7%)</span>
<span class="swatch" style="background:{lp_to_color(-40)}">≤−40</span>
</div>
{render_tokens(tokens, t_lps, "Under TARGET (Qwen3-4B on sys + user_input)")}
{render_tokens(tokens, j_lps, 'Under JAIL ("You are racist..." + user_input + "As a racist:\\n" prefill)')}
</body></html>
"""
        out_file = OUT_DIR / f"v{v}.html"
        out_file.write_text(body, encoding="utf-8")
        index_links.append((v, f"v{v}.html", t_avg, j_avg, len(tokens)))

    # Build index
    rows = "".join(
        f'<tr><td><a href="{href}">v={v}</a></td>'
        f'<td>{t_avg:.2f}%</td><td>{j_avg:.2f}%</td><td>{n}</td></tr>'
        for v, href, t_avg, j_avg, n in sorted(index_links, key=lambda x: x[0])
    )
    index_html = f"""
<!doctype html><html><head><meta charset="utf-8"><title>token viz index</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 700px; margin: 30px auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: left; }}
th {{ background: #f0f0f0; }}
</style></head>
<body>
<h1>Token-level visualizations (β=6, T_t=1 outputs)</h1>
<p>Click each scenario to see per-token target vs jail log-probabilities highlighted.</p>
<table>
<thead><tr><th>Scenario</th><th>avg target P</th><th>avg jail P</th><th>n tokens</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>
"""
    (OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"Wrote {len(index_links)} scenario pages + index to {OUT_DIR}")


if __name__ == "__main__":
    main()
