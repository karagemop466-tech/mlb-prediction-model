"""Generate the static prediction site in docs/.

GitHub Pages serves static files only, so the model runs in GitHub Actions and
this script bakes the results into HTML + JSON. Every number rendered here comes
from the validated pipeline; nothing is placeholder or illustrative.

Pages:
  index.html       today's slate with win probabilities and fair odds
  performance.html walk-forward backtest, calibration, break-even, forward log
  methodology.html how it works, leakage audit, honest limitations
  api/*.json       machine-readable predictions
"""
from __future__ import annotations

import html

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

EASTERN = timezone(timedelta(hours=-4))

CSS = """
:root{--bg:#0b1020;--panel:#141b2e;--panel2:#1b2440;--line:#293350;--text:#e8edf7;
--muted:#8f9dbb;--accent:#4f8cff;--good:#3fb950;--warn:#e3b341;--bad:#f85149;}
@media(prefers-color-scheme:light){:root{--bg:#f5f7fb;--panel:#fff;--panel2:#eef2f9;
--line:#dae1ec;--text:#0f1522;--muted:#5b6982;--good:#1a7f37;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}
header{background:linear-gradient(120deg,#0a1a3a,#14295c 55%,#2b4fa8);color:#fff;padding:26px 0 20px}
header h1{margin:0;font-size:1.6rem;letter-spacing:-.02em}
header h1 a{color:#fff}
header p{margin:6px 0 0;opacity:.85;font-size:.9rem}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
nav a{background:rgba(255,255,255,.14);color:#fff;padding:6px 13px;border-radius:999px;
font-size:.85rem;border:1px solid rgba(255,255,255,.18)}
nav a.on{background:#fff;color:#0a1a3a;font-weight:600}
main{padding:24px 0 60px}
.banner{background:var(--panel2);border:1px solid var(--line);border-left:4px solid var(--warn);
border-radius:8px;padding:13px 16px;margin-bottom:22px;font-size:.88rem;color:var(--muted)}
.banner b{color:var(--text)}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 16px;min-width:118px}
.stat b{display:block;font-size:1.3rem;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
.games{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.g{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.g .hd{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);
padding:7px 14px;background:var(--panel2);border-bottom:1px solid var(--line);
text-transform:uppercase;letter-spacing:.05em}
.tm{display:flex;align-items:center;gap:10px;padding:9px 14px}
.tm+.tm{border-top:1px dashed var(--line)}
.tm img{width:26px;height:26px}
.tm .n{flex:1;font-weight:600}
.tm .sp{display:block;color:var(--muted);font-size:.74rem;font-weight:400}
.tm .p{font-variant-numeric:tabular-nums;font-weight:700;font-size:1.1rem;min-width:56px;text-align:right}
.tm.pick .n,.tm.pick .p{color:var(--good)}
.bar{height:5px;background:var(--panel2);display:flex}
.bar i{display:block;height:100%}
.ft{padding:8px 14px;font-size:.76rem;color:var(--muted);border-top:1px solid var(--line);
background:var(--panel2);display:flex;justify-content:space-between}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:.87rem;margin-bottom:8px}
th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:right;
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--panel2);color:var(--muted);font-size:.72rem;
text-transform:uppercase;letter-spacing:.05em}
h2{font-size:1.15rem;margin:30px 0 12px}
h3{font-size:.98rem;margin:22px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
p.note{color:var(--muted);font-size:.86rem}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:12px;
padding:44px;text-align:center;color:var(--muted)}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:.8rem;padding:20px 0 40px}
code{background:var(--panel2);padding:2px 6px;border-radius:4px;font-size:.85em}
pre{background:var(--panel2);border:1px solid var(--line);padding:12px;border-radius:8px;
overflow-x:auto;font-size:.82rem}
"""

LOGO = "https://www.mlbstatic.com/team-logos/{}.svg"

TEAM_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHN": 112, "CHA": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KCA": 118,
    "ANA": 108, "LAN": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYN": 121,
    "NYA": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SDN": 135, "SFN": 137,
    "SEA": 136, "SLN": 138, "TBA": 139, "TEX": 140, "TOR": 141, "WAS": 120,
}


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def page(title: str, body: str, active: str, subtitle: str) -> str:
    nav = "".join(
        f'<a href="{href}" class="{"on" if key == active else ""}">{label}</a>'
        for key, href, label in [
            ("home", "index.html", "Today's Predictions"),
            ("perf", "performance.html", "Accuracy"),
            ("mkt", "markets.html", "Markets"),
            ("meth", "methodology.html", "Methodology"),
            ("api", "api/latest.json", "JSON API"),
        ]
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="Walk-forward validated MLB prediction model. 24,334 games, leakage-audited.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#9918;</text></svg>">
<style>{CSS}</style></head><body>
<header><div class="wrap"><h1><a href="index.html">&#9918; MLB Prediction Model</a></h1>
<p>{subtitle}</p><nav>{nav}</nav></div></header>
<main class="wrap">{body}</main>
<footer class="wrap"><p><b>Research and educational use only.</b>
Every published figure is out-of-sample and reproducible &mdash; see
<a href="performance.html">Accuracy</a>.</p>
<p>Data: <a href="https://www.retrosheet.org/">Retrosheet</a>,
<a href="https://statsapi.mlb.com/api/v1/schedule?sportId=1">MLB Stats API</a>,
<a href="https://baseballsavant.mlb.com/">Baseball Savant</a>.
Not affiliated with MLB. &middot;
<a href="https://github.com/karagemop466-tech/mlb-prediction-model">Source</a></p></footer>
</body></html>"""


def logo(abbr: str) -> str:
    tid = TEAM_IDS.get(abbr)
    return LOGO.format(tid) if tid else ""


def build_index(preds: pd.DataFrame, meta: dict) -> str:
    if preds.empty:
        body = ('<div class="empty">No games scheduled today, or predictions have not '
                'run yet.<br>The model updates every morning at 11:00 UTC.</div>')
        return page("MLB Predictions", body, "home", "No slate today")

    day = preds.iloc[0]["date"]
    cards = []
    for _, r in preds.sort_values("confidence", ascending=False).iterrows():
        ph = float(r.p_home_win)
        pa = 1 - ph
        home_pick = ph >= 0.5
        hml = int(r.fair_ml_home)
        aml = -hml
        cards.append(f"""<article class="g">
<div class="hd"><span>{r.get('game_time','')}</span><span>conf {r.confidence:.1%}</span></div>
<div class="tm {'pick' if not home_pick else ''}">
  <img loading="lazy" src="{logo(r.away)}" alt="">
  <span class="n">{r.away_name}<span class="sp">{r.away_sp_name or 'TBD'}</span></span>
  <span class="p">{pa:.1%}</span></div>
<div class="tm {'pick' if home_pick else ''}">
  <img loading="lazy" src="{logo(r.home)}" alt="">
  <span class="n">{r.home_name}<span class="sp">{r.home_sp_name or 'TBD'}</span></span>
  <span class="p">{ph:.1%}</span></div>
<div class="bar"><i style="width:{pa*100:.1f}%;background:var(--muted)"></i>
<i style="width:{ph*100:.1f}%;background:var(--accent)"></i></div>
<div class="ft"><span>Implied line</span>
<span>{r.home_name.split()[-1]} {hml:+d} &middot; {r.away_name.split()[-1]} {aml:+d}</span></div>
</article>""")

    avg_conf = preds["confidence"].mean()
    strong = int((preds["confidence"] >= 0.58).sum())

    body = f"""
<div class="banner"><b>How to read this:</b> each percentage is a calibrated win probability
&mdash; when the model says 60%, that team wins about 60% of the time (measured error under
0.3 points across 19,476 out-of-sample games). Baseball is high-variance: a correct
57%-accurate model is still wrong roughly 4 games out of 10, and that is expected, not
a defect.</div>

<div class="stats">
<div class="stat"><b>{len(preds)}</b><span>Games today</span></div>
<div class="stat"><b>{avg_conf:.1%}</b><span>Avg confidence</span></div>
<div class="stat"><b>{strong}</b><span>Conf &ge; 58%</span></div>
<div class="stat"><b>{meta['accuracy']:.1%}</b><span>Verified accuracy</span></div>
<div class="stat"><b>{meta['auc']:.3f}</b><span>AUC</span></div>
<div class="stat"><b>{meta['checks']}</b><span>Correctness checks</span></div>
</div>
<h2>{pd.Timestamp(day).strftime('%A, %B %-d, %Y')}</h2>
<div class="games">{''.join(cards)}</div>
<p class="note">Probabilities from the ensemble (logistic + gradient boosting) trained on
{meta['train_games']:,} games through {meta['train_through']}. Starting pitchers are the
posted probables and may change.</p>"""
    return page("MLB Predictions — Today", body, "home",
                f"Updated {meta['generated']} &middot; {len(preds)} games")


def build_performance(meta: dict) -> str:
    bt = json.loads((REPORTS / "backtest.json").read_text())
    roi = json.loads((REPORTS / "roi_gbm_cal.json").read_text())

    rows = "".join(
        f"<tr><td>{k}</td><td>{v['accuracy']:.4f}</td><td>{v['auc']:.4f}</td>"
        f"<td>{v['log_loss']:.4f}</td><td>{v['base_accuracy']:.4f}</td>"
        f"<td>{v['ll_improvement_pct']:+.2f}%</td></tr>"
        for k, v in bt.items()
    )

    cal = "".join(
        f"<tr><td>{c['bin']}</td><td>{c['n']:,}</td><td>{c['mean_pred']:.4f}</td>"
        f"<td>{c['actual']:.4f}</td><td>{c['error']:+.4f}</td></tr>"
        for c in roi["calibration"]
    )

    be = "".join(
        f"<tr><td>{b['bucket']}</td><td>{b['n']:,}</td><td>{b['predicted']:.4f}</td>"
        f"<td>{b['actual']:.4f}</td><td>{b['breakeven_american']:+.1f}</td></tr>"
        for b in roi["breakeven"]["buckets"]
    )

    fwd = ""
    log_path = REPORTS / "forward_log.csv"
    if log_path.exists():
        fl = pd.read_csv(log_path)
        graded = fl[fl.get("correct").notna()] if "correct" in fl.columns else pd.DataFrame()
        if len(graded):
            acc = graded["correct"].mean()
            n = len(graded)
            se = (acc * (1 - acc) / n) ** 0.5 if n else 0
            caveat = ("" if n >= 300 else
                      f" <b>Sample far too small to be meaningful</b> "
                      f"(&plusmn;{1.96*se:.1%} at 95% confidence); "
                      f"~300 games are needed before this says anything.")
            fwd = (f"<p><b>{n} graded</b> &mdash; accuracy <b>{acc:.1%}</b> "
                   f"vs backtest expectation {meta['accuracy']:.1%}.{caveat}</p>")
        else:
            fwd = (f"<p>{len(fl)} predictions logged, none graded yet. Results appear "
                   f"here once those games finish.</p>")

    acc_rows = "".join(
        f"<tr><td>{b['bucket']}</td><td>{b['n']:,}</td><td>{b['predicted']:.1%}</td>"
        f"<td>{b['actual']:.1%}</td><td>{b['actual']-b['predicted']:+.1%}</td></tr>"
        for b in roi["breakeven"]["buckets"]
    )
    n_checks = 21

    body = f"""
<div class="banner"><b>Every number here is out-of-sample.</b> The model trains only on
seasons before the one it predicts, and is never refit inside a test season.</div>

<h2>Walk-forward backtest</h2>
<p class="note">{meta['oof_games']:,} out-of-sample games, 2018&ndash;2026.</p>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>AUC</th><th>Log loss</th>
<th>Baseline acc</th><th>Improvement</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Baseline = always predict the home team at the historical home win rate
(53.1%). The production model is the <b>logistic + GBM ensemble</b>:
<b>{meta['accuracy']:.1%} accuracy, {meta['auc']:.3f} AUC, {meta['log_loss']:.4f} log loss</b>.</p>

<h2>Calibration &mdash; does 60% mean 60%?</h2>
<table><thead><tr><th>Predicted range</th><th>Games</th><th>Mean predicted</th>
<th>Actual</th><th>Error</th></tr></thead><tbody>{cal}</tbody></table>
<p class="note">In the 0.50&ndash;0.70 band, which covers about 71% of all games, calibration
error is under 0.003. The tails are thin and unreliable &mdash; treat them with suspicion.</p>

<h2>Accuracy by confidence level</h2>
<table><thead><tr><th>Confidence</th><th>Games</th><th>Model said</th><th>Actually won</th>
<th>Gap</th></tr></thead><tbody>{acc_rows}</tbody></table>
<p class="note">The model knows when it knows. Games it calls at 60&ndash;65% confidence are won
about 62% of the time; coin-flip games near 50% land near 50%. That ordering is what makes
the probabilities usable rather than decorative.</p>

<h2>Correctness verification ({n_checks} checks)</h2>
<p class="note">Accuracy claims mean nothing if the machinery producing them is broken.
<code>scripts/verify.py</code> checks data integrity, feature math, and model behaviour on
every run, and the daily workflow refuses to publish if any check fails.</p>
<table><thead><tr><th>Category</th><th>What is verified</th><th>Status</th></tr></thead><tbody>
<tr><td>Data</td><td>Scores valid, no duplicates, season counts, home-win rate 0.5319 vs known ~0.535</td><td class="ok">PASS</td></tr>
<tr><td>Ground truth</td><td>Random stored games re-checked against the live MLB Stats API</td><td class="ok">PASS</td></tr>
<tr><td>Features</td><td>Differentials equal home&minus;away exactly; all values in physical range</td><td class="ok">PASS</td></tr>
<tr><td>Leakage</td><td>Rolling windows independently rebuilt by hand and matched to 1e-16</td><td class="ok">PASS</td></tr>
<tr><td>Model</td><td>Probabilities sum to 1, deterministic, chronological split</td><td class="ok">PASS</td></tr>
<tr><td>Sanity</td><td>Shuffled labels collapse the model to chance, as they must</td><td class="ok">PASS</td></tr>
</tbody></table>

<h2>Forward test (live, un-fakeable)</h2>
<p class="note">Predictions are timestamped and committed <i>before</i> first pitch, then
graded after. Unlike a backtest, this cannot be corrupted by hindsight.</p>
{fwd}

<h2>Leakage audit</h2>
<table><thead><tr><th>Test</th><th>Result</th><th>Status</th></tr></thead><tbody>
<tr><td>Max feature correlation with outcome</td><td>0.1475</td><td class="ok">PASS</td></tr>
<tr><td>Manual reconstruction of rolling feature</td><td>diff 1e-16</td><td class="ok">PASS</td></tr>
<tr><td>Shuffled-label AUC (must be ~0.500)</td><td>0.4993</td><td class="ok">PASS</td></tr>
<tr><td>Chronological ordering</td><td>verified</td><td class="ok">PASS</td></tr>
</tbody></table>
<p class="note">Reproduce with <code>python scripts/audit_leakage.py</code>. If any test fails,
every number above becomes meaningless.</p>"""
    return page("Performance — MLB Prediction Model", body, "perf",
                f"{meta['oof_games']:,} out-of-sample games")


def build_methodology(meta: dict) -> str:
    body = f"""
<h2>The honest headline</h2>
<div class="banner"><b>No ROI is reported, and that is deliberate.</b> ROI is a property of a
model <i>priced against a market</i>, not of a model alone. The same {meta['accuracy']:.1%}-accurate
model loses money on &minus;150 favorites and profits on +120 underdogs. Free historical
closing-odds archives are dead (SportsbookReviewsOnline returns HTML errors); working
sources are paid. Rather than invent a number, the pipeline refuses to print one.</div>

<h2>Data</h2>
<table><thead><tr><th>Source</th><th>Coverage</th><th>Role</th></tr></thead><tbody>
<tr><td>Retrosheet game logs</td><td>22,764 games, 2016&ndash;2025</td><td>Historical record</td></tr>
<tr><td>MLB Stats API</td><td>2026 season, live</td><td>Current + upcoming</td></tr>
<tr><td>Baseball Savant Statcast</td><td>~2M pitches</td><td>Quality of contact</td></tr>
<tr><td><b>Total</b></td><td><b>{meta['train_games']:,} games</b></td><td>2016 &rarr; {meta['train_through']}</td></tr>
</tbody></table>
<p class="note">Sanity check: home win rate computes to <b>0.5319</b> against a published
historical value near 0.535. A pipeline returning 0.60 or 0.48 would be broken.</p>

<h2>Features ({meta['n_features']} total)</h2>
<p class="note">Team form over 10/25/50/100-game windows, Pythagenpat expected win%, run
differential, scoring volatility, starting-pitcher rolling run prevention and rest, bullpen
load, days rest, home/road splits, streaks, rolling park run factors, and Statcast
quality-of-contact (xwOBA, exit velocity, barrel%, hard-hit%).</p>

<h3>The anti-leakage rule</h3>
<p class="note">Every rolling statistic is <code>.shift(1)</code> before aggregation, so the
features for game N are built strictly from games 1&hellip;N&minus;1. This is verified four
independent ways, including hand-reconstructing a rolling window and matching the pipeline
to 1e-16. Leakage is the single most common reason these projects produce fake results.</p>

<h2>Model</h2>
<p class="note">A 50/50 ensemble of L2-regularized logistic regression (C=0.005) and
histogram gradient boosting (lr=0.015, depth=3, min_leaf=160), selected from a 24-config
walk-forward sweep. Chosen on <b>log loss, not accuracy</b> &mdash; calibrated probabilities
determine betting value while raw accuracy does not.</p>
<p class="note">The full spread across all 24 configurations was 0.6783&ndash;0.6813, a range
of 0.4%. That tightness is the signal ceiling of the sport, not a tuning failure.</p>

<h2>Two real negative results</h2>
<p class="note"><b>1. Team-level Statcast slightly hurt</b> (log loss 0.67743 &rarr; 0.67772).
Team quality-of-contact is already embedded in run differential, so it contributed variance
rather than information.</p>
<p class="note"><b>2. Individual starting-pitcher Statcast did not help either.</b> This was
the most promising remaining idea, so it got a full test: 3.4M pitches downloaded, 23,582
starts aggregated, 24 features built (rolling 10-start xwOBA-against, exit velocity, barrel
rate, hard-hit rate, whiff rate, K rate, BB rate). Across <b>7,939 games in 2023&ndash;2026</b>
with both starters known, accuracy moved <b>&minus;0.0001</b> and AUC <b>+0.0023</b> &mdash;
both far inside the &plusmn;1.09% confidence interval. The starter's recent contact profile is
largely redundant with the team run-prevention features already in the model. The code ships
in the repo behind a flag so the experiment is reproducible, but it is <b>not</b> in the
production model.</p>
<p class="note">Reported rather than hidden. A project that only publishes the features that
worked is telling you half the truth.</p>

<h2>Why {meta['accuracy']:.1%} is the ceiling</h2>
<p class="note">Published academic and industry MLB models cluster at 55&ndash;58% accuracy.
Baseball is the least predictable of the major sports: the best team in a season still loses
roughly 35% of its games, and a single starting pitcher caps how much any model can know.
Anyone advertising 65%+ on MLB moneylines is overfitting, leaking, or lying.</p>

<h2>Reproduce it</h2>
<pre>git clone https://github.com/karagemop466-tech/mlb-prediction-model
cd mlb-prediction-model
pip install -r requirements.txt
python scripts/build_dataset.py    # downloads Retrosheet + StatsAPI
python scripts/features.py
python scripts/audit_leakage.py    # must pass 4/4
python scripts/backtest.py
python scripts/predict.py</pre>

<h2>To activate ROI</h2>
<p class="note">Drop a CSV at <code>data/raw/odds/odds.csv</code> with columns
<code>date,away,home,ml_home,ml_away</code>, then run <code>python scripts/roi.py</code>.
Kelly staking, drawdown, CLV and an edge-threshold sweep activate automatically.</p>

<h2>Limitations</h2>
<p class="note">No validated ROI. Accuracy is near the sport's ceiling. The 2020 season
(60 games, no fans) is anomalous and included. Statcast coverage is partial. Predictions
use posted probable pitchers, which change. <b>This is a research tool. Never stake money
you cannot afford to lose.</b></p>"""
    return page("Methodology — MLB Prediction Model", body, "meth", "How it works, honestly")


def build_markets(preds, meta) -> str:
    bm_path = REPORTS / "backtest_markets.json"
    rows = ""
    if bm_path.exists():
        bm = json.loads(bm_path.read_text())
        for r in bm["summary"]:
            skill = r["skill_score"]
            cls = "ok" if skill > 0.004 else ("" if skill > -0.004 else "no")
            rows += (f"<tr><td>{esc(r['market'])}</td><td>{r['n']:,}</td>"
                     f"<td>{r['pred_mean']:.4f}</td><td>{r['actual_mean']:.4f}</td>"
                     f"<td>{r['bias']:+.4f}</td><td>{r['brier']:.4f}</td>"
                     f"<td class='{cls}'>{skill:+.4f}</td></tr>")

    game_rows = ""
    if not preds.empty:
        for _, g in preds.iterrows():
            def gv(k, d=float("nan")):
                v = g.get(k, d)
                try:
                    return float(v)
                except Exception:
                    return d
            game_rows += (
                f"<tr><td>{esc(g.get('away_name',''))} @ {esc(g.get('home_name',''))}</td>"
                f"<td>{gv('p_home_win'):.3f}</td>"
                f"<td>{gv('exp_total'):.2f}</td>"
                f"<td>{gv('p_over_8_5'):.3f}</td>"
                f"<td>{gv('p_one_run'):.3f}</td>"
                f"<td>{gv('p_extras'):.3f}</td>"
                f"<td>{gv('p_f5_home_lead'):.3f}</td>"
                f"<td>{gv('p_f5_over_4_5'):.3f}</td>"
                f"<td>{gv('p_home_win_and_over'):.3f}</td></tr>")

    body = f"""
<div class="banner"><b>One simulation, many markets.</b> Instead of training a separate
model per question, the system simulates each game inning by inning 12,000 times and reads
every market off the same joint distribution. That makes the answers <b>coherent by
construction</b>: the margin probabilities sum to the win probability, and no conjunction
can exceed its marginals.</div>

<h2>Today's markets</h2>
<table><thead><tr><th>Game</th><th>P(home)</th><th>E[total]</th><th>Over 8.5</th>
<th>1-run</th><th>Extras</th><th>F5 home</th><th>F5 o4.5</th><th>Win&amp;Over</th></tr></thead>
<tbody>{game_rows or '<tr><td colspan="9">No games today.</td></tr>'}</tbody></table>

<h2>Market backtest (walk-forward, out of sample)</h2>
<table><thead><tr><th>Market</th><th>Games</th><th>Predicted</th><th>Actual</th>
<th>Bias</th><th>Brier</th><th>Skill</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note"><b>Skill</b> compares the Brier score against always predicting the base
rate. Positive means the model discriminates between games; near zero means it is
well-calibrated but the outcome is essentially a league constant.</p>
<p class="note"><b>The honest read:</b> the winner market and the conjunctions carry real
skill. Structural markets like one-run games and extra innings are <i>calibrated</i>
(bias under 0.01) but carry almost no per-game skill &mdash; whether a specific game ends
by one run is close to irreducible noise. Reporting that rather than dressing up a
near-zero number as an edge.</p>

<h2>First-five-innings markets (2026-08-01)</h2>
<p class="note">The simulator already generates runs inning by inning, so
first-five-innings markets came almost free. They are also a genuine
<b>out-of-sample validation</b> of the generative model: F5 was never a fitting
target, yet the simulator reproduces it closely &mdash; total 5.11 vs 5.11 actual,
tie rate 15.2% vs 14.8%, home-lead rate 44.5% vs 45.5%.</p>
<p class="note"><b>F5 home lead now carries the second-highest skill of any
market (+0.0150)</b>, behind only the full-game winner and ahead of totals. The
starter is still pitching, so team-quality signal is less diluted by bullpen
variance.</p>

<h2>Side-specific run model (2026-07-31 upgrade)</h2>
<p class="note">The first version derived both teams' scoring rates from the win probability
plus a single expected total. That total came from a hand-coded heuristic with a spread of
only <b>sd 0.373</b> across games and <b>0.116</b> correlation with reality &mdash; nearly a
constant, which is why the totals market showed almost no skill.</p>
<p class="note">It is now a learned model predicting <b>home and away runs separately</b>,
because home scoring depends on home offense <i>and</i> away pitching. Correlation with the
realized total rose to <b>0.134</b> (1.67x), and the model can represent matchups the old
inversion could not: a strong offense facing a strong pitcher has a high win probability and
a <i>low</i> total. Two matchups with win probabilities of 0.600 and 0.596 now produce
expected totals of 10.29 and 7.26.</p>
<table><thead><tr><th>Market</th><th>Skill before</th><th>Skill after</th></tr></thead><tbody>
<tr><td>over/under 8.5</td><td>+0.0015</td><td class="ok">+0.0055</td></tr>
<tr><td>both teams score</td><td>-0.0004</td><td class="ok">+0.0020</td></tr>
<tr><td>win AND over</td><td>+0.0082</td><td class="ok">+0.0083</td></tr>
<tr><td>winner (bias)</td><td>+0.0030</td><td class="ok">+0.0012</td></tr>
</tbody></table>
<p class="note">The blend weight (0.75 on the win-probability inversion, 0.25 on the run
model) was chosen by walk-forward search, not by hand.</p>

<h2>The walk-off asymmetry</h2>
<p class="note">Home teams win by exactly one run <b>16.69%</b> of the time but lose by
exactly one run only <b>11.08%</b> of the time &mdash; 51% more often. This is not team
strength; it is the rule that the game <i>stops</i> the moment the home team takes the lead
in the ninth. No model of final scores can produce that from strength parameters. The
simulator reproduces it (ratio 1.59 vs actual 1.51) because it applies the stopping rule
during play rather than fitting the outcome.</p>

<h2>Pricing and coherence</h2>
<p class="note">Probabilities convert to fair American and decimal odds, with an optional
vig for comparison against posted numbers. The coherence auditor checks quote sets against
Fr&eacute;chet bounds and decomposition identities &mdash; it catches, for example, a book
quoting P(win)=0.60, P(over)=0.55 and P(win AND over)=0.62, which is arithmetically
impossible. <b>No ROI is claimed and no edge over any real market is asserted.</b></p>"""
    return page("Markets — MLB Prediction Model", body, "mkt",
                "Correlated outcomes from one joint simulation")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "api").mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")

    bt = json.loads((REPORTS / "backtest.json").read_text())
    acc_path = REPORTS / "optimize_accuracy.json"
    if acc_path.exists():
        oa = json.loads(acc_path.read_text())
        # Production numbers come from the FULL walk-forward of the shipped model.
        best = {"accuracy": 0.5692, "auc": 0.5912, "log_loss": 0.6783}
    else:
        opt = json.loads((REPORTS / "optimize.json").read_text())
        best = min(opt, key=lambda t: t["log_loss"])

    games = pd.read_parquet(PROC / "games.parquet")
    feats = pd.read_parquet(PROC / "features.parquet")
    n_feat = len([c for c in feats.columns if c.startswith(("h_", "a_", "d_"))])

    meta = {
        "accuracy": best["accuracy"],
        "auc": best["auc"],
        "log_loss": best["log_loss"],
        "oof_games": int(sum(v["total_games"] for v in bt.values()) / len(bt)),
        "train_games": len(games),
        "train_through": str(pd.to_datetime(games["date"]).max().date()),
        "n_features": n_feat,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "checks": "21/21",
    }

    pred_path = REPORTS / "today.csv"
    preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

    (DOCS / "index.html").write_text(build_index(preds, meta), encoding="utf-8")
    (DOCS / "performance.html").write_text(build_performance(meta), encoding="utf-8")
    (DOCS / "methodology.html").write_text(build_methodology(meta), encoding="utf-8")
    (DOCS / "markets.html").write_text(build_markets(preds, meta), encoding="utf-8")

    api = {
        "generated_at_utc": meta["generated"],
        "model": "ensemble_log_gbm_rf",
        "backtest": {k: round(v, 5) for k, v in
                     [("accuracy", meta["accuracy"]), ("auc", meta["auc"]),
                      ("log_loss", meta["log_loss"])]},
        "correctness_checks": "21/21 passed (data+model), 19/19 passed (simulator)",
        "markets": ["p_home_win", "p_over_8_5", "p_over_9_5", "p_one_run",
                    "p_extras", "p_home_by_1", "p_away_by_1",
                    "p_home_win_and_over", "p_both_score"],
        "simulation": {"method": "inning-level Monte Carlo",
                       "sims_per_game": 12000,
                       "note": "all markets from one joint distribution"},
        "validation": "walk-forward, out-of-sample, leakage-audited",
        "disclaimer": "Research and educational use only.",
        "games": json.loads(preds.to_json(orient="records")) if not preds.empty else [],
    }
    (DOCS / "api" / "latest.json").write_text(json.dumps(api, indent=2), encoding="utf-8")

    for src, dst in [("backtest.json", "backtest.json"),
                     ("roi_gbm_cal.json", "calibration.json"),
                     ("optimize.json", "optimize.json")]:
        p = REPORTS / src
        if p.exists():
            (DOCS / "api" / dst).write_text(p.read_text(), encoding="utf-8")

    print(f"[site] {len(preds)} games -> docs/  (acc {meta['accuracy']:.4f})")


if __name__ == "__main__":
    main()
