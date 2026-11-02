"""
build_site.py · Builds the project website in docs/ (served by GitHub Pages).

    python site/build_site.py          # ~5 minutes; run after notebooks 01-06

It writes three things:
  * docs/index.html          the write-up (notebook 06, code hidden)
  * docs/data/rules.json     the three kept rules (C08, C09, C10) scored on the holdout machines for every
                             H, W and event on a grid, for docs/rules.html. Two rule sets: as notebook 04 kept
                             them (tuned for H=0, W=48), and as notebook 04 re-tuned them on dev machines for
                             H=12, W=36 (the recommended deployment window; notebook 04, section 7)
  * docs/data/machines.json  daily sensor means plus errors, replacements and failures for every machine,
                             for docs/machines.html

The rules are the ones notebook 04 kept (read from outputs/04_test_and_select.json): same smoothing, same
threshold scope and level, or the same three rules re-tuned for H=12, W=36. Changing H, W or the event changes which hours count as "normal operation"
(so the thresholds are re-fitted from those hours, as the brief requires) and which alerts count. For a
single-part event, C09 and C10 use only that part's own signal and error codes or age.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import Config, SIGNALS, build_masks, dataset_dir, dev_holdout_split, load_panel  # noqa: E402
from detectors import FeatureBank, build_concepts, fire  # noqa: E402
from evaluation import MAPPING, bootstrap, lead_times, load_results, machine_stats, summarize  # noqa: E402

DOCS = ROOT / "docs"
H_VALUES = tuple(range(0, 73, 3))          # lead time, hours
W_VALUES = tuple(range(6, 97, 6))          # window length, hours
EVENTS = ("any", "comp1", "comp2", "comp3", "comp4")
RULES = ("C09", "C08", "C10")


def _setup(event: str = "any"):
    saved = load_results("04_test_and_select")
    MAPPING.clear(); MAPPING.update(saved["mapping"])
    cfg = Config(**{**saved["config"], "percentiles": tuple(saved["config"]["percentiles"])}).with_(event=event)
    panel = load_panel()
    return saved, cfg, panel, FeatureBank(panel), *dev_holdout_split(panel, cfg.dev_fraction, cfg.seed)


def _score_event(job) -> dict:
    """Score one rule set (cid -> variant) for one event type over the whole H x W grid (own process)."""
    tuning, variants, event = job
    saved, cfg, panel, bank, dev, hold = _setup(event)
    concepts = {c.cid: c for c in build_concepts(cfg.percentiles, event, MAPPING)}
    dets = {cid: next(v for v in concepts[cid].variants if v.variant == variants[cid]) for cid in RULES}
    rng = np.random.default_rng(cfg.seed)
    r4 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)
    cells = {}
    for H in H_VALUES:
        for W in W_VALUES:
            mk = build_masks(panel, cfg.with_(H=H, W=W))
            row = {}
            for cid, d in dets.items():
                f = fire(d, bank, mk.normal, dev)
                st = machine_stats(f, mk)
                s, ci = summarize(st[hold]), bootstrap(st, hold, 200, rng)
                lt = lead_times(f, mk, hold)
                row[cid] = dict(f1=r4(s["f1"]), f1_lo=r4(ci["f1"][0]), f1_hi=r4(ci["f1"][1]),
                                precision=r4(s["precision"]), coverage=r4(s["coverage"]), lift=r4(s["lift"]),
                                alerts_pm=r4(s["alerts_per_machine_month"]), base=r4(s["base_precision"]),
                                lead=float(np.median(lt)) if lt else None, events=int(s["n_events"]))
            cells[f"{H},{W}"] = row
    return dict(tuning=tuning, event=event, cells=cells,
                rules={cid: dict(rule=d.rule_text(), variant=d.variant) for cid, d in dets.items()})


def build_rules() -> None:
    t0 = time.time()
    saved = load_results("04_test_and_select")
    kept = {c["cid"]: c["variant"] for c in saved["kinds"]["any"]["concepts"] if c["status"] == "kept"}
    dep = saved["deployment"]                    # notebook 04, section 7: the rules re-tuned for H=12, W=36 (dev)
    tunings = {f"{dep['H']},{dep['W']}": {cid: dep["rules"][cid]["retuned_variant"] for cid in RULES},
               "0,48": {cid: kept[cid] for cid in RULES}}
    print("rule settings:", tunings)
    jobs = [(t, v, e) for t, v in tunings.items() for e in EVENTS]
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(_score_event, jobs))
    titles = {c["cid"]: c["title"] for c in saved["kinds"]["any"]["concepts"]}
    data, text = {t: {} for t in tunings}, {t: {} for t in tunings}
    for r in results:
        data[r["tuning"]][r["event"]] = r["cells"]; text[r["tuning"]][r["event"]] = r["rules"]
    out = dict(H=H_VALUES, W=W_VALUES, events=EVENTS, rules=RULES, titles=titles, tunings=list(tunings),
               variants=tunings, best=dict(H=0, W=48), recommended=dict(H=12, W=36), data=data, rule_text=text)
    (DOCS / "data" / "rules.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"rules.json: 2 rule sets x {len(EVENTS)} events x {len(H_VALUES) * len(W_VALUES)} windows x 3 rules "
          f"in {time.time() - t0:.0f} s")


def build_machines() -> None:
    panel = load_panel()
    raw = Path(dataset_dir())
    days = pd.DatetimeIndex(panel.times).floor("D")
    daily = {}
    for s in SIGNALS:
        by_day = pd.DataFrame(panel.X[s].T, index=days).groupby(level=0).mean()
        daily[s] = np.round(by_day.to_numpy().T, 2).tolist()      # (machines, days)
    start, end = panel.times[0], panel.times[-1]

    def events(name, col):
        df = pd.read_csv(raw / f"PdM_{name}.csv", parse_dates=["datetime"])
        df = df[(df.datetime >= start) & (df.datetime <= end)]
        return {int(m): [[t.strftime("%Y-%m-%d %H:%M"), str(v)] for t, v in zip(g.datetime, g[col])]
                for m, g in df.groupby("machineID")}

    machines = pd.read_csv(raw / "PdM_machines.csv")
    out = dict(days=[d.strftime("%Y-%m-%d") for d in by_day.index], signals=SIGNALS, daily=daily,
               errors=events("errors", "errorID"), replacements=events("maint", "comp"),
               failures=events("failures", "failure"),
               machines={int(r.machineID): dict(model=r.model, age=int(r.age)) for r in machines.itertuples()})
    (DOCS / "data" / "machines.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"machines.json: {panel.M} machines x {len(out['days'])} days")


def build_writeup() -> None:
    """Notebook 06 as a web page: code cells hidden, outputs and text kept, plus the site's navigation bar."""
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--no-input", "--template", "lab",
                    "--output-dir", str(DOCS), "--output", "index", str(ROOT / "notebooks" / "06_writeup.ipynb")],
                   check=True, capture_output=True)
    page = DOCS / "index.html"
    html = page.read_text()
    nav = (DOCS / "nav.html").read_text().replace('href="index.html"', 'href="index.html" aria-current="page"')
    html = html.replace("<title>06_writeup</title>", "<title>Oden take-home · Write-up</title>")
    html = html.replace("</head>", '<link rel="stylesheet" href="site.css">\n</head>', 1)
    i = html.index(">", html.index("<body")) + 1
    page.write_text(html[:i] + "\n" + nav + "\n" + html[i:])
    print("index.html: write-up")


if __name__ == "__main__":
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    build_writeup()
    build_machines()
    build_rules()
