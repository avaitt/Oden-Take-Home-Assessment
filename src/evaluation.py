"""
evaluation.py · How rules are scored, selected and audited.

    §6 Metrics: lift / precision / coverage, cluster bootstrap, circular-shift null
    §7 Selection protocol and audits: the dev/holdout protocol, what happens before failures, the audits
    §8 ML benchmark: ML as a ceiling and an idea generator, not a detector
    §9 Saving results: each notebook (01-05) saves its key results to outputs/<notebook>.json for notebook 06

Part of a 3-file set (data -> detectors -> evaluation). Notebooks import from these
files after adding ../src to the path; each notebook's import cell lists exactly what it uses.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    COMPONENTS,
    Config,
    ERRORS,
    H_GRID,
    Masks,
    Panel,
    SIGNALS,
    W_GRID,
    build_masks,
)
from detectors import (
    FeatureBank,
    build_concepts,
    fire,
    rolling_mean,
    run_at_least,
    short_rule,
    thresholds,
)


# ================================================================================================
# §6 · METRICS
#     lift / precision / coverage, cluster bootstrap, circular-shift null
# ================================================================================================
# Lift, precision and coverage exactly as defined in the brief, plus the
# statistics that tell us how much to trust them.
#
#     Lift       firing rate inside W windows / firing rate in normal operation.
#                (the normal rate uses +0.5 / +1 smoothing so a detector that
#                never fires in normal operation gets a large but finite lift.)
#     Precision  of all alert ONSETS (off->on switches), the share followed by a
#                target event within H+W hours.
#     Coverage   share of evaluable events with >= 1 firing hour in their W window.
#
# We compute everything from per-MACHINE sufficient statistics. That makes the
# cluster bootstrap (resampling whole machines) a cheap re-summation, which is
# the honest way to get intervals when events cluster within machines.


# columns of the per-machine statistics array
COLS = ("fW", "nW", "fN", "nN", "cov", "nE", "on", "tp", "base_hit", "base_n")


C = {c: i for i, c in enumerate(COLS)}


def machine_stats(fire: np.ndarray, mk: Masks):
    M, T = fire.shape
    st = np.zeros((M, len(COLS)))
    st[:, C["fW"]] = (fire & mk.in_w).sum(1)
    st[:, C["nW"]] = mk.in_w.sum(1)
    st[:, C["fN"]] = (fire & mk.normal).sum(1)
    st[:, C["nN"]] = mk.normal.sum(1)

    # coverage: any firing hour in each evaluable event's own window
    ev = mk.ev[mk.ev.evaluable]
    live = fire & ~mk.recov
    cs = np.concatenate([np.zeros((M, 1), np.int64), np.cumsum(live, axis=1)], axis=1)
    m = ev.m.to_numpy()
    ws = np.clip(ev.ws.to_numpy(), 0, T - 1)
    we = np.clip(ev.we.to_numpy(), 0, T - 1)
    hits = cs[m, we + 1] - cs[m, ws]
    covered = hits > 0
    st[:, C["cov"]] = np.bincount(m, covered, minlength=M)
    st[:, C["nE"]] = np.bincount(m, minlength=M)

    # alert onsets (off -> on) and whether an event follows within H+W
    prev = np.zeros_like(fire)
    prev[:, 1:] = fire[:, :-1]
    onset = fire & ~prev & mk.onset_ok
    hit = mk.next_lag <= mk.horizon
    st[:, C["on"]] = onset.sum(1)
    st[:, C["tp"]] = (onset & hit).sum(1)
    # precision of a detector that fires at random = share of judgeable hours
    # that sit within H+W before an event (the base rate to beat)
    st[:, C["base_hit"]] = (mk.onset_ok & hit).sum(1)
    st[:, C["base_n"]] = mk.onset_ok.sum(1)

    return st


def summarize(st: np.ndarray) -> dict:
    """Pool machine rows (st is (n, C) or (B, n, C)) into headline metrics."""
    s = st.sum(axis=-2)
    g = lambda c: s[..., C[c]]
    # Only the DENOMINATOR is smoothed. (Smoothing both made a detector that
    # never fires at all score lift = nN/nW ~ 47 - caught in the smoke test.)
    rate_w = g("fW") / g("nW")
    rate_n = (g("fN") + 0.5) / (g("nN") + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(g("on") > 0, g("tp") / g("on"), np.nan)
        cov = g("cov") / g("nE")
        f1 = np.where((prec + cov) > 0, 2 * prec * cov / (prec + cov), 0.0)
        base = g("base_hit") / g("base_n")
        # alarm burden: onsets per machine per 30 days of judgeable time
        alerts_pm = g("on") / g("base_n") * 720
    return dict(lift=rate_w / rate_n, precision=prec, coverage=cov, f1=np.nan_to_num(f1), base_precision=base,
                alerts_per_machine_month=alerts_pm, n_events=g("nE"), n_alerts=g("on"))


def bootstrap(st: np.ndarray, rows: np.ndarray, B: int, rng: np.random.Generator) -> dict:
    """95% cluster-bootstrap intervals: resample machines with replacement."""
    samp = rng.choice(rows, size=(B, len(rows)), replace=True)
    out = summarize(st[samp])
    ci = {}
    for k in ("lift", "precision", "coverage", "f1"):
        v = out[k]
        v = v[np.isfinite(v)]
        ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return ci


def shift_null(fire: np.ndarray, mk: Masks, rows: np.ndarray, n: int,
               rng: np.random.Generator, min_shift: int = 24 * 7) -> dict:
    """Circular-shift null distribution.

    Each machine's alert series is rotated in time by a random offset (>= 1
    week), which keeps the detector's firing rate and its burstiness
    (autocorrelation) but destroys any real alignment with the events. If the
    real alignment is no better than these rotations, the detector is luck.
    """
    M, T = fire.shape
    tg = np.arange(T)
    res = {k: np.empty(n) for k in ("lift", "precision", "coverage", "f1")}
    sub = fire[rows]
    for i in range(n):
        off = rng.integers(min_shift, T - min_shift, size=len(rows))
        idx = (tg[None, :] - off[:, None]) % T
        shifted = np.zeros_like(fire)
        shifted[rows] = np.take_along_axis(sub, idx, axis=1)
        s = summarize(machine_stats(shifted, mk)[rows])
        for k in res:
            res[k][i] = s[k]
    return res


def null_pvalue(obs: float, null: np.ndarray) -> tuple[float, float]:
    """One-sided empirical p (with +1 correction) and a z-score vs the null."""
    p = (1 + np.sum(null >= obs)) / (1 + len(null))
    sd = null.std()
    z = (obs - null.mean()) / sd if sd > 0 else np.inf
    return float(p), float(z)


# ================================================================================================
# §7 · SELECTION PROTOCOL AND AUDITS
#     the dev/holdout protocol, what happens before failures, the audits
# ================================================================================================
# The protocol, end to end. Read this file top to bottom to see every decision.
#
# SELECTION PROTOCOL (the guard against "test many detectors, keep the lucky ones")
# ---------------------------------------------------------------------------
#  0. Split machines once into DEV and HOLDOUT (50/50, stratified by model).
#  1. Sweep: every concept is implemented many ways (655 variants in total).
#     Thresholds always come from normal-operation hours (global/model scopes:
#     dev machines only).
#  2. Tune on DEV: each concept keeps the variant with the best DEV F1, where
#     F1 = harmonic mean of precision and coverage. Why F1: lift alone rewards
#     detectors that almost never fire; coverage alone rewards detectors that
#     are always on; precision alone ignores missed events. F1 punishes all
#     three failure modes. Lift is used as a sanity gate instead.
#  3. Gate on DEV:
#       a. circular-shift null test on F1, Bonferroni-corrected over concepts
#       b. lift >= 2  (fires at least twice as often before events as normally)
#       c. coverage >= 5% (catches a meaningful share of events)
#       d. causal check: re-fit thresholds on Jan-Jun normal hours only and
#          score Jul-Dec; the detector must keep >= half of the F1 it gets on
#          Jul-Dec with full-year thresholds. Added after the "overdue
#          maintenance" counter (C07) passed a-c with lift ~1,800 yet collapsed
#          to F1 ~0 when only past data set its threshold: "normal" excludes
#          pre-failure hours, so a counter's normal maximum quietly encodes
#          where the failures are.
#  4. Rank survivors by DEV F1 and keep up to three, skipping any detector that
#     is essentially the SAME ALARM as a higher-ranked kept one (Jaccard overlap
#     of firing hours >= 0.8 on dev). We deliberately do not call a detector
#     redundant just because it catches the same events: a drift alarm and an
#     error alarm can catch the same failure a day apart, and that earlier
#     warning is the point.
#  5. Only now look at HOLDOUT: report every concept's holdout metrics with
#     cluster-bootstrap 95% intervals. Nothing is re-chosen after this point.


# Component -> {signal, direction, errors}, learned on DEV machines (learn_mapping) and set by the
# notebooks. Module-level so every concept builder sees the same one.
MAPPING: dict = {}


KEYS = ("lift", "precision", "coverage", "f1", "base_precision", "alerts_per_machine_month",
        "n_events", "n_alerts")


def _clean(d: dict) -> dict:
    out = {}
    for k in KEYS:
        v = float(d[k])
        out[k] = None if not np.isfinite(v) else round(v, 4)
    return out


def evaluate(f: np.ndarray, mk: Masks, dev, hold) -> tuple[dict, dict, np.ndarray]:
    st = machine_stats(f, mk)
    return _clean(summarize(st[dev])), _clean(summarize(st[hold])), st


def lead_times(f: np.ndarray, mk: Masks, rows) -> list:
    """Earliest alert inside each covered event's window, in hours before the event."""
    ev = mk.ev[mk.ev.evaluable & mk.ev.m.isin(rows)]
    live = f & ~mk.recov
    out = []
    for m, ws, we, t in zip(ev.m, ev.ws, ev.we, ev.t):
        hit = np.flatnonzero(live[m, max(ws, 0): we + 1])
        if len(hit):
            out.append(int(t - (max(ws, 0) + hit[0])))
    return out


def run_event_kind(panel: Panel, cfg: Config, bank: FeatureBank, dev, hold, n_null: int,
                   verbose: bool = True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    mk = build_masks(panel, cfg)
    concepts = build_concepts(cfg.percentiles, cfg.event, MAPPING)
    fit = mk.normal

    # ---- 1. sweep every variant of every concept ----------------------------------
    sweep, best = [], {}
    for c in concepts:
        for v in c.variants:
            f = fire(v, bank, fit, dev)
            d, h, _ = evaluate(f, mk, dev, hold)
            leaf = v
            while leaf.combine:
                leaf = leaf.children[0]
            row = dict(concept=c.cid, variant=v.variant, transform=leaf.transform, norm=leaf.norm,
                       pct=leaf.pct, k=leaf.k, dev=d, hold=h)
            sweep.append(row)
            if c.cid not in best or d["f1"] > best[c.cid][0]["dev"]["f1"]:
                best[c.cid] = (row, v)

    # ---- 2-3. tuned variant per concept: CIs, null test, lead time -------------------
    n_tests = len(concepts)
    alpha = 0.05 / n_tests
    rows, fires = [], {}
    # causal check: thresholds fitted on Jan-Jun normal hours only, scored on Jul-Dec
    mid = panel.T // 2
    mk_h2 = build_masks(panel, cfg, time_range=(mid, panel.T))
    fit_h1 = mk.normal.copy()
    fit_h1[:, mid:] = False
    for c in concepts:
        row, v = best[c.cid]
        f = fire(v, bank, fit, dev)
        fires[c.cid] = f
        st = machine_stats(f, mk)
        causal_ref = summarize(machine_stats(f, mk_h2)[dev])["f1"]
        causal = summarize(machine_stats(fire(v, bank, fit_h1, dev), mk_h2)[dev])["f1"]
        nd = shift_null(f, mk, dev, n_null, rng)
        p, z = null_pvalue(row["dev"]["f1"], nd["f1"])
        lt = lead_times(f, mk, hold)
        rows.append(dict(
            cid=c.cid, title=c.title, rule=v.rule_text(), rule_short=short_rule(v), variant=v.variant, dev=row["dev"], hold=row["hold"],
            hold_ci=bootstrap(st, hold, cfg.n_bootstrap, rng),
            null_p=p, null_z=round(z, 2), null_f1_mean=round(float(nd["f1"].mean()), 4),
            null_f1_p95=round(float(np.percentile(nd["f1"], 95)), 4),
            causal_f1_ref=round(float(causal_ref), 4), causal_f1=round(float(causal), 4),
            lead_median=float(np.median(lt)) if lt else None,
        ))

    # ---- 4. gates + ranking + redundancy -----------------------------------------
    from scipy.stats import norm as _N
    floor = 1 / (n_null + 1)
    for r in rows:
        why = []
        # An empirical p can never go below 1/(n_null+1). When it sits on that
        # floor (no null draw came close), use the normal tail of the null
        # z-score instead, so the Bonferroni gate is not a function of n_null.
        r["null_p_gate"] = float(_N.sf(r["null_z"])) if r["null_p"] <= floor + 1e-12 else r["null_p"]
        if r["null_p_gate"] >= alpha:
            why.append(f"not distinguishable from chance (shift-null p={r['null_p_gate']:.3g} ≥ {alpha:.4f})")
        if (r["dev"]["lift"] or 0) < 2:
            why.append(f"lift {r['dev']['lift']:.2f} < 2")
        if r["dev"]["coverage"] < 0.05:
            why.append(f"coverage {r['dev']['coverage']:.1%} < 5%")
        if r["causal_f1"] < 0.5 * r["causal_f1_ref"]:
            why.append(f"collapses with past-only thresholds (Jul–Dec F1 {r['causal_f1']:.2f} vs "
                       f"{r['causal_f1_ref']:.2f} with full-year thresholds): its skill leaks from the future")
        r["gate_fail"] = why
    passed = sorted([r for r in rows if not r["gate_fail"]], key=lambda r: -r["dev"]["f1"])
    kept = []
    for r in passed:
        red = None
        a = fires[r["cid"]][dev]
        for k in kept:
            b = fires[k["cid"]][dev]
            jac = (a & b).sum() / max(1, (a | b).sum())
            if jac >= 0.8:
                red = (k["cid"], jac)
                break
        if red:
            r["status"], r["reason"] = "rejected", (f"same alarm as {red[0]} (firing-hour overlap "
                                                    f"Jaccard {red[1]:.2f} ≥ 0.8)")
        elif len(kept) < 3:
            r["status"], r["reason"] = "kept", f"rank {len(kept) + 1} by dev F1 among detectors passing all gates"
            kept.append(r)
        else:
            r["status"], r["reason"] = "rejected", "passed all gates but outranked by three stronger, non-redundant detectors"
    for r in rows:
        if r["gate_fail"]:
            r["status"], r["reason"] = "rejected", "; ".join(r["gate_fail"])

    ev = mk.ev
    out = dict(
        event=cfg.event, H=cfg.H, W=cfg.W, alpha=alpha,
        n_hold_events=int((ev.evaluable & ev.m.isin(hold)).sum()),
        concepts=rows, kept=[r["cid"] for r in kept], sweep=sweep,
    )
    if verbose:
        print(f"  [{cfg.event} {cfg.tag}] {len(sweep)} variants, kept {out['kept']} ({time.time() - t0:.0f}s)")
    out["_fires"] = fires  # in-memory only, dropped before JSON
    out["_mk"] = mk
    return out


def retune_for_window(panel: Panel, cfg: Config, bank: FeatureBank, dev, hold, kept: list, H: int, W: int) -> dict:
    """Re-tune the kept rules for a different window (e.g. the deployment window H=12, W=36).

    For each kept rule: the variant with the best F1 on DEV machines at (H, W), chosen exactly as in the
    selection protocol. Then, on HOLDOUT machines at (H, W): the rule as kept (tuned for cfg's window) next to
    the re-tuned rule, with 95% cluster-bootstrap intervals. `kept` is the "concepts" rows of run_event_kind.
    """
    rng = np.random.default_rng(cfg.seed)
    mk = build_masks(panel, cfg.with_(H=H, W=W))
    concepts = {c.cid: c for c in build_concepts(cfg.percentiles, cfg.event, MAPPING)}
    out = {}
    for r in kept:
        variants = concepts[r["cid"]].variants
        as_kept = next(v for v in variants if v.variant == r["variant"])
        retuned = max(variants, key=lambda v: summarize(machine_stats(fire(v, bank, mk.normal, dev), mk)[dev])["f1"])
        row = dict(cid=r["cid"], kept_variant=as_kept.variant, retuned_variant=retuned.variant,
                   retuned_rule_short=short_rule(retuned))
        for name, v in (("as_kept", as_kept), ("retuned", retuned)):
            f = fire(v, bank, mk.normal, dev)
            st = machine_stats(f, mk)
            lt = lead_times(f, mk, hold)
            row[name] = dict(_clean(summarize(st[hold])), f1_ci=bootstrap(st, hold, cfg.n_bootstrap, rng)["f1"],
                             lead_median=float(np.median(lt)) if lt else None)
        out[r["cid"]] = row
    return dict(H=H, W=W, rules=out)


def lookalike_excursions(panel: Panel, cfg: Config, min_len: int = 24) -> dict:
    """How many sustained excursions look exactly like a pre-failure shift but
    are NOT followed by a failure? This bounds the precision ANY signal-based
    rule can reach.

    For each component we use the signal whose pre-failure shift is largest
    (found from the data, not assumed), standardise it per machine with normal
    stats, smooth over 12h, and cut episodes where the smoothed z exceeds half
    the typical pre-failure shift. An episode "ends in a failure" if a failure
    of that component is logged between 12h before and 24h after its end.
    """
    mk = build_masks(panel, cfg)
    prof = lag_profiles(panel, cfg, lag_max=72)
    out = {}
    for c in COMPONENTS:
        # signal with the biggest mean |z| over lags 0..47 for this component
        s = max(SIGNALS, key=lambda s: abs(np.mean(prof["signals"][c][s][:48])))
        shift = float(np.mean(prof["signals"][c][s][:48]))
        x = panel.X[s]
        mu = np.array([x[m, mk.normal[m]].mean() for m in range(panel.M)])[:, None]
        sd = np.array([x[m, mk.normal[m]].std() for m in range(panel.M)])[:, None]
        z = np.sign(shift) * (x - mu) / sd
        hi = np.nan_to_num(rolling_mean(z, 12)) > abs(shift) / 2
        ev = panel.events[panel.events.comps.map(lambda q: c in q)]
        E = {m: np.sort(g.t.to_numpy()) for m, g in ev.groupby("m")}
        L_f, L_n, err_f, err_n = [], [], [], []
        anyerr = panel.err.sum(axis=2)
        for m in range(panel.M):
            d = np.diff(np.r_[0, hi[m].astype(int), 0])
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            te = E.get(m, np.array([]))
            for a, b in zip(st, en):
                if b - a < min_len:
                    continue
                hit = ((te >= b - 12) & (te <= b + 24)).any()
                # was any error logged during the excursion, before its last
                # 6 hours? (excludes errors logged with the failure itself)
                had_err = bool(anyerr[m, a: max(a + 1, b - 6)].sum() > 0)
                (L_f if hit else L_n).append(int(b - a))
                (err_f if hit else err_n).append(had_err)
        out[c] = dict(signal=s, episodes_ending_in_failure=len(L_f),
                      episodes_without_failure=len(L_n),
                      precision_ceiling=round(len(L_f) / max(1, len(L_f) + len(L_n)), 3),
                      share_with_error_failure=round(float(np.mean(err_f)), 3) if err_f else None,
                      share_with_error_none=round(float(np.mean(err_n)), 3) if err_n else None,
                      # precision ceiling if we only alarm on excursions WITH an error
                      precision_ceiling_with_error=round(sum(err_f) / max(1, sum(err_f) + sum(err_n)), 3))
    return out


def lag_profiles(panel: Panel, cfg: Config, lag_max: int = 168) -> dict:
    """Event-triggered averages: what each signal does in the hours before each
    component's failures, as a z-score vs that machine's normal operation."""
    mk = build_masks(panel, cfg)
    out = {"lags": list(range(lag_max + 1)), "signals": {}}
    z = {}
    for s in SIGNALS:
        x = panel.X[s]
        mu = np.array([x[m, mk.normal[m]].mean() for m in range(panel.M)])[:, None]
        sd = np.array([x[m, mk.normal[m]].std() for m in range(panel.M)])[:, None]
        z[s] = (x - mu) / sd
    ev = panel.events
    lags = np.arange(lag_max + 1)
    for c in COMPONENTS:
        e = ev[ev.comps.map(lambda x: c in x)]
        tt = e.t.to_numpy()[:, None] - lags[None, :]
        ok = tt >= 0
        mm = e.m.to_numpy()[:, None]
        out["signals"][c] = {}
        for s in SIGNALS:
            v = np.where(ok, z[s][mm, np.clip(tt, 0, None)], np.nan)
            out["signals"][c][s] = np.round(np.nanmean(v, 0), 3).tolist()
    return out


def audits(panel, bank, cfg: Config, res_any: dict, dev, hold, rng) -> dict:
    out = {}
    kept = [r for r in res_any["concepts"] if r["status"] == "kept"]
    concepts = {c.cid: c for c in build_concepts(cfg.percentiles, cfg.event, MAPPING)}

    def det_of(r):
        return [v for v in concepts[r["cid"]].variants if v.variant == r["variant"] and v.rule_text() == r["rule"]][0]

    dets = {r["cid"]: det_of(r) for r in kept}

    # 1. time drift: fit thresholds on Jan-Jun normal hours, score Jul-Dec only
    T = panel.T
    mid = T // 2
    mk_full = build_masks(panel, cfg)
    fit_h1 = mk_full.normal.copy(); fit_h1[:, mid:] = False
    mk_h2 = build_masks(panel, cfg, time_range=(mid, T))
    allm = np.arange(panel.M)
    temporal = []
    for cid, d in dets.items():
        f_ref = fire(d, bank, mk_full.normal, dev)
        f_h1 = fire(d, bank, fit_h1, dev)
        a = summarize(machine_stats(f_ref, mk_h2)[allm])
        b = summarize(machine_stats(f_h1, mk_h2)[allm])
        temporal.append(dict(cid=cid, full_year_thresholds=_clean(a), first_half_thresholds=_clean(b)))
    out["temporal"] = temporal

    # 2. recovery guardrail: how much do results depend on the hours muted after a repair?
    rec = []
    for R in (0, 6, 24, 72):
        mk = build_masks(panel, cfg.with_(recovery_h=R))
        for cid, d in dets.items():
            s = summarize(machine_stats(fire(d, bank, mk.normal, dev), mk)[hold])
            rec.append(dict(cid=cid, R=R, **_clean(s)))
    out["recovery"] = rec

    # 3. machine mode changes: by machine model and by age band (all machines; thresholds from dev)
    models = panel.machines.model.to_numpy()
    ages = panel.machines.age.to_numpy()
    reg = []
    for cid, d in dets.items():
        f = fire(d, bank, mk_full.normal, dev)
        st = machine_stats(f, mk_full)
        for mdl in sorted(set(models)):
            rows = np.flatnonzero(models == mdl)
            reg.append(dict(cid=cid, group=mdl, kind="model", **_clean(summarize(st[rows]))))
        for lo, hi in ((0, 6), (6, 12), (12, 16), (16, 21)):
            rows = np.flatnonzero((ages >= lo) & (ages < hi))
            reg.append(dict(cid=cid, group=f"age {lo}-{hi - 1}y", kind="age", **_clean(summarize(st[rows]))))
    out["regimes"] = reg

    # 4. leakage: score the event hour itself (H=0) vs the honest window
    leak = []
    maint_now = panel.maint.any(axis=2)  # "a maintenance record exists this hour"
    for flag in (False, True):
        mk = build_masks(panel, cfg.with_(H=0, W=24, score_event_hour=flag))
        s = summarize(machine_stats(maint_now, mk)[allm])
        leak.append(dict(score_event_hour=flag, **_clean(s)))
    out["leakage"] = leak

    # 5. keeping the lucky ones: pure-noise detectors, pick the best on dev
    out["lucky"] = lucky_detectors(panel, mk_full, dev, hold, rng, n=300)

    return out


def lucky_detectors(panel, mk, dev, hold, rng, n=300) -> dict:
    """Generate n detectors that are pure noise (random AR(1) 'signals' with a
    random smoothing, threshold and run length), select the best by dev F1 like
    a careless analyst would, then check it on holdout."""
    from scipy.signal import lfilter

    M, T = panel.M, panel.T
    devf, holdf = [], []
    for i in range(n):
        phi = rng.uniform(0.0, 0.98)
        x = lfilter([1.0], [1.0, -phi], rng.standard_normal((M, T)), axis=1)
        w = int(rng.choice([1, 6, 24]))
        if w > 1:
            x = rolling_mean(x, w)
        q = float(rng.choice([95, 99, 99.5]))
        thr = thresholds(x, mk.normal, "machine", q, dev, panel.machines.model.to_numpy())
        with np.errstate(invalid="ignore"):
            f = run_at_least((x > thr) & np.isfinite(x), int(rng.choice([1, 3])))
        st = machine_stats(f, mk)
        devf.append(float(summarize(st[dev])["f1"]))
        holdf.append(float(summarize(st[hold])["f1"]))
    devf, holdf = np.array(devf), np.array(holdf)
    b = int(np.argmax(devf))
    return dict(dev_f1=np.round(devf, 4).tolist(), hold_f1=np.round(holdf, 4).tolist(),
                best_dev=round(float(devf[b]), 4), best_hold=round(float(holdf[b]), 4))


def stamp_alignment(panel: Panel, cfg: Config, mapping: dict, before: int = 49, after: int = 24) -> dict:
    """Does a failure stamp hide post-failure data? Hour-by-hour averages around the stamp.

    For each part: its own signal (z-score vs each machine's normal operation, signed so a
    precursor is positive) and its own error codes, averaged over all its failures, from
    `after` hours AFTER the stamp to `before` hours before it. If the stamp were logged up to
    a day late, the last hours before it would look post-failure (signal back to normal, or
    bursts of errors). Returns {part: DataFrame indexed by lag (negative = after the stamp)}.
    """
    mk = build_masks(panel, cfg)
    lags = np.arange(-after, before + 1)
    out = {}
    for c, mp in mapping.items():
        x = panel.X[mp["signal"]]
        mu = np.array([x[m, mk.normal[m]].mean() for m in range(panel.M)])[:, None]
        sd = np.array([x[m, mk.normal[m]].std() for m in range(panel.M)])[:, None]
        z = (1 if mp["direction"] == "high" else -1) * (x - mu) / sd
        e = panel.events[panel.events.comps.map(lambda q: c in q)]
        e = e[(e.t - before >= 0) & (e.t + after < panel.T)]          # full span inside the data
        tt = e.t.to_numpy()[:, None] - lags[None, :]
        mm = e.m.to_numpy()[:, None]
        own = sum(panel.err[:, :, ERRORS.index(k)] for k in mp["errors"])
        out[c] = pd.DataFrame({"own_signal_z": z[mm, tt].mean(0),
                               "own_error_codes_per_failure": own[mm, tt].mean(0)},
                              index=pd.Index(lags, name="hours_before_stamp"))
    return out


def f1_grid(panel: Panel, bank: FeatureBank, base: Config, dev, mapping: dict, cid: str = "C09",
            H_values=H_GRID, W_values=W_GRID) -> pd.DataFrame:
    """Choose the window by ONE number: the dev F1 of the leading detector, re-tuned in every cell.

    For each (H, W): rebuild the scoring window and "normal operation", re-fit every variant's
    thresholds on normal hours, and keep the best variant's F1 on DEV machines. The leading
    detector (C09, matched signature) topped the shortlist at every window tried, so its tuned F1
    measures how good a window is. Holdout machines are not used. ~4 minutes for the default grid.
    Returns an (H x W) table of dev F1.
    """
    concept = [c for c in build_concepts(base.percentiles, base.event, mapping) if c.cid == cid][0]
    out = pd.DataFrame(index=pd.Index(H_values, name="H"), columns=pd.Index(W_values, name="W"), dtype=float)
    for H in H_values:
        for W in W_values:
            mk = build_masks(panel, base.with_(H=H, W=W))
            out.loc[H, W] = max(summarize(machine_stats(fire(v, bank, mk.normal, dev), mk)[dev])["f1"]
                                for v in concept.variants)
    return out


def learn_mapping(panel: Panel, cfg: Config, dev) -> dict:
    """Which signal and which error codes belong to which component - learned
    on DEV machines only, so C09 never sees holdout labels.

    signal: the one with the largest mean |z| over the 48h before that
            component's failures (z vs each machine's normal operation)
    errors: codes whose rate in the 48h before the failures is at least half
            as enriched (vs normal) as the most enriched code. (A fixed
            cut-off of 5x was tried first and wrongly pulled in error5 for
            comp2/comp3 - a side effect of multi-part failures.)
    """
    mk = build_masks(panel, cfg)
    ev = panel.events[panel.events.m.isin(dev)]
    lags = np.arange(48)
    out = {}
    for c in COMPONENTS:
        e = ev[ev.comps.map(lambda q: c in q)]
        tt = e.t.to_numpy()[:, None] - lags[None, :]
        ok = tt >= 0
        mm = e.m.to_numpy()[:, None]
        best, bz = None, 0.0
        for s in SIGNALS:
            x = panel.X[s]
            mu = np.array([x[m, mk.normal[m]].mean() for m in range(panel.M)])
            sd = np.array([x[m, mk.normal[m]].std() for m in range(panel.M)])
            z = np.where(ok, (x[mm, np.clip(tt, 0, None)] - mu[mm]) / sd[mm], np.nan)
            mz = float(np.nanmean(z))
            if abs(mz) > abs(bz):
                best, bz = s, mz
        enr = {}
        for k, er in enumerate(ERRORS):
            x = panel.err[:, :, k]
            before = np.nanmean(np.where(ok, x[mm, np.clip(tt, 0, None)], np.nan))
            normal = x[dev][mk.normal[dev]].mean()
            enr[er] = float(before / normal) if normal > 0 else 0.0
        top = max(enr.values())
        out[c] = dict(signal=best, direction="high" if bz > 0 else "low",
                      errors=[er for er, v in enr.items() if v >= 0.5 * top])
    return out


# ================================================================================================
# §8 · ML BENCHMARK
#     gradient boosting / logistic regression as a ceiling and an idea generator, not a detector
# ================================================================================================
# Trained on DEV machines only; every feature is causal (trailing windows only). Notebook 03 turns the
# scores into alerts the same way as every rule (P99 of normal operation) and explains the purpose.


def feature_cube(panel: Panel, bank: FeatureBank) -> tuple[np.ndarray, list[str]]:
    feats, names = [], []
    for s in SIGNALS:
        for tr in ("raw", "roll6", "roll24", "std24"):
            feats.append(bank.get(s, tr)); names.append(f"{s}_{tr}")
    for e in ERRORS:
        feats.append(bank.get(f"err:{e}", "count24")); names.append(f"{e}_count24")
        feats.append(bank.get(f"err:{e}", "count48")); names.append(f"{e}_count48")
    for c in COMPONENTS:
        feats.append(bank.get(f"age:{c}", "raw")); names.append(f"hours_since_maint_{c}")
    M, T = panel.M, panel.T
    feats.append(np.repeat(panel.machines.age.to_numpy()[:, None], T, 1).astype(float)); names.append("machine_age_years")
    mdl = panel.machines.model.str[-1].astype(int).to_numpy()
    feats.append(np.repeat(mdl[:, None], T, 1).astype(float)); names.append("machine_model")
    return np.stack(feats, axis=-1), names


def train_and_score(panel, bank, mk, dev, hold, rng, kind="hgb"):
    """Return (score cube (M,T), feature names, importances dict).

    kind: "hgb" (all features), "hgb_nocounter" (maintenance counters removed),
    "logreg". The no-counter variant exists because the first ML run leaned
    hardest on hours-since-maintenance - the same leaky counter the causal gate
    rejects (see §7, gate d) - so the full model's score is flattered.
    """
    F, names = feature_cube(panel, bank)
    if kind == "hgb_nocounter":
        keep = [i for i, n in enumerate(names) if not n.startswith("hours_since_maint")]
        F, names = F[..., keep], [names[i] for i in keep]
    M, T, P = F.shape
    sel = np.zeros(M, bool); sel[dev] = True
    pos = mk.in_w & sel[:, None]
    neg = mk.normal & sel[:, None]
    # subsample normal hours (they outnumber window hours ~50:1) for speed;
    # class weights are left at default: we only need a ranking score.
    neg_idx = np.flatnonzero(neg.ravel())
    neg_idx = rng.choice(neg_idx, size=min(len(neg_idx), 40 * pos.sum()), replace=False)
    pos_idx = np.flatnonzero(pos.ravel())
    idx = np.concatenate([pos_idx, neg_idx])
    Xf = F.reshape(-1, P)
    Xtr, ytr = Xf[idx], np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
    ok = np.isfinite(Xtr).all(1)
    Xtr, ytr = Xtr[ok], ytr[ok]
    if kind.startswith("hgb"):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Xtr, ytr)

    Xall = np.nan_to_num(Xf, nan=0.0)
    score = model.predict_proba(Xall)[:, 1].reshape(M, T)
    score[:, :24] = np.nan  # warm-up: rolling features not defined

    # permutation importance measured on DEV machines (window hours vs a sample of normal hours): these
    # importances fed the brainstorm of candidates, so holdout machines must play no part in them
    hs = np.zeros(M, bool); hs[dev] = True
    hp = np.flatnonzero((mk.in_w & hs[:, None]).ravel())
    hn = np.flatnonzero((mk.normal & hs[:, None]).ravel())
    hn = rng.choice(hn, size=min(len(hn), 10 * len(hp)), replace=False)
    Xh = np.nan_to_num(Xf[np.r_[hp, hn]], nan=0.0)
    yh = np.r_[np.ones(len(hp)), np.zeros(len(hn))]
    pi = permutation_importance(model, Xh, yh, scoring="average_precision",
                                n_repeats=3, random_state=0)
    imp = dict(sorted(zip(names, pi.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))
    return score, names, imp


# ================================================================================================
# §9 · SAVING EACH NOTEBOOK'S RESULTS
#     notebooks 01-05 each save their key results to outputs/<notebook>.json; notebook 06 reads them
# ================================================================================================
OUT = Path(__file__).resolve().parents[1] / "outputs"
NOTEBOOKS = ("01_dataset_and_parameters", "02_candidates", "03_ml_benchmark", "04_test_and_select", "05_audit")
# Which saved results each notebook's results are built from (05 audits the detectors that 04 kept).
DEPENDS_ON = {"05_audit": ("04_test_and_select",)}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def save_results(name: str, results: dict, cfg: Config) -> Path:
    """Save one notebook's key results to outputs/<name>.json, stamped with the settings used and the
    time it ran (the time is stored INSIDE the file, because file timestamps can change without a re-run)."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(notebook=name, generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
               config=asdict(cfg), **results)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(out, default=_json_default, ensure_ascii=False))
    print(f"saved {path.relative_to(OUT.parent)}")
    return path


def load_results(name: str) -> dict:
    """Read the results that notebooks/<name>.ipynb saved."""
    path = OUT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(OUT.parent)} not found: run notebooks/{name}.ipynb first")
    return json.loads(path.read_text())


def check_results(names=NOTEBOOKS) -> pd.DataFrame:
    """One row per saved results file: when it ran, with which H and W, and whether it is out of date
    (the code in src/ changed after it ran, or a notebook it depends on was re-run after it)."""
    code_changed = max(f.stat().st_mtime for f in (OUT.parent / "src").glob("*.py"))
    saved = {n: load_results(n) for n in names}
    rows = []
    for n, r in saved.items():
        ran = _dt.datetime.fromisoformat(r["generated_at"])
        stale = []
        if code_changed > ran.timestamp():
            stale.append("code in src/ changed since")
        for d in DEPENDS_ON.get(n, ()):
            if d in saved and _dt.datetime.fromisoformat(saved[d]["generated_at"]) > ran:
                stale.append(f"{d} was re-run since")
        rows.append(dict(notebook=n, ran_at=r["generated_at"].replace("T", " "), H=r["config"]["H"],
                         W=r["config"]["W"], status="re-run: " + "; ".join(stale) if stale else "up to date"))
    out = pd.DataFrame(rows).set_index("notebook")
    if out[["H", "W"]].drop_duplicates().shape[0] > 1:
        print("WARNING: the notebooks were run with different H / W settings; re-run them with the same Config")
    return out
