"""
detectors.py · What a precursor rule is, and the candidate ideas.

    §4 Features and detector rules: causal transforms, thresholds from normal operation, the rule engine
    §5 Candidate precursors: the 19 ideas, each with a story, a first-draft rule and every variant

Part of a 3-file set (data -> detectors -> evaluation). Notebooks import from these
files after adding ../src to the path; each notebook's import cell lists exactly what it uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from data import (
    COMPONENTS,
    ERRORS,
    Panel,
    SIGNALS,
)


# ================================================================================================
# §4 · FEATURES AND DETECTOR RULES
#     causal transforms, thresholds from normal operation, the rule engine
# ================================================================================================
# Features and detectors.
#
# A *detector* here is always a simple, fixed rule of the form
#
#     feature  >  (or <)  the p-th percentile of that feature in NORMAL operation,
#     for k or more consecutive hourly readings
#
# optionally combined with other detectors by OR / AND. Thresholds are fitted
# on normal-operation hours only (never on pre-event windows), as the brief
# requires.
#
# Every feature is CAUSAL: the value at hour t only uses data from hours <= t.
# A rolling mean that looked forward would see the failure coming for free.


# ---------------------------------------------------------------------------------
# Causal feature transforms on (M, T) arrays
# ---------------------------------------------------------------------------------
def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w hours (t-w+1 .. t). First w-1 hours = NaN."""
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1)
    out = np.full_like(x, np.nan, dtype=float)
    out[:, w - 1:] = (c[:, w:] - c[:, :-w]) / w
    return out


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.pad(x.astype(float), ((0, 0), (1, 0))), axis=1)
    out = np.full(x.shape, np.nan)
    out[:, w - 1:] = c[:, w:] - c[:, :-w]
    return out


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    m1 = rolling_mean(x, w)
    m2 = rolling_mean(x * x, w)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0))


def cusum(x: np.ndarray, fit: np.ndarray, direction: str, slack: float = 0.5) -> np.ndarray:
    """One-sided CUSUM on a per-machine standardised signal.

    z is standardised with each machine's NORMAL-operation mean and std. The
    statistic accumulates evidence of a sustained shift and resets towards 0
    when the signal is back to normal: S_t = max(0, S_{t-1} + z_t - slack).
    """
    mu = np.array([x[m, fit[m]].mean() for m in range(x.shape[0])])[:, None]
    sd = np.array([x[m, fit[m]].std() for m in range(x.shape[0])])[:, None]
    z = (x - mu) / sd
    if direction == "low":
        z = -z
    S = np.zeros_like(z)
    s = np.zeros(z.shape[0])
    for t in range(z.shape[1]):  # 8,761 steps, vectorised over machines
        s = np.maximum(0.0, s + z[:, t] - slack)
        S[:, t] = s
    return S


def run_at_least(cond: np.ndarray, k: int) -> np.ndarray:
    """True at t if cond held for the k consecutive hours ending at t."""
    if k <= 1:
        return cond
    r = rolling_sum(cond, k)
    return np.nan_to_num(r, nan=0) >= k


class FeatureBank:
    """Computes features once and caches them (many detectors share features)."""

    def __init__(self, panel: Panel):
        self.p = panel
        self._cache: dict = {}

    def get(self, source: str, transform: str, fit: np.ndarray | None = None,
            direction: str = "high") -> np.ndarray:
        # CUSUM depends on the normal-operation mask (it is standardised with
        # normal stats), so the mask's content is part of its cache key.
        key = (source, transform, direction if transform == "cusum" else None,
               hash(np.packbits(fit).tobytes()) if transform == "cusum" else None)
        if key in self._cache:
            return self._cache[key]
        p = self.p
        if source in SIGNALS:
            x = p.X[source]
            if transform == "raw":
                f = x.astype(float)
            elif transform.startswith("roll"):
                f = rolling_mean(x, int(transform[4:]))
            elif transform.startswith("std"):
                f = rolling_std(x, int(transform[3:]))
            elif transform == "cusum":
                f = cusum(x, fit, direction)
            else:
                raise ValueError(transform)
        elif source.startswith("err:"):
            # trailing count of error-log entries ("count24" = last 24 hours)
            e = source[4:]
            if e == "any":
                cnt = p.err.sum(axis=2)
            else:  # one code, or several joined by "|" (e.g. "error2|error3")
                cnt = sum(p.err[:, :, ERRORS.index(x)] for x in e.split("|"))
            f = rolling_sum(cnt, int(transform[5:]))
        elif source.startswith("age:"):
            # hours since last maintenance: a COUNTER, not a live reading
            c = source[4:]
            a = p.hours_since_maint
            f = np.nanmax(a, axis=2) if c == "max" else a[:, :, COMPONENTS.index(c)]
        else:
            raise ValueError(source)
        self._cache[key] = f
        return f


# ---------------------------------------------------------------------------------
# Detector specification
# ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Detector:
    concept: str                 # candidate idea it belongs to (C01, C02, ...)
    source: str = ""             # signal / error stream / counter
    transform: str = "raw"       # raw | rollN | stdN | cusum | countN
    direction: str = "high"      # high | low
    norm: str = "machine"        # global | model | machine  (threshold scope)
    pct: float = 99.0            # percentile of NORMAL operation
    k: int = 1                   # consecutive readings required
    combine: str | None = None   # None | "or" | "and"
    children: tuple = field(default_factory=tuple)

    @property
    def name(self) -> str:
        if self.combine:
            inner = f" {self.combine.upper()} ".join(c.name for c in self.children)
            return f"[{inner}]"
        return f"{self.source}|{self.transform}|{self.direction}|{self.norm}|p{self.pct:g}|k{self.k}"

    @property
    def variant(self) -> str:
        """Short label of the HOW (approach), independent of the signal."""
        c = self
        while c.combine:
            c = c.children[0]
        return f"{c.transform}·{c.norm}·p{c.pct:g}·k{c.k}"

    def rule_text(self) -> str:
        """Human-readable rule, e.g. for the write-up table."""
        if self.combine == "or" and len({(c.transform, c.norm, c.pct, c.k) for c in self.children}) == 1 \
                and not any(c.combine for c in self.children):
            # compact form for "any of these signals, same recipe"
            c0 = self.children[0]
            arrows = ", ".join(f"{c.source} {'↑' if c.direction == 'high' else '↓'}" for c in self.children)
            tr = c0.rule_text().split(" ", 1)[1].split(" > ")[0].split(" < ")[0]  # e.g. "24h trailing mean"
            tr = tr.replace("upward CUSUM", "CUSUM").replace("downward CUSUM", "CUSUM")
            kk = f", for {c0.k}+ consecutive hours" if c0.k > 1 else ""
            if c0.norm == "robust":
                from scipy.stats import norm as _N
                lim = f"median ± {abs(_N.ppf(c0.pct / 100)):.2f}×MAD-σ of its own machine's normal operation"
            elif c0.transform == "cusum":
                lim = f"P{c0.pct:g} of normal operation"
            else:
                scope = {"global": "all machines", "model": "its machine model", "machine": "its own machine"}[c0.norm]
                lim = f"P{c0.pct:g} (↑) / P{100 - c0.pct:g} (↓) of normal operation ({scope})"
            return f"any of [{arrows}]: {tr} beyond {lim}, in the arrow's direction{kk}"
        if self.combine:
            return f" {self.combine.upper()} ".join(f"({c.rule_text()})" for c in self.children)
        tr = {
            "raw": "hourly reading", "cusum": "CUSUM of standardised reading",
        }.get(self.transform)
        if tr is None:
            if self.transform.startswith("roll"):
                tr = f"{self.transform[4:]}h trailing mean"
            elif self.transform.startswith("std"):
                tr = f"{self.transform[3:]}h trailing std"
            elif self.transform.startswith("count"):
                tr = f"count in last {self.transform[5:]}h"
        if self.transform == "cusum":
            tr = f"{'downward' if self.direction == 'low' else 'upward'} CUSUM of standardised reading"
        src = (self.source.replace("err:any", "error-log").replace("err:", "").replace("|", "+")
               .replace("age:max", "hours since maint. of the oldest component")
               .replace("age:", "hours since maint. of "))
        high = self.direction == "high" or self.transform == "cusum"
        op = ">" if high else "<"
        q = self.pct if high else 100 - self.pct
        kk = f" for {self.k}+ consecutive hours" if self.k > 1 else ""
        if self.norm == "robust":
            from scipy.stats import norm as _N
            z = abs(_N.ppf(self.pct / 100))
            sign = "+" if high else "−"
            return (f"{src} {tr} {op} median {sign} {z:.2f}×MAD-σ of its own machine's "
                    f"normal operation{kk}")
        scope = {"global": "all machines", "model": "its machine model", "machine": "its own machine"}[self.norm]
        return f"{src} {tr} {op} P{q:g} of normal operation ({scope}){kk}"


def short_rule(det: Detector) -> str:
    """Compact rule text for "one (signal AND evidence) pair per part, joined by OR"
    detectors (C09, C10), which are unreadable written out in full. Falls back to
    the full rule for everything else."""
    pairs = det.children if det.combine == "or" else ((det,) if det.combine == "and" else ())
    ok = pairs and all(p.combine == "and" and len(p.children) == 2 and not any(c.combine for c in p.children)
                       for p in pairs)
    if not ok:
        return det.rule_text()
    sig = ", ".join(f"{p.children[0].source} {'↑' if p.children[0].direction == 'high' else '↓'}" for p in pairs)
    ev = " / ".join(p.children[1].source.replace("err:", "").replace("age:", "").replace("|", "+") for p in pairs)
    a, b = pairs[0].children
    drift = a.rule_text().split(" ", 1)[1]                 # e.g. "24h trailing mean > median + 3.72×MAD-σ ..."
    drift = drift.replace("> median +", "beyond median ±").replace("< median −", "beyond median ±")
    if len(pairs) > 1 and a.norm != "robust" and a.transform != "cusum":
        # parts drift in different directions (rotation falls, the others rise): name both percentiles
        hi, lo = f"P{a.pct:g}", f"P{100 - a.pct:g}"
        both = f"beyond {hi} (↑) / {lo} (↓)"
        drift = drift.replace(f"> {hi}", both).replace(f"< {lo}", both)
    if b.source.startswith("err:"):
        what = f"its own error code ({ev}) " + b.rule_text().split(" ", 1)[1]
    else:
        what = f"its own age since replacement ({ev}) " + b.rule_text().split("hourly reading ", 1)[-1]
    if len(pairs) > 1:
        return f"for each part [{sig}]: own signal {drift} AND {what}"
    return f"{a.source} {drift} AND {what}"


def thresholds(f: np.ndarray, fit: np.ndarray, norm: str, q: float,
               fit_machines: np.ndarray, models: np.ndarray) -> np.ndarray:
    """Per-machine threshold column (M,1) fitted on NORMAL hours only.

    global/model thresholds are fitted on DEV machines only, so holdout machines
    never shape the threshold. A per-machine threshold uses that machine's own
    normal hours (no event information from other machines).
    """
    M = f.shape[0]
    thr = np.full(M, np.nan)
    sel = np.zeros(M, dtype=bool)
    sel[fit_machines] = True
    if norm == "global":
        vals = f[sel[:, None] & fit]
        thr[:] = np.nanpercentile(vals, q)
    elif norm == "model":
        for mdl in np.unique(models):
            rows = sel & (models == mdl)
            thr[models == mdl] = np.nanpercentile(f[rows[:, None] & fit], q)
    elif norm == "machine":
        for m in range(M):
            v = f[m, fit[m]]
            thr[m] = np.nanpercentile(v, q) if np.isfinite(v).any() else np.nan
    elif norm == "robust":
        # Per machine: median + z_q * 1.4826 * MAD of normal hours, where z_q is
        # the standard-normal quantile of q (p99 -> 2.33, p99.99 -> 3.72).
        # Why: "normal" hours contain ~1,300 unlabeled look-alike excursions per
        # signal (same size and length as true pre-failure shifts, but no
        # failure follows). They fill the top ~2% of normal hours, so an
        # empirical p99 lands INSIDE that cluster and misses real precursors.
        # Median and MAD ignore a 2% contamination; the threshold then reflects
        # the noise of truly quiet operation.
        from scipy.stats import norm as _N
        z = _N.ppf(q / 100)
        for m in range(M):
            v = f[m, fit[m]]
            v = v[np.isfinite(v)]
            if len(v):
                med = np.median(v)
                thr[m] = med + z * 1.4826 * np.median(np.abs(v - med))
    else:
        raise ValueError(norm)
    return thr[:, None]


def fire(det: Detector, bank: FeatureBank, fit: np.ndarray, fit_machines: np.ndarray) -> np.ndarray:
    """Boolean (M,T) array: True where the detector is ON."""
    if det.combine:
        parts = [fire(c, bank, fit, fit_machines) for c in det.children]
        out = parts[0].copy()
        for p_ in parts[1:]:
            out = (out | p_) if det.combine == "or" else (out & p_)
        return out
    f = bank.get(det.source, det.transform, fit=fit, direction=det.direction)
    models = bank.p.machines.model.to_numpy()
    # CUSUM already folds the direction in (it accumulates downward shifts for
    # "low"), so its statistic is always compared from above.
    high = det.direction == "high" or det.transform == "cusum"
    q = det.pct if high else 100 - det.pct
    thr = thresholds(f, fit, det.norm, q, fit_machines, models)
    with np.errstate(invalid="ignore"):
        cond = (f > thr) if high else (f < thr)
    cond &= np.isfinite(f)
    return run_at_least(cond, det.k)


# ================================================================================================
# §5 · CANDIDATE PRECURSORS
#     the 19 ideas: story, first-draft rule, every variant the sweep tries
# ================================================================================================
# Step 2 of the brief: brainstormed candidate precursors.
#
# Each CONCEPT has a physical story (why it could come before a failure) and a
# default detector (the rule as first written down, before any tuning). The
# sweep then tries several ways of implementing the same idea - raw readings vs
# smoothed vs CUSUM, global vs per-model vs per-machine thresholds, different
# percentiles and run lengths - which is how we compare APPROACHES on equal
# footing.
#
# Where the ideas came from
#   * Domain intuition: classic rotating-equipment failure modes (bearing wear ->
#     vibration, friction -> speed loss, blockage -> pressure, electrical fault
#     -> supply voltage), maintenance-interval wear-out, error codes as early
#     software-side symptoms.
#   * An ML model (gradient boosting, see §8) trained on DEV machines only,
#     used as a brainstorming tool: its feature importances pointed at the
#     24h-mean features and the error counts, which we then turned back into
#     simple rules. It is also reported as a benchmark, not as a detector.
#   * Some deliberately "plausible but probably wrong" ideas (single signals, overdue
#     maintenance) so the rejection step has something to reject.


SIG_DIR = {"volt": "high", "rotate": "low", "pressure": "high", "vibration": "high"}


# threshold scopes tried for telemetry (see detectors.thresholds for "robust")
NORMS = ("global", "model", "machine", "robust")


TELEMETRY_VARIANTS = (("raw", 1), ("raw", 3), ("roll6", 1), ("roll24", 1), ("cusum", 1))


@dataclass(frozen=True)
class Concept:
    cid: str
    title: str
    story: str
    family: str             # telemetry | combo | errors | counter
    default: Detector       # the rule as first proposed (before tuning)
    variants: tuple         # all implementations tried in the sweep


def _ok(tr, nm):
    # CUSUM sits at exactly 0 most of the time in normal operation, so its
    # MAD is 0 and a robust threshold degenerates to "fire whenever S > 0".
    return not (tr == "cusum" and nm == "robust")


def _single(cid, src, direction, pcts):
    vs = tuple(Detector(cid, src, tr, direction, nm, p, k)
               for (tr, k), nm, p in product(TELEMETRY_VARIANTS, NORMS, pcts) if _ok(tr, nm))
    return vs


def build_concepts(pcts=(95.0, 99.0, 99.5, 99.9, 99.99), event: str = "any",
                   mapping: dict | None = None) -> list[Concept]:
    """mapping: {component: {"signal": s, "direction": d, "errors": [..]}}, learned
    on DEV machines (analysis.learn_mapping). Needed only for C09 and C10."""
    C: list[Concept] = []

    stories = {
        "volt": ("C01", "Voltage creeps up",
                 "An electrical part that is degrading (e.g. a winding or contactor) disturbs supply "
                 "regulation, so the measured voltage drifts above its usual band before it gives out."),
        "rotate": ("C02", "Rotation sags",
                   "Wear or a seizing part adds friction and drag, so the shaft turns slower than usual "
                   "for the same drive in the days before a mechanical failure."),
        "pressure": ("C03", "Pressure builds",
                     "A sticking valve, clogged passage or failing seal restricts flow, so upstream "
                     "pressure rises before the component fails."),
        "vibration": ("C04", "Vibration rises",
                      "Bearing wear, misalignment or a loosening mount raises vibration amplitude well "
                      "before a mechanical breakage."),
    }
    for s in SIGNALS:
        cid, title, story = stories[s]
        d = SIG_DIR[s]
        C.append(Concept(cid, title, story, "telemetry",
                         Detector(cid, s, "raw", d, "global", 99.0, 3),  # the brief's own example
                         _single(cid, s, d, pcts)))

    def and_or(cid, parts, op, trs, norms=NORMS):
        out = []
        for (tr, k), nm, p in product(trs, norms, pcts):
            if not _ok(tr, nm):
                continue
            kids = tuple(Detector(cid, s, tr, SIG_DIR[s], nm, p, k) for s in parts)
            out.append(Detector(cid, combine=op, children=kids))
        return tuple(out)

    or_vars = and_or("C05", SIGNALS, "or", trs=TELEMETRY_VARIANTS)
    C.append(Concept(
        "C05", "Any signal drifts",
        "Each component has its own physical signature (C01-C04). A machine-level alarm that "
        "watches all four at once should catch failures of any component.",
        "combo", [v for v in or_vars if v.variant == "roll24·machine·p99·k1"][0], or_vars))

    # ---- error log ------------------------------------------------------------------
    # Error counts are small integers, so a percentile threshold is coarse:
    # in normal operation P80 of the 24h/48h count is 0 ("fire on >= 1 error")
    # while P90+ is usually 1 ("fire on >= 2 errors"). Both are in the sweep;
    # without P80 only comp2 (which logs TWO codes per failure) could ever
    # pass a ">= 2 errors" rule - a trap found in the first trial run.
    err_pcts = (80.0, 90.0, 95.0, 99.0, 99.5)
    def err_vars(cid, src):
        return tuple(Detector(cid, src, tr, "high", nm, p, 1)
                     for tr, nm, p in product(("count24", "count48"), ("global", "machine"), err_pcts))
    C.append(Concept(
        "C06", "Any error code",
        "Controllers log faults (overcurrent, stall, sensor out of range) when a part starts to "
        "misbehave; a recent error is an early, software-side symptom.",
        "errors", Detector("C06", "err:any", "count24", "high", "global", 95.0, 1), err_vars("C06", "err:any")))

    # ---- counters -------------------------------------------------------------------
    cnt_vars = lambda cid, src: tuple(Detector(cid, src, "raw", "high", nm, p, 1)
                                      for nm, p in product(NORMS, pcts))
    # "any failure": the oldest part; a single component: that component's own age.
    age = "age:max" if event == "any" else f"age:{event}"
    C.append(Concept(
        "C07", "Overdue maintenance",
        "Parts wear out with use; the longer since a part was last replaced, the closer it is to "
        "failing. This is a COUNTER (it only climbs, then resets at maintenance).",
        "counter", Detector("C07", age, "raw", "high", "global", 99.0, 1), cnt_vars("C07", age)))

    # ---- confirmation combo -------------------------------------------------------------
    combo, default = [], None
    for tr, nm, p, ep in product(("roll6", "roll24"), ("machine", "robust"), pcts, (80.0, 90.0)):
        tel = Detector("C08", combine="or", children=tuple(
            Detector("C08", s, tr, SIG_DIR[s], nm, p, 1) for s in SIGNALS))
        err = Detector("C08", "err:any", "count48", "high", "global", ep, 1)
        combo.append(Detector("C08", combine="and", children=(tel, err)))
        if (tr, nm, p, ep) == ("roll24", "machine", 99.0, 80.0):
            default = combo[-1]
    C.append(Concept(
        "C08", "Drift confirmed by an error",
        "An error code on its own is noisy and a signal drift on its own may be a sensor blip; "
        "requiring both within two days should keep the alarms that matter.",
        "combo", default, tuple(combo)))

    # ---- C09: matched signature ----------------------------------------------------------
    # Each component has one signature signal and its own error code(s) (notebook 01).
    # C08 pairs ANY drift with ANY error. C09 makes the pairing component-specific and
    # explicit: a component's own signal drifting AND that
    # component's own error code(s). Which signal and which codes belong to which
    # component is learned from DEV machines only (analysis.learn_mapping).
    if mapping:
        comps = list(mapping) if event == "any" else [event]
        out, default = [], None
        for tr, nm, p, etr in product(("roll6", "roll24"), ("machine", "robust"), pcts, ("count24", "count48")):
            pairs = []
            for c in comps:
                mp = mapping[c]
                drift = Detector("C09", mp["signal"], tr, mp["direction"], nm, p, 1)
                err = Detector("C09", "err:" + "|".join(mp["errors"]), etr, "high", "global", 80.0, 1)
                pairs.append(Detector("C09", combine="and", children=(drift, err)))
            d = pairs[0] if len(pairs) == 1 else Detector("C09", combine="or", children=tuple(pairs))
            out.append(d)
            if (tr, nm, p, etr) == ("roll24", "machine", 99.0, "count48"):
                default = d
        C.append(Concept(
            "C09", "Matched signature (part's own signal + own error)",
            "A failing part shows up twice: its own physical signal drifts AND the controller logs the "
            "fault code tied to that part. A drift with an unrelated error code is probably a look-alike. "
            "The pairing is learned on dev machines only.",
            "combo", default, tuple(out)))

        # ---- C10: drift on an aging part ------------------------------------------------
        # Real pre-failure excursions start on parts with a median age
        # of 58-73 days vs ~30 for look-alikes. C10 turns that into a rule that can fire
        # BEFORE the error code: the part's own signal drifts AND the part is older than a
        # percentile of normal operation. Moderate percentiles only (P50-P90): the extreme
        # tail of a counter's "normal" distribution is what made C07 circular.
        out19, d19 = [], None
        for tr, nm, p, ap in product(("roll6", "roll24"), ("machine", "robust"), pcts, (50.0, 75.0, 90.0)):
            pairs = []
            for c in comps:
                mp = mapping[c]
                drift = Detector("C10", mp["signal"], tr, mp["direction"], nm, p, 1)
                aged = Detector("C10", f"age:{c}", "raw", "high", "global", ap, 1)
                pairs.append(Detector("C10", combine="and", children=(drift, aged)))
            d = pairs[0] if len(pairs) == 1 else Detector("C10", combine="or", children=tuple(pairs))
            out19.append(d)
            if (tr, nm, p, ap) == ("roll24", "machine", 99.0, 75.0):
                d19 = d
        C.append(Concept(
            "C10", "Drift on an aging part",
            "A part that has run long since its last replacement is worn; if its own signal starts to "
            "drift, that drift is more likely the start of a failure than a harmless excursion. Unlike "
            "C09 it does not wait for the error code, so it could warn earlier.",
            "combo", d19, tuple(out19)))
    return C
