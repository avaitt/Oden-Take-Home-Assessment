"""
data.py · What the data is and how the brief's definitions apply to it.

    §1 Settings: every knob (H, W, event, guardrails, split) and why each default was chosen
    §2 Data: load the five CSVs, check them, merge same-time failures into one event, dev/holdout split
    §3 Event windows and normal operation: the brief's definitions as yes/no flags per machine-hour

Part of a 3-file set (data -> detectors -> evaluation). Notebooks import from these
files after adding ../src to the path; each notebook's import cell lists exactly what it uses.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================================================
# §1 · SETTINGS
#     every tunable knob (H, W, event, guardrails, split, sweep percentiles)
# ================================================================================================
# Every tunable knob of the analysis lives here, so a reviewer can change one
# value and re-run the whole pipeline.
#
# The assignment asks that H, W and the target event be configurable. They are
# fields of `Config`; each notebook sets them in one line, e.g.
#
#     cfg = Config(H=24, W=24, event="comp2")


# Events the pipeline knows how to target.
#   "any"   -> any component failure; failures on the same machine at the same
#              timestamp are merged into ONE event (as the brief requires).
#   "compN" -> only events in which component N failed.
EVENT_KINDS = ("any", "comp1", "comp2", "comp3", "comp4")


@dataclass(frozen=True)
class Config:
    # ---- the two parameters the brief asks for -------------------------------
    # H: lead time. The scoring window ends H hours before the event.
    # W: window length. The window covers lags [H, H+W) hours before the event.
    #
    # Why H=0, W=48 ("will it fail within the next 48 hours?"; scored lags 1..47, the failure hour
    # itself is never scored). Chosen by F1 on the dev-only H x W grid (notebook 04, section 6):
    #  * The framing "will it fail within the next N hours?" (H=0), with the smallest W at
    #    which F1 reaches its plateau: W=48 (F1 0.945 vs 0.935 for a one-day window). The window then
    #    covers the whole precursor, which starts ~48 h before a failure.
    #  * No leakage concern: an hour-by-hour audit (notebook 01, section 3) shows each part's signal
    #    stays shifted up to its 06:00 failure stamp and is back to normal one hour after it.
    #  * In practice we would deploy with H=12, W=36, so every alert that counts leaves half a day
    #    to act. With C09's threshold re-tuned for that window it scores F1 0.94 on holdout (0.92 at
    #    H=0, W=48); left unchanged it drops to 0.81 (notebook 06, section 11).
    H: int = 0
    W: int = 48

    # Which event we look for precursors of. See EVENT_KINDS.
    event: str = "any"

    # ---- guardrails -------------------------------------------------------------
    # Hours after a failure or a maintenance action that are NOT "normal"
    # operation (machine just restarted / new part bedding in). The data shows
    # no visible post-repair transient, so 24h is a cheap, conservative choice.
    recovery_h: int = 24

    # Hours at the start of the record where rolling features are not yet
    # "warm" (a 24h rolling mean needs 24 readings). Excluded everywhere.
    warmup_h: int = 24

    # Leakage switch, ONLY flipped by the audit to show what goes wrong: if
    # True (and H=0) the event hour itself is scored, where the repair record
    # and counter resets already "know" about the failure.
    score_event_hour: bool = False

    # ---- honest-statistics settings ---------------------------------------------
    # Machines are split once into DEV (used to choose anything) and HOLDOUT
    # (used only to report). Stratified by machine model.
    dev_fraction: float = 0.5
    seed: int = 7
    n_bootstrap: int = 1000      # cluster bootstrap, resampling MACHINES
    n_null_shifts: int = 500     # circular-shift null draws per finalist

    # Percentiles of normal operation used as thresholds in the sweep.
    # 99.99 was added after the robust-threshold experiment showed that the
    # best robust cut-offs sit far out (z ~ 3.7); it is part of the sweep, so
    # the dev/holdout protocol still decides.
    percentiles: tuple = (95.0, 99.0, 99.5, 99.9, 99.99)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    @property
    def tag(self) -> str:
        return f"H{self.H}_W{self.W}"


# The H x W grid used to choose the window (notebook 04, section 6; dev machines only). It spans
# windows ending at the failure up to windows starting 3 days ahead, so it also answers "is there anything
# days ahead?" without a separate run.
H_GRID = (0, 3, 6, 9, 12, 18, 24, 36, 48, 72)
W_GRID = (6, 12, 18, 24, 30, 36, 48, 60, 72, 96)


# ================================================================================================
# §2 · DATA
#     load the five CSVs, check them, reshape into (machine x hour) arrays, dev/holdout split
# ================================================================================================
# Loading the Microsoft Azure Predictive Maintenance dataset into dense arrays.
#
# Source: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance
# Download: kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
# (no Kaggle account needed; files are cached under ~/.cache/kagglehub).
#
# Design choice: the telemetry is a perfect hourly grid (100 machines x 8,761
# hours, no gaps, verified below with asserts). So instead of long pandas
# frames we hold every signal as a (machines, hours) numpy array. All detectors
# and metrics then become vectorised array operations, which is what makes it
# cheap to test hundreds of detector variants and thousands of null draws.


DATASET = "arnabbiswas1/microsoft-azure-predictive-maintenance"


SIGNALS = ("volt", "rotate", "pressure", "vibration")


COMPONENTS = ("comp1", "comp2", "comp3", "comp4")


ERRORS = ("error1", "error2", "error3", "error4", "error5")


def dataset_dir() -> Path:
    """Download (or reuse the cached copy of) the dataset and return its folder."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET))


@dataclass
class Panel:
    """Everything the analysis needs, aligned on one hourly time axis."""

    times: pd.DatetimeIndex            # length T, hourly
    machine_ids: np.ndarray            # length M (1..100)
    X: dict                            # signal -> (M, T) float array
    err: np.ndarray                    # (M, T, 5) error counts per hour
    maint: np.ndarray                  # (M, T, 4) bool, maintenance on comp k at hour t
    hours_since_maint: np.ndarray      # (M, T, 4) hours since last maint of comp k (counter!)
    events: pd.DataFrame               # one row per (machine, timestamp) failure event
    failures_raw: pd.DataFrame         # raw failure rows (761)
    maint_raw: pd.DataFrame
    errors_raw: pd.DataFrame
    machines: pd.DataFrame             # machineID, model, age

    @property
    def M(self) -> int:
        return len(self.machine_ids)

    @property
    def T(self) -> int:
        return len(self.times)

    def hour_index(self, ts) -> np.ndarray:
        """Convert timestamps to integer hour offsets on the panel's time axis."""
        return ((pd.to_datetime(ts) - self.times[0]) / pd.Timedelta("1h")).astype(int).to_numpy()


def load_panel(path: Path | None = None) -> Panel:
    path = Path(path) if path else dataset_dir()
    tel = pd.read_csv(path / "PdM_telemetry.csv", parse_dates=["datetime"])
    err = pd.read_csv(path / "PdM_errors.csv", parse_dates=["datetime"])
    mnt = pd.read_csv(path / "PdM_maint.csv", parse_dates=["datetime"])
    fail = pd.read_csv(path / "PdM_failures.csv", parse_dates=["datetime"])
    mach = pd.read_csv(path / "PdM_machines.csv")

    # --- telemetry -> dense arrays ------------------------------------------------
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    mids = np.sort(tel.machineID.unique())
    times = pd.DatetimeIndex(np.sort(tel.datetime.unique()))
    M, T = len(mids), len(times)
    # Guardrail: confirm the "perfect grid" assumption instead of trusting it.
    assert len(tel) == M * T, "telemetry is not a complete machine x hour grid"
    assert (np.diff(times.values) == np.timedelta64(1, "h")).all(), "time axis has gaps"
    assert not tel[list(SIGNALS)].isna().any().any(), "telemetry has missing values"
    X = {s: tel[s].to_numpy(float).reshape(M, T) for s in SIGNALS}

    def idx(ts):
        return ((ts - times[0]) / pd.Timedelta("1h")).astype(int).to_numpy()

    mpos = {m: i for i, m in enumerate(mids)}

    # --- error log -> hourly counts per error type ---------------------------------
    E = np.zeros((M, T, len(ERRORS)), dtype=np.int16)
    e_in = err[(err.datetime >= times[0]) & (err.datetime <= times[-1])]
    np.add.at(
        E,
        (e_in.machineID.map(mpos).to_numpy(), idx(e_in.datetime),
         e_in.errorID.map({e: k for k, e in enumerate(ERRORS)}).to_numpy()),
        1,
    )

    # --- maintenance -> per-component flags + "hours since maintenance" ---------------
    # Maintenance starts in mid-2014 (before telemetry) so each component has a
    # known age on day one. Note: a failure is ALWAYS accompanied by a
    # maintenance record at the same hour (the repair), which makes this
    # counter reset exactly at the event -> a leakage trap we guard against by
    # never letting the event hour itself into a scoring window.
    MT = np.zeros((M, T, 4), dtype=bool)
    hsm = np.full((M, T, 4), np.nan)
    tgrid = np.arange(T)
    for (m, comp), g in mnt.groupby(["machineID", "comp"]):
        k = COMPONENTS.index(comp)
        h = np.sort(((g.datetime - times[0]) / pd.Timedelta("1h")).astype(int).to_numpy())
        inside = h[(h >= 0) & (h < T)]
        MT[mpos[m], inside, k] = True
        # last maintenance hour at or before each t (searchsorted on sorted list)
        j = np.searchsorted(h, tgrid, side="right") - 1
        ok = j >= 0
        hsm[mpos[m], ok, k] = tgrid[ok] - h[j[ok]]

    # --- failures -> events (merge same machine + same timestamp) ---------------------
    ev = (
        fail.groupby(["machineID", "datetime"])["failure"]
        .apply(lambda s: tuple(sorted(s)))
        .reset_index(name="comps")
        .sort_values(["machineID", "datetime"])
        .reset_index(drop=True)
    )
    ev["m"] = ev.machineID.map(mpos)
    ev["t"] = idx(ev.datetime)
    ev["n_comps"] = ev.comps.map(len)
    ev["hour_of_day"] = ev.datetime.dt.hour

    return Panel(
        times=times, machine_ids=mids, X=X, err=E, maint=MT, hours_since_maint=hsm,
        events=ev, failures_raw=fail, maint_raw=mnt, errors_raw=err,
        machines=mach.set_index("machineID").loc[mids].reset_index(),
    )


def target_events(panel: Panel, event: str) -> pd.DataFrame:
    """Events of the requested kind ("any" or "compN")."""
    ev = panel.events
    if event == "any":
        return ev.copy()
    return ev[ev.comps.map(lambda c: event in c)].copy()


def dev_holdout_split(panel: Panel, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split machine positions into DEV / HOLDOUT, stratified by machine model.

    Splitting by machine (not by time or by event) means no machine contributes
    to both choosing a detector and grading it, and events that cluster on one
    machine cannot leak across the split.
    """
    rng = np.random.default_rng(seed)
    dev = []
    for _, g in panel.machines.reset_index().groupby("model"):
        pos = g["index"].to_numpy().copy()  # pandas 3 hands out read-only views
        rng.shuffle(pos)
        dev.extend(pos[: int(round(len(pos) * frac))])
    dev = np.sort(np.array(dev))
    hold = np.setdiff1d(np.arange(panel.M), dev)
    return dev, hold


# ================================================================================================
# §3 · EVENT WINDOWS AND NORMAL OPERATION
#     the brief's definitions as yes/no flags per machine-hour
# ================================================================================================
# Turning the definitions in the brief into boolean (machine, hour) masks.
#
#     Event window   hour t is in the window of event e if  H' <= t_e - t < H + W,
#                    with H' = max(H, 1). The event hour itself (lag 0) is NEVER
#                    scored: at that hour the logs already contain the failure
#                    (repair record, counter reset), which would be leakage.
#
#     Normal         not within H+W hours before ANY failure (any component, even
#                    when the target is one component - a comp2 pre-failure
#                    period is not "normal" just because we target comp1),
#                    not within `recovery_h` after a failure or maintenance,
#                    and not in the warm-up hours. (The machines never stop:
#                    rotation never drops to 0, so "running" is every hour.)
#
#     Evaluable      an event counts towards coverage only if its whole window
#     event          lies after warm-up. Events in the first ~2 days of data
#                    would otherwise be scored on partial windows.
#
#     Alert onsets   an alert is an off->on switch. Onsets in recovery periods are
#                    suppressed (an operator would mute alarms right after a
#                    repair), and onsets in the last H+W hours are dropped because
#                    we cannot see whether an event followed (right-censoring).


@dataclass
class Masks:
    cfg: Config
    in_w: np.ndarray        # (M,T) inside the W window of some evaluable target event
    normal: np.ndarray      # (M,T) normal operation
    recov: np.ndarray       # (M,T) recovery period (or warm-up) -> never scored
    onset_ok: np.ndarray    # (M,T) hours where an alert onset can be judged
    next_lag: np.ndarray    # (M,T) hours until the next target event (big if none)
    ev: pd.DataFrame        # target events with window bounds + `evaluable`

    @property
    def horizon(self) -> int:
        return self.cfg.H + self.cfg.W


def _mark(mask: np.ndarray, m: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Set mask[m, start..end] = True (inclusive) for many intervals at once.

    Uses a difference array so it is O(M*T) regardless of the number of intervals.
    """
    M, T = mask.shape
    d = np.zeros((M, T + 1), dtype=np.int32)
    s = np.clip(start, 0, T)
    e = np.clip(end + 1, 0, T)
    ok = e > s
    np.add.at(d, (m[ok], s[ok]), 1)
    np.add.at(d, (m[ok], e[ok]), -1)
    mask |= np.cumsum(d[:, :T], axis=1) > 0


def build_masks(panel: Panel, cfg: Config, time_range: tuple[int, int] | None = None) -> Masks:
    M, T = panel.M, panel.T
    H, W = cfg.H, cfg.W
    Hp = H if cfg.score_event_hour else max(H, 1)

    # --- recovery: [t_x, t_x + R] after every failure or maintenance -------------------
    recov = np.zeros((M, T), dtype=bool)
    all_ev = panel.events
    _mark(recov, all_ev.m.to_numpy(), all_ev.t.to_numpy(), all_ev.t.to_numpy() + cfg.recovery_h)
    mm, tt = np.nonzero(panel.maint.any(axis=2))
    _mark(recov, mm, tt, tt + cfg.recovery_h)
    recov[:, : cfg.warmup_h] = True  # warm-up treated the same way: never scored

    # --- normal: also exclude H+W before ANY failure ---------------------------------
    pre_any = np.zeros((M, T), dtype=bool)
    _mark(pre_any, all_ev.m.to_numpy(), all_ev.t.to_numpy() - H - W + 1, all_ev.t.to_numpy())
    normal = ~pre_any & ~recov

    # --- target events and their windows ------------------------------------------
    ev = target_events(panel, cfg.event)
    if cfg.score_event_hour:
        # leakage demo only: the event hour is normally both "recovery" and
        # outside every window; here we deliberately let it be scored
        recov[ev.m.to_numpy(), ev.t.to_numpy()] = False
    ev["ws"] = ev.t - H - W + 1          # window start (inclusive)
    ev["we"] = ev.t - Hp                 # window end (inclusive)
    ev["evaluable"] = ev.ws >= cfg.warmup_h
    # a window that is entirely recovery (e.g. a 2nd failure 3h after the 1st)
    # contains nothing we are allowed to score
    rc = np.concatenate([np.zeros((M, 1), int), np.cumsum(~recov, axis=1)], axis=1)
    m_, ws_, we_ = ev.m.to_numpy(), np.clip(ev.ws.to_numpy(), 0, T - 1), np.clip(ev.we.to_numpy(), 0, T - 1)
    ev["scorable_hours"] = rc[m_, we_ + 1] - rc[m_, ws_]
    ev["evaluable"] &= ev.scorable_hours > 0
    if time_range is not None:  # temporal audit: only events fully inside the slice
        ev["evaluable"] &= (ev.ws >= time_range[0]) & (ev.t < time_range[1])

    in_w = np.zeros((M, T), dtype=bool)
    e2 = ev[ev.evaluable]
    _mark(in_w, e2.m.to_numpy(), e2.ws.to_numpy(), e2.we.to_numpy())
    in_w &= ~recov

    # --- hours to the next target event (for precision) ---------------------------
    next_lag = np.full((M, T), 10**9, dtype=np.int64)
    tgrid = np.arange(T)
    for m, g in ev.groupby("m"):
        te = np.sort(g.t.to_numpy())
        j = np.searchsorted(te, tgrid, side="right")  # first event strictly after t
        ok = j < len(te)
        next_lag[m, ok] = te[j[ok]] - tgrid[ok]

    onset_ok = ~recov.copy()
    onset_ok[:, T - (H + W):] = False  # right-censoring guardrail

    # Optional restriction to a time slice (used by the temporal audit).
    if time_range is not None:
        a, b = time_range
        tm = np.zeros(T, dtype=bool)
        tm[a:b] = True
        in_w &= tm
        normal &= tm
        onset_ok &= tm

    return Masks(cfg=cfg, in_w=in_w, normal=normal, recov=recov, onset_ok=onset_ok,
                 next_lag=next_lag, ev=ev.reset_index(drop=True))
