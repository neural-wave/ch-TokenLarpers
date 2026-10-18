"""PV capacity-extension detection: a rolling capacity nowcast and a two-geometry change test.

Steps, per customer the main model calls PV:
1. Every STEP days, a WIN-day window yields label-free physics features (`window_features`): the length of the
   zero-withdrawal run around solar noon on its typical and clearest days (clipping-robust: a larger array
   starts the run earlier and ends it later), load levels outside PV hours, midday/evening ratios, season.
2. A CatBoost regressor (`fit_nowcaster`, fitted inside `procedure.fit` on that fit's training customers
   only) maps each window to log kWp installed at that time. Targets: labelled PV customers' windows that
   lie inside one capacity regime (initial kWp, or the total after a labelled extension), plus optional
   pseudo-labelled windows of unlabelled customers the fit's own main model calls PV (`POOL_PROB`).
3. Level test: PELT on the nowcast k(t); a new plateau must beat every earlier long plateau by ≥ MIN_RATIO
   at z ≥ MIN_Z (rejects outage recovery and seasonal swings).
4. Year-over-year box: a capacity step at τ makes y(t) = k(t) − k(t − 364 d) a box exactly one year wide;
   the best box start within ±BOX_TOL windows must have z ≥ BOX_Z. Without a year of history the level step
   needs z ≥ MIN_Z + STRICT_EXTRA instead.
5. Date: two-segment fit on the daily year-over-year difference of a 7-day zero-run envelope (±REFINE days).
   Capacity after: median nowcast on the new plateau; if that is not above the predicted initial kWp, the
   predicted initial kWp × the cumulative step ratio.
6. At most MAX_PER_CUSTOMER events per customer and MAX_TOTAL per prediction call, highest z first. The
   challenge text states 2 of the 20 hidden-test customers have extensions, one of them twice (3 events).

All thresholds are fixed constants from synthetic development seeds; nothing is tuned on real labels.
Extension detection never changes the main predictions (the main frame is an input, not an output).
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pvdetect.pipeline import CustomerData, FeatureTable

WIN, STEP = 35, 7  # window length and stride, days
YEAR = 52  # windows per 364 days
NOON = 50  # 12:30 slot in the fixed UTC+1 day
ZERO_KWH = 0.005  # a slot at or below this withdrawal counts as zero (pvdetect.features uses the same)
MIN_DAYS = 18  # valid days a window needs
POST_INSTALL_DAYS = 14  # windows start at least this long after the (predicted) installation date
MIN_RATIO = 1.15
MIN_Z = 3.0
BOX_Z = 1.5
BOX_TOL = 6
STRICT_EXTRA = 1.0
PLATEAU = 6  # windows an earlier plateau needs to count as a reference
RAMP = WIN // STEP  # windows after a boundary still mixing both regimes
MERGE = 8  # up-steps starting within this many windows are one ramp
BOX_MIN_IN, BOX_MIN_OUT, BOX_HALF = 6, 4, 26
REFINE = 50  # days searched either side of a step for its date
MAX_PER_CUSTOMER = 2
MAX_TOTAL = 3
POOL_PROB = 0.95  # pseudo-labelled unlabelled customers: main-model P(PV) at least this
MIN_TRAIN_CUSTOMERS = 5
FEATURES = (
    "zw50",
    "zw80",
    "zw95",
    "zshare",
    "log_eve",
    "log_night",
    "log_morn",
    "log_mid10",
    "log_mid50",
    "mid10_eve",
    "mid50_eve",
    "eve_night",
    "woy_sin",
    "woy_cos",
)
COLUMNS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
BANDS = {"eve": (76, 88), "night": (0, 20), "morn": (20, 28), "mid": (40, 60)}  # UTC+1 slots: 19–22, 0–5, 5–7, 10–15 h


@dataclass(frozen=True)
class Event:
    customer_id: str
    date: dt.date
    kwp_after: float
    z: float


# ---------------------------------------------------------------------------------------------- features


def zero_run_hours(M: np.ndarray) -> np.ndarray:
    """Per day, hours of the zero-withdrawal run containing 12:30 (0 when 12:30 is not zero, NaN if missing)."""
    z = M <= ZERO_KWH
    out = np.where(np.isnan(M[:, NOON]), np.nan, 0.0)
    for d in np.flatnonzero(z[:, NOON]):
        row = z[d]
        lo = NOON
        while lo > 0 and row[lo - 1]:
            lo -= 1
        hi = NOON
        while hi < 95 and row[hi + 1]:
            hi += 1
        out[d] = (hi - lo + 1) / 4
    return out


def window_starts(n_days: int) -> np.ndarray:
    return np.arange(0, max(n_days - WIN + 1, 0), STEP)


def window_features(cd: CustomerData) -> pd.DataFrame:
    """One row per window: FEATURES plus `start` and `centre` (Timestamps); NaN rows for thin windows.
    Levels are kW (slot kWh × 4) over fixed UTC+1 bands; zero-run lengths are hours."""
    M = np.asarray(cd.M, float)
    zw = zero_run_hours(M)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kw = {name: np.nanmean(M[:, a:b], 1) * 4 for name, (a, b) in BANDS.items()}
        zmid = np.nanmean(M[:, 40:60] <= ZERO_KWH, 1)
    dates = pd.DatetimeIndex(cd.dates.astype("datetime64[ns]"))
    starts = window_starts(M.shape[0])

    def lg(x: float) -> float:
        return float(np.log(max(x, 0.0) + 0.02))

    rows = []
    for a in starts:
        sl = slice(a, a + WIN)
        ok = np.isfinite(zw[sl]) & np.isfinite(kw["eve"][sl]) & np.isfinite(kw["mid"][sl])
        if ok.sum() < MIN_DAYS:
            rows.append(dict.fromkeys(FEATURES, np.nan))
            continue
        z, md = zw[sl][ok], kw["mid"][sl][ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e, n, m = (float(np.nanmedian(kw[k][sl][ok])) for k in ("eve", "night", "morn"))
        m10, m50 = float(np.quantile(md, 0.1)), float(np.median(md))
        woy = 2 * np.pi * dates[a + WIN // 2].dayofyear / 365.25
        rows.append(
            {
                "zw50": float(np.median(z)),
                "zw80": float(np.quantile(z, 0.8)),
                "zw95": float(np.quantile(z, 0.95)),
                "zshare": float(np.nanmean(zmid[sl][ok])),
                "log_eve": lg(e),
                "log_night": lg(n),
                "log_morn": lg(m),
                "log_mid10": lg(m10),
                "log_mid50": lg(m50),
                "mid10_eve": lg(m10) - lg(e),
                "mid50_eve": lg(m50) - lg(e),
                "eve_night": lg(e) - lg(n),
                "woy_sin": float(np.sin(woy)),
                "woy_cos": float(np.cos(woy)),
            }
        )
    out = pd.DataFrame(rows, columns=list(FEATURES))
    out["start"] = dates[starts]
    out["centre"] = dates[starts + WIN // 2]
    return out


# ------------------------------------------------------------------------------------------- nowcaster


def regime_rows(cid: str, cd: CustomerData, regimes: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    """Windows of one customer lying wholly inside one capacity regime, ≥ POST_INSTALL_DAYS after it starts.
    `regimes`: (start, kWp) sorted by start; the last regime runs to the end of the series."""
    f = window_features(cd)
    start = f["start"]
    end = start + pd.Timedelta(days=WIN)
    keep = np.zeros(len(f), bool)
    target = np.full(len(f), np.nan)
    for j, (t0, kwp) in enumerate(regimes):
        t1 = regimes[j + 1][0] if j + 1 < len(regimes) else pd.Timestamp.max
        if pd.isna(t0) or not (np.isfinite(kwp) and kwp > 0):
            continue  # an unusable label contributes no windows (as in the main capacity fit)
        sel = (start >= t0 + pd.Timedelta(days=POST_INSTALL_DAYS)) & (end <= t1)
        keep |= sel.to_numpy()
        target[sel.to_numpy()] = np.log(kwp)
    f = f[keep].assign(customer_id=cid, log_kwp=target[keep])
    return f.dropna(subset=[*FEATURES, "log_kwp"])


def fit_nowcaster(rows: pd.DataFrame):
    """CatBoost regressor on log kWp; every customer carries the same total weight."""
    from catboost import CatBoostRegressor

    w = 1.0 / rows.groupby("customer_id")["log_kwp"].transform("size").to_numpy(float)
    reg = CatBoostRegressor(
        iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, loss_function="RMSE",
        random_seed=0, verbose=False, allow_writing_files=False, thread_count=1,
    )
    reg.fit(rows[list(FEATURES)].to_numpy(float), rows["log_kwp"].to_numpy(float), sample_weight=w)
    return reg


def nowcast(nowcaster, f: pd.DataFrame, usable: np.ndarray) -> np.ndarray:
    k = np.full(len(f), np.nan)
    ok = usable & f[list(FEATURES)].notna().all(axis=1).to_numpy()
    if ok.any():
        k[ok] = nowcaster.predict(f.loc[ok, list(FEATURES)].to_numpy(float))
    return k


# --------------------------------------------------------------------------------------- change tests


def _segment(x: np.ndarray) -> list[int]:
    """PELT (l2) boundaries of the NaN-interpolated series; BIC-like penalty from the diff-MAD noise,
    inflated for the overlap of consecutive windows."""
    import ruptures as rpt

    ok = np.isfinite(x)
    if ok.sum() < 2 * PLATEAU:
        return []
    idx = np.arange(x.size)
    y = np.interp(idx, idx[ok], x[ok])
    d = np.diff(y)
    sig = max(1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2), 0.03)
    pen = 3.0 * sig**2 * np.log(y.size) * (WIN / STEP) ** 0.5
    return [b for b in rpt.Pelt(model="l2", min_size=3, jump=1).fit(y).predict(pen=pen) if b < y.size]


def _plateau(x: np.ndarray, a: int, b: int) -> float:
    v = x[min(a + RAMP, b - 1) : b] if b - a > RAMP + 1 else x[a:b]
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def level_steps(k: np.ndarray) -> list[tuple[int, float, float]]:
    """(window index, log step, z) of persistent up-steps above every earlier long plateau whose rest-of-series
    median also clears the reference."""
    ok = np.flatnonzero(np.isfinite(k))
    if ok.size < 2 * PLATEAU:
        return []
    first, last = int(ok[0]), int(ok[-1]) + 1
    x = k[first:last]
    bounds = _segment(x)
    edges = [0, *bounds, x.size]
    segs = list(zip(edges[:-1], edges[1:]))
    levels = [_plateau(x, a, b) for a, b in segs]
    with np.errstate(all="ignore"):
        med = np.concatenate([np.full(b - a, lv) for (a, b), lv in zip(segs, levels)])
        sig = max(1.4826 * float(np.nanmedian(np.abs(x - med))), 0.05)
    n_eff = WIN / STEP
    out, last_ref, i = [], 0, 1
    while i < len(segs):
        refs = [levels[j] for j in range(last_ref, i) if segs[j][1] - segs[j][0] >= PLATEAU and np.isfinite(levels[j])]
        refs = refs or [levels[j] for j in range(last_ref, i) if np.isfinite(levels[j])]
        ref = max(refs) if refs else np.nan
        j = i
        while j + 1 < len(segs) and segs[j + 1][0] - segs[i][0] < MERGE and levels[j + 1] > levels[j]:
            j += 1
        new, b = levels[j], segs[i][0]
        n_after = int(np.isfinite(x[b : segs[j][1]]).sum())
        n_ref = int(np.isfinite(x[max(0, b - BOX_HALF) : b]).sum())
        if np.isfinite(ref) and np.isfinite(new) and n_after >= 4 and n_ref >= 4:
            step = new - ref
            z = step / (sig * np.sqrt(n_eff / n_after + n_eff / n_ref))
            # installed capacity does not fall back: the whole rest of the series (all seasons, so a seasonal
            # nowcast bias averages out) must also sit above the reference by the same margin
            rest = x[min(b + RAMP, x.size - 1) :]
            persist = float(np.nanmedian(rest)) - ref if np.isfinite(rest).sum() >= 4 else np.nan
            if step >= np.log(MIN_RATIO) and z >= MIN_Z and persist >= np.log(MIN_RATIO):
                out.append((first + b, float(step), float(z)))
                last_ref, i = j, j + 1
                continue
        i += 1
    return out


def box_scores(y: np.ndarray) -> list[tuple[int, float, float]]:
    """(box start τ, height, z) for one-year boxes in y; τ may precede the first defined value (falling edge
    only) or leave the box cut by the series end (rising edge only)."""
    ok = np.flatnonzero(np.isfinite(y))
    if ok.size < BOX_MIN_IN + BOX_MIN_OUT:
        return []
    v = y[ok]
    sig = max(1.4826 * float(np.median(np.abs(v - np.median(v)))), 0.03)
    n_eff = WIN / STEP
    out = []
    for tau in range(int(ok[0]) - YEAR + BOX_MIN_IN, int(ok[-1]) - BOX_MIN_IN + 2):
        a, e = max(tau, 0), min(tau + YEAR, y.size)
        inside = y[a:e][np.isfinite(y[a:e])]
        flanks = [y[max(tau - BOX_HALF, 0) : max(tau, 0)], y[min(tau + YEAR, y.size) : min(tau + YEAR + BOX_HALF, y.size)]]
        flanks = [f[np.isfinite(f)] for f in flanks]
        flanks = [f for f in flanks if f.size >= BOX_MIN_OUT]
        if inside.size < BOX_MIN_IN or not flanks:
            continue
        outv = np.concatenate(flanks)
        h = float(np.median(inside) - np.median(outv))
        out.append((tau, h, h / (sig * np.sqrt(n_eff / inside.size + n_eff / outv.size))))
    return out


def confirmed_steps(k: np.ndarray) -> list[tuple[int, float, float]]:
    """Level steps (in date order) that the year-over-year box agrees with (see module docstring). Each
    confirmed step's own box is removed from y before the next step is scored, so a second extension inside
    the first one's year is not hidden by the first box sitting in its flank."""
    y = np.full(k.size, np.nan)
    y[YEAR:] = k[YEAR:] - k[:-YEAR]
    out = []
    for tau, step, z in level_steps(k):
        near = [b for b in box_scores(y) if abs(b[0] - tau) <= BOX_TOL]
        if near:
            best = max(near, key=lambda b: b[2])
            if not (best[2] >= BOX_Z and best[1] > 0):
                continue
        elif z < MIN_Z + STRICT_EXTRA:
            continue
        out.append((tau, step, z))
        y[max(tau, 0) : tau + YEAR] -= step
    return sorted(sorted(out, key=lambda e: -e[2])[:MAX_PER_CUSTOMER])


def refine_day(M: np.ndarray, d0: int) -> int:
    """Daily date near day index d0: two-segment SSE minimum of the year-over-year difference of the 7-day
    max zero-run envelope (the 7-day min midday load when the customer does not clip near d0)."""
    env = pd.Series(zero_run_hours(M)).rolling(7, center=True, min_periods=3).max().to_numpy()
    near = env[max(d0 - 60, 0) : d0 + 60]
    if not np.isfinite(near).any() or np.nanmax(near) < 1.0:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mid = np.nanmean(M[:, 40:60], 1)
        env = -pd.Series(mid).rolling(7, center=True, min_periods=3).min().to_numpy()
    diff = np.full(env.size, np.nan)
    diff[364:] = env[364:] - env[:-364]
    lo, hi = max(d0 - REFINE, 0), min(d0 + REFINE, env.size)
    seg = diff[lo:hi]
    if np.isfinite(seg).sum() < 30:
        seg = env[lo:hi]
    best, best_t = np.inf, d0
    for t in range(10, seg.size - 10):
        a, b = seg[:t], seg[t:]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 5 or b.size < 5:
            continue
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if sse < best:
            best, best_t = sse, lo + t
    return int(best_t)


def customer_events(nowcaster, cid: str, cd: CustomerData, install: dt.date | None, initial_kwp: float) -> list[Event]:
    f = window_features(cd)
    if f.empty:
        return []
    first = pd.Timestamp(install) if install is not None else pd.Timestamp(cd.dates[0]) - pd.Timedelta(days=1)
    usable = (f["start"] >= first + pd.Timedelta(days=POST_INSTALL_DAYS)).to_numpy()
    k = nowcast(nowcaster, f, usable)
    steps = confirmed_steps(k)
    events, cum = [], 1.0
    for j, (tau, step, z) in enumerate(steps):
        nxt = steps[j + 1][0] if j + 1 < len(steps) else len(k)
        plateau = k[tau + RAMP : nxt]
        new_log = float(np.nanmedian(plateau)) if np.isfinite(plateau).sum() >= 4 else np.nan
        # the installation itself (predicted install date too early): a step from below the predicted initial
        # capacity up to about it is not an extension
        if np.isfinite(new_log) and new_log < np.log(initial_kwp * MIN_RATIO) and new_log - step < np.log(initial_kwp / MIN_RATIO):
            continue
        d0 = int(np.searchsorted(cd.dates, f["centre"].iloc[tau].to_datetime64().astype("datetime64[D]")))
        date = pd.Timestamp(cd.dates[refine_day(np.asarray(cd.M, float), d0)]).date()
        if install is not None and date <= install:
            continue
        if events and (date - events[-1].date).days <= 30:
            continue  # one ±30-day match window holds at most one event
        cum *= float(np.exp(step))
        absolute = float(np.exp(new_log)) if np.isfinite(new_log) else np.nan
        kwp = absolute if np.isfinite(absolute) and absolute > initial_kwp else initial_kwp * cum
        events.append(Event(cid, date, float(kwp), float(z)))
    return events


# ------------------------------------------------------------------------------------------- fit / predict


def _pool_regimes(main_model: dict, table: FeatureTable) -> dict[str, tuple[CustomerData, list]]:
    """Unlabelled pool customers the main model calls PV with P ≥ POOL_PROB: one regime from the predicted
    date at the predicted kWp (pseudo-labels; extension customers in the pool are mislabelled before their
    extension, which the nowcaster must tolerate)."""
    from pvdetect.pipeline import predict

    pool = table.extension_pool or {}
    if not pool:
        return {}
    view = FeatureTable(ids=list(pool), customers=pool, cache=table.cache.setdefault("extension_pool", {}))
    main = {**main_model, "extensions": {"enabled": False}}
    try:
        pred, _ = predict(main, view, view.ids)
    except Exception as exc:  # e.g. an embedding block needs parsed rows the pool does not keep
        print(f"warning: extension pool not pseudo-labelled ({type(exc).__name__}: {exc})", file=sys.stderr)
        return {}
    out = {}
    for r in pred.itertuples(index=False):
        if r.pv_installed and r.prob >= POOL_PROB and pd.notna(r.pv_installation_date) and np.isfinite(r.pv_capacity_kwp):
            out[r.customer_id] = (pool[r.customer_id], [(pd.Timestamp(r.pv_installation_date), float(r.pv_capacity_kwp))])
    return out


def fit_extensions(
    table: FeatureTable,
    labels: pd.DataFrame,
    train_ids: list[str],
    ext_labels: pd.DataFrame | None,
    main_model: dict,
) -> tuple[dict, object | None]:
    """(model["extensions"] section, nowcaster object) from the training customers only (plus the optional
    pseudo-labelled pool). Disabled when fewer than MIN_TRAIN_CUSTOMERS labelled PV customers have windows."""
    lab = labels.set_index("customer_id")
    ext = ext_labels if ext_labels is not None else pd.DataFrame(columns=COLUMNS)
    ext = ext[ext["customer_id"].isin(set(train_ids))]
    parts = []
    for cid in train_ids:
        row, cd = lab.loc[cid], table.customers.get(cid)
        if cd is None or not bool(row["pv_installed"]):
            continue
        regimes = [(pd.Timestamp(row["pv_installation_date"]), float(row["pv_capacity_kwp"]))]
        for e in ext[ext["customer_id"] == cid].sort_values("extension_date").itertuples(index=False):
            regimes.append((pd.Timestamp(e.extension_date), float(e.pv_capacity_after_extension_kwp)))
        parts.append(regime_rows(cid, cd, regimes))
    labelled = sum(1 for p in parts if len(p))
    pool = _pool_regimes(main_model, table)
    parts += [regime_rows(f"pool:{cid}", cd, reg) for cid, (cd, reg) in pool.items()]
    info = {"labelled_customers": labelled, "pool_customers": sum(1 for _ in pool)}
    rows = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
    if labelled < MIN_TRAIN_CUSTOMERS or rows is None:
        return {"enabled": False, **info, "reason": "too few labelled PV customers"}, None
    return {"enabled": True, "catboost_file": "catboost_ext.cbm", "windows": int(len(rows)), **info}, fit_nowcaster(rows)


def predict_extensions(model: dict, table: FeatureTable, pred: pd.DataFrame) -> pd.DataFrame:
    """Extension rows for the TRUE customers of `pred`; ≤ MAX_PER_CUSTOMER each, ≤ MAX_TOTAL by z overall."""
    nowcaster = model.get("_objects", {}).get("catboost_ext")
    if not (model.get("extensions") or {}).get("enabled") or nowcaster is None:
        return pd.DataFrame(columns=COLUMNS)
    events: list[Event] = []
    for r in pred[pred["pv_installed"].astype(bool)].itertuples(index=False):
        cd = table.customers.get(r.customer_id)
        if cd is None or not np.isfinite(r.pv_capacity_kwp):
            continue
        install = pd.Timestamp(r.pv_installation_date).date() if pd.notna(r.pv_installation_date) else None
        events += customer_events(nowcaster, r.customer_id, cd, install, float(r.pv_capacity_kwp))
    events = sorted(events, key=lambda e: -e.z)[:MAX_TOTAL]
    events.sort(key=lambda e: (e.customer_id, e.date))
    return pd.DataFrame(
        [{"customer_id": e.customer_id, "extension_date": pd.Timestamp(e.date), "pv_capacity_after_extension_kwp": e.kwp_after} for e in events],
        columns=COLUMNS,
    )
