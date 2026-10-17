"""Error analysis of predictions against labels, for the data holder only.

    python -m pvdetect.diagnose PRED TRUTH [--repeat 0] [--out-dir PRIVATE_DIR]

PRED is a submission-format `predictions.csv` or the shared out-of-fold file from
`train.py --cv --oof-out` (preferred: in-sample predictions of training customers are optimistic).
Prints aggregate lines only. `--out-dir` (outside the repository) receives the per-customer table.
The output is AIL-derived when run on real data: keep it with the team, never in an agent session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.io import _read_str_table, _to_date, _to_float, load_labels, parse_bool
from pvdetect.selection import ape_array

REPO = Path(__file__).resolve().parents[1]
DIAGNOSIS_FILE = "diagnosis.json"
DIAGNOSIS_SCHEMA = 1
SIZE_EDGES = (0.0, 10.0, 30.0, 100.0, 300.0, np.inf)
SIZE_LABELS = ("<10", "10-30", "30-100", "100-300", ">=300")
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
CAP_CEILING = 500.0  # pipeline.to_kwp upper clip


def load_predictions(path: str | Path, repeat: int | None = 0) -> tuple[pd.DataFrame, int]:
    """Submission-format or OOF predictions; returns (frame, number of repeats in the file)."""
    df, _ = _read_str_table(path)
    n_rep = 1
    if "repeat" in df.columns:
        reps = sorted(df["repeat"].astype(int).unique())
        n_rep = len(reps)
        if repeat not in reps:
            raise ValueError(f"repeat {repeat} not in file (repeats 0..{n_rep - 1})")
        df = df[df["repeat"].astype(int) == repeat]
    out = pd.DataFrame({"customer_id": df["customer_id"]})
    out["pv_installed"] = parse_bool(df["pv_installed"])
    out["pv_installation_date"] = _to_date(df["pv_installation_date"])
    out["pv_capacity_kwp"] = _to_float(df["pv_capacity_kwp"])
    if out["customer_id"].duplicated().any():
        raise ValueError("duplicate customer rows in predictions")
    return out.reset_index(drop=True), n_rep


def error_table(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled customer: outcome and the date / capacity errors that drive the score."""
    t = truth.set_index("customer_id")
    p = pred.set_index("customer_id").reindex(t.index)
    t_pv = t["pv_installed"].to_numpy(bool)
    p_pv = p["pv_installed"].fillna(False).to_numpy(bool)
    outcome = np.select([t_pv & p_pv, ~t_pv & p_pv, t_pv & ~p_pv], ["TP", "FP", "FN"], "TN")
    err = (p["pv_installation_date"] - t["pv_installation_date"]).dt.days.to_numpy(float)
    pk = p["pv_capacity_kwp"].to_numpy(float)
    tk = t["pv_capacity_kwp"].to_numpy(float)
    tp = outcome == "TP"
    err = np.where(tp, err, np.nan)
    ape = np.where(tp, ape_array(pk, tk), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tp & (tk > 0), pk / tk, np.nan)
    size = pd.cut(pd.Series(tk, index=t.index), SIZE_EDGES, labels=SIZE_LABELS, right=False).astype(str)
    dates = t["pv_installation_date"]
    return pd.DataFrame(
        {
            "outcome": outcome,
            "date_err_days": err,
            "D_i": np.where(tp, np.where(np.isfinite(err), np.maximum(0, 1 - np.abs(err) / 30), 0.0), np.nan),
            "capacity_ape": ape,
            "capacity_ratio": ratio,
            "P_i": np.where(tp, np.where(np.isfinite(ape), np.maximum(0, 1 - ape / 0.5), 0.0), np.nan),
            "true_kwp": tk,
            "pred_kwp": np.where(p_pv, pk, np.nan),
            "size_bin": np.where(t_pv, size.to_numpy(), ""),
            "install_year": np.where(t_pv, dates.dt.year.astype("Int64").astype(str), ""),
            "install_season": np.where(t_pv, dates.dt.month.map(SEASONS).fillna("").to_numpy(), ""),
        },
        index=t.index,
    )


def summarize(tab: pd.DataFrame) -> dict:
    n = {k: int((tab["outcome"] == k).sum()) for k in ("TP", "FP", "FN", "TN")}
    tp, fp, fn = n["TP"], n["FP"], n["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    t = tab[tab["outcome"] == "TP"]
    D = float(t["D_i"].mean()) if tp else 0.0
    P = float(t["P_i"].mean()) if tp else 0.0
    lost = {"detection": 100 * (1 - f1), "dates": 15 * f1 * (1 - D), "capacity": 15 * f1 * (1 - P)}
    e = t["date_err_days"].dropna()
    r = t["capacity_ratio"].dropna()
    a = t["capacity_ape"].dropna()
    return {
        "counts": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "D": D,
        "P": P,
        "score": f1 * (70 + 15 * D + 15 * P),
        "lost": lost,
        "largest": max(lost, key=lost.get),
        "dates": {
            "n": int(e.size),
            "missing": int(t["date_err_days"].isna().sum()),
            "median_abs": float(e.abs().median()) if e.size else np.nan,
            "median_signed": float(e.median()) if e.size else np.nan,
            "late_share": float((e > 0).mean()) if e.size else np.nan,
            "within": {k: float((e.abs() <= k).mean()) if e.size else np.nan for k in (0, 7, 15, 30)},
        },
        "capacity": {
            "n": int(a.size),
            "median_ape": float(a.median()) if a.size else np.nan,
            "median_ratio": float(r.median()) if r.size else np.nan,
            "under_share": float((r < 1).mean()) if r.size else np.nan,
            "within": {k: float((a <= k / 100).mean()) if a.size else np.nan for k in (10, 25, 50)},
            "at_ceiling": int((t["pred_kwp"] >= CAP_CEILING - 1e-6).sum()),
        },
        "by_size": _group(tab, "size_bin", SIZE_LABELS),
        "by_season": _group(tab, "install_season", ("winter", "spring", "summer", "autumn")),
        "by_year": _group(tab, "install_year", sorted(v for v in tab["install_year"].unique() if v)),
    }


def _group(tab: pd.DataFrame, col: str, keys) -> dict:
    out = {}
    for k in keys:
        g = tab[(tab[col] == k) & tab["outcome"].isin(["TP", "FN"])]
        if g.empty:
            continue
        tp = g[g["outcome"] == "TP"]
        out[k] = {
            "n": len(g),
            "missed": int((g["outcome"] == "FN").sum()),
            "D": float(tp["D_i"].mean()) if len(tp) else np.nan,
            "P": float(tp["P_i"].mean()) if len(tp) else np.nan,
            "ratio": float(tp["capacity_ratio"].median()) if len(tp) else np.nan,
        }
    return out


def failure_modes(s: dict) -> list[str]:
    """Rule-based reading of the aggregates: the main failure mode inside each component."""
    modes = []
    c = s["counts"]
    if c["FN"] or c["FP"]:
        modes.append(
            f"detection: {'missed PV (recall)' if c['FN'] > c['FP'] else 'false alarms (precision)' if c['FP'] > c['FN'] else 'missed PV and false alarms equally'}"
        )
    d = s["dates"]
    if d["n"]:
        med = d["median_signed"]
        if abs(med) > 7 and max(d["late_share"], 1 - d["late_share"]) >= 0.65:
            modes.append(f"dates: systematic {'late' if med > 0 else 'early'} bias (median {med:+.0f} d)")
        else:
            modes.append("dates: spread, no consistent bias (onset detection, not a fixed offset)")
    k = s["capacity"]
    if k["n"]:
        ratio = k["median_ratio"]
        kind = "underestimation" if ratio < 0.85 else "overestimation" if ratio > 1.15 else "spread around the truth"
        worst = min(
            ((b, v) for b, v in s["by_size"].items() if np.isfinite(v["P"])), key=lambda bv: bv[1]["P"], default=None
        )
        tail = f"; weakest size bin {worst[0]} kWp (P={worst[1]['P']:.2f})" if worst else ""
        ceil = f"; {k['at_ceiling']} prediction(s) at the {CAP_CEILING:.0f} kWp ceiling" if k["at_ceiling"] else ""
        modes.append(f"capacity: {kind} (median pred/true {ratio:.2f}){tail}{ceil}")
    return modes


def _fmt(x: float | None, nd: int = 3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def report_lines(s: dict) -> list[str]:
    f = _fmt
    c, d, k = s["counts"], s["dates"], s["capacity"]
    lines = [
        f"score: {s['score']:.2f} f1={f(s['f1'])} D={f(s['D'])} P={f(s['P'])}",
        "points_lost: " + " ".join(f"{n}={v:.2f}" for n, v in s["lost"].items()) + f" largest={s['largest']}",
        f"detection: precision={f(s['precision'])} recall={f(s['recall'])} TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}",
        f"dates (TP): n={d['n']} missing={d['missing']} median_abs_days={f(d['median_abs'], 1)} "
        f"median_signed_days={f(d['median_signed'], 1)} late_share={f(d['late_share'], 2)} "
        + " ".join(f"within_{w}d={f(v, 2)}" for w, v in d["within"].items()),
        f"capacity (TP): n={k['n']} median_ape={f(k['median_ape'], 2)} median_ratio={f(k['median_ratio'], 2)} "
        f"under_share={f(k['under_share'], 2)} "
        + " ".join(f"within_{w}pct={f(v, 2)}" for w, v in k["within"].items())
        + f" at_{CAP_CEILING:.0f}kwp_ceiling={k['at_ceiling']}",
    ]
    for name in ("by_size", "by_season", "by_year"):
        if s[name]:
            lines.append(
                f"{name}: "
                + " | ".join(
                    f"{b} n={v['n']} missed={v['missed']} D={f(v['D'], 2)} P={f(v['P'], 2)} ratio={f(v['ratio'], 2)}"
                    for b, v in s[name].items()
                )
            )
    lines += [f"failure_mode: {m}" for m in failure_modes(s)]
    lines.append(f"next_step: {s['largest']}")
    return lines


def _outside_repo(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p == REPO or REPO in p.parents:
        raise SystemExit("refusing: --out-dir is inside the repository; choose a private folder outside it")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("truth")
    ap.add_argument("--repeat", type=int, default=0, help="repeat to analyse in an OOF file (default 0)")
    ap.add_argument("--out-dir", type=Path, help="private folder for the per-customer error table")
    args = ap.parse_args(argv)
    out_dir = _outside_repo(args.out_dir) if args.out_dir else None
    pred, n_rep = load_predictions(args.pred, args.repeat)
    truth = load_labels(args.truth)
    missing = int((~truth["customer_id"].isin(pred["customer_id"])).sum())
    tab = error_table(pred, truth)
    source = f"out-of-fold, repeat {args.repeat} of {n_rep}" if n_rep > 1 else "predictions file"
    if n_rep > 1:
        print(f"source: out-of-fold file, repeat {args.repeat} of {n_rep}")
    else:
        print("source: predictions file (if the model was trained on these customers, every number is optimistic)")
    if missing:
        print(f"warning: {missing} labelled customers have no prediction (counted as not PV)")
    summary = summarize(tab)
    for line in report_lines(summary):
        print(line)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tab.to_csv(out_dir / "errors.csv", sep=";", index_label="customer_id", float_format="%.4f")
        doc = {
            "schema": DIAGNOSIS_SCHEMA,
            "source": source,
            "out_of_fold": n_rep > 1,
            "missing_predictions": missing,
            "failure_modes": failure_modes(summary),
            **summary,
        }
        (out_dir / DIAGNOSIS_FILE).write_text(json.dumps(_json_safe(doc), indent=2), encoding="utf-8")
        print(f"per-customer table (errors.csv) and {DIAGNOSIS_FILE} written to the private --out-dir")
    return 0


def _json_safe(x):
    """Aggregates only; NaN/inf become null so the GUI can read the file as strict JSON."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    raise SystemExit(main())
