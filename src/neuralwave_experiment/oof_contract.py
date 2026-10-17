"""Comparability contract shared by every pipeline in the team.

Two files make results comparable across branches without a package merge:

* a **split file** (``splits.parquet`` from ``neuralwave_data.prepare --folds 5
  --seed S``, or a CSV with the same two columns) that assigns every labelled
  customer to exactly one outer fold.  Each split file is one outer repeat.
* an **out-of-fold prediction file**: one semicolon CSV over all repeats with
  the columns ``customer_id;repeat;fold;prob;pv_installed;pv_installation_date;
  pv_capacity_kwp``, using the submission formatting rules (``TRUE``/``FALSE``,
  ``YYYY-MM-DD``, capacities with two decimals, empty date and capacity when
  ``FALSE``).  It never contains labels, features or regimes.

``python -m neuralwave_experiment.oof_contract --labels LABELS OOF [OOF ...]``
scores each file per repeat with the official formula and prints aggregates
only.  Any comparison used to choose a design is a Tier B decision: log it.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from neuralwave_experiment.scoring import score_predictions

SPLIT_COLUMNS = ("customer_id", "fold")
OOF_COLUMNS = ("customer_id", "repeat", "fold", "prob", "pv_installed", "pv_installation_date", "pv_capacity_kwp")


def _sniff_sep(path: Path) -> str:
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline()
    return ";" if ";" in header else ","


def _read_table(path: Path, string_columns: Sequence[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep=_sniff_sep(path), dtype={c: str for c in string_columns},
                            encoding="utf-8-sig", keep_default_na=False)
    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()
    return frame


def load_split_file(path: str | Path, labelled_ids: Iterable[str]) -> dict[str, int]:
    """Map every labelled customer to its fold; extra customers are ignored.

    Raises ``ValueError`` when the file lacks the two columns, assigns a
    labelled customer to several folds, or leaves any labelled customer
    without a fold.  Messages carry counts, never identifiers.
    """
    path = Path(path)
    frame = _read_table(path, ["customer_id"])
    missing_columns = [c for c in SPLIT_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"split file {path} lacks columns: {missing_columns}")
    labelled = {str(c).strip() for c in labelled_ids}
    frame = frame[frame["customer_id"].isin(labelled)]
    if frame["customer_id"].duplicated().any():
        raise ValueError(f"split file {path} assigns a labelled customer to several folds")
    folds = pd.to_numeric(frame["fold"], errors="coerce")
    if folds.isna().any() or (folds % 1 != 0).any():
        raise ValueError(f"split file {path} has non-integer folds")
    missing = labelled - set(frame["customer_id"])
    if missing:
        raise ValueError(f"split file {path} does not cover labelled customers: {len(missing)} missing")
    return {cid: int(fold) for cid, fold in zip(frame["customer_id"], folds)}


def load_split_files(paths: Sequence[str | Path], labelled_ids: Iterable[str]) -> list[dict[str, int]]:
    """One fold map per file; the position in the list is the repeat index."""
    ids = list(labelled_ids)
    return [load_split_file(p, ids) for p in paths]


@dataclass(frozen=True)
class OOFRow:
    customer_id: str
    repeat: int
    fold: int
    prob: float
    pv_installed: bool
    pv_installation_date: date | None = None
    pv_capacity_kwp: float | None = None


def format_oof_row(row: OOFRow) -> list[str]:
    installed = bool(row.pv_installed)
    day = ""
    kwp = ""
    if installed:
        if row.pv_installation_date is not None:
            day = pd.Timestamp(row.pv_installation_date).strftime("%Y-%m-%d")
        if row.pv_capacity_kwp is not None and math.isfinite(float(row.pv_capacity_kwp)):
            kwp = f"{float(row.pv_capacity_kwp):.2f}"
    return [str(row.customer_id), str(int(row.repeat)), str(int(row.fold)), f"{float(row.prob):.6f}",
            "TRUE" if installed else "FALSE", day, kwp]


def write_oof_predictions(rows: Iterable[OOFRow], path: str | Path) -> Path:
    """Write the shared out-of-fold file; rows are sorted by repeat, fold, customer."""
    path = Path(path)
    ordered = sorted(rows, key=lambda r: (int(r.repeat), int(r.fold), str(r.customer_id)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(OOF_COLUMNS)
        for row in ordered:
            writer.writerow(format_oof_row(row))
    return path


def read_oof_predictions(path: str | Path) -> pd.DataFrame:
    """Read a shared out-of-fold file back into typed columns."""
    path = Path(path)
    frame = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in OOF_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"out-of-fold file {path} lacks columns: {missing}")
    flags = frame["pv_installed"].str.strip().str.upper()
    bad = ~flags.isin(["TRUE", "FALSE"])
    if bad.any():
        raise ValueError(f"out-of-fold file {path} has {int(bad.sum())} rows with pv_installed outside TRUE/FALSE")
    out = pd.DataFrame({
        "customer_id": frame["customer_id"].str.strip(),
        "repeat": pd.to_numeric(frame["repeat"]).astype(int),
        "fold": pd.to_numeric(frame["fold"]).astype(int),
        "prob": pd.to_numeric(frame["prob"], errors="coerce"),
        "pv_installed": flags.eq("TRUE"),
        "predicted_installation_date": pd.to_datetime(frame["pv_installation_date"].replace("", pd.NA),
                                                      format="%Y-%m-%d", errors="coerce"),
        "predicted_capacity_kwp": pd.to_numeric(frame["pv_capacity_kwp"].replace("", pd.NA), errors="coerce"),
    })
    return out


def load_labels(path: str | Path) -> pd.DataFrame:
    """Labels from parquet or a delimited file; ``pv_installed`` becomes bool."""
    frame = _read_table(Path(path), ["customer_id"])
    required = {"customer_id", "pv_installed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"labels lack columns: {sorted(missing)}")
    flags = frame["pv_installed"]
    if flags.dtype != bool:
        text = flags.astype(str).str.strip().str.upper()
        unknown = ~text.isin(["TRUE", "FALSE", "1", "0"])
        if unknown.any():
            raise ValueError(f"labels have {int(unknown.sum())} rows with pv_installed outside TRUE/FALSE/1/0")
        frame["pv_installed"] = text.isin(["TRUE", "1"])
    for column in ("pv_installation_date", "pv_capacity_kwp"):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def score_oof_file(path: str | Path, labels: pd.DataFrame) -> dict[str, object]:
    """Official score per repeat of one out-of-fold file, plus mean and sd.

    Each repeat must cover every labelled customer exactly once.
    """
    predictions = read_oof_predictions(path)
    labelled = labels.copy()
    labelled["customer_id"] = labelled["customer_id"].astype(str).str.strip()
    expected = set(labelled["customer_id"])
    per_repeat: list[dict[str, float | int]] = []
    for repeat, block in predictions.groupby("repeat", sort=True):
        if set(block["customer_id"]) != expected or block["customer_id"].duplicated().any():
            raise ValueError(f"out-of-fold file {path} repeat {int(repeat)} does not cover the labelled customers exactly once")
        metrics = score_predictions(labelled, block.drop(columns=["repeat", "fold", "prob"]))
        per_repeat.append({"repeat": int(repeat), **metrics})
    if not per_repeat:
        raise ValueError(f"out-of-fold file {path} is empty")
    scores = np.array([r["main_score"] for r in per_repeat], dtype=float)
    return {
        "file": str(path), "repeats": len(per_repeat), "per_repeat": per_repeat,
        "score_mean": float(scores.mean()), "score_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "f1_mean": float(np.mean([r["f1"] for r in per_repeat])),
        "D_mean": float(np.mean([r["D"] for r in per_repeat])),
        "P_mean": float(np.mean([r["P"] for r in per_repeat])),
    }


def summary_line(result: dict[str, object]) -> str:
    return (f"file={Path(str(result['file'])).name} repeats={result['repeats']} "
            f"score={result['score_mean']:.2f}±{result['score_sd']:.2f} "
            f"f1={result['f1_mean']:.3f} D={result['D_mean']:.3f} P={result['P_mean']:.3f}")


def paired_line(a: dict[str, object], b: dict[str, object]) -> str | None:
    """Paired comparison when both files share the same repeat indices."""
    ra = {r["repeat"]: r["main_score"] for r in a["per_repeat"]}  # type: ignore[index]
    rb = {r["repeat"]: r["main_score"] for r in b["per_repeat"]}  # type: ignore[index]
    shared = sorted(set(ra) & set(rb))
    if not shared:
        return None
    diffs = [ra[k] - rb[k] for k in shared]
    better = sum(d > 0 for d in diffs)
    return (f"paired: {Path(str(a['file'])).name} vs {Path(str(b['file'])).name} "
            f"mean_diff={float(np.mean(diffs)):.2f} repeats_A_better={better}/{len(shared)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score shared out-of-fold prediction files; aggregates only.")
    parser.add_argument("--labels", required=True, help="labels parquet or delimited file")
    parser.add_argument("oof", nargs="+", help="out-of-fold prediction files")
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    results = [score_oof_file(p, labels) for p in args.oof]
    for result in results:
        print(summary_line(result))
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            line = paired_line(results[i], results[j])
            if line:
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
