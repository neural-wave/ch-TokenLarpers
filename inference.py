"""Offline inference: python inference.py path/to/test.csv -> predictions.csv + pv_extensions_predictions.csv in cwd."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pvdetect.io import load_timeseries
from pvdetect.model import load_model
from pvdetect.output import write_extensions, write_predictions
from pvdetect.pipeline import FeatureTable, build_feature_table, predict

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.json"


@dataclass(frozen=True)
class InferenceRun:
    """What one inference call computed: predictions, extensions, the label-free table and the loader report."""

    pred: pd.DataFrame
    ext: pd.DataFrame
    table: FeatureTable
    report: dict
    customer_ids: list[str]


def run_inference(
    test_csv: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    *,
    on_step: Callable[[str], None] | None = None,
) -> InferenceRun:
    """Load the model and input, predict, write both submission files into `out_dir`.

    `on_step`, when given, is called with "model", "data", "predict" and "write" before each step starts.
    """
    step = on_step or (lambda _name: None)
    step("model")
    model = load_model(model_path)
    step("data")
    loaded = load_timeseries(test_csv)
    step("predict")
    table = build_feature_table(loaded)
    pred, ext = predict(model, table, loaded.customer_ids)
    step("write")
    write_predictions(pred, Path(out_dir))
    write_extensions(ext, Path(out_dir))
    return InferenceRun(pred=pred, ext=ext, table=table, report=loaded.report, customer_ids=list(loaded.customer_ids))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test_csv", help="path to the test time-series CSV file")
    args = ap.parse_args(argv)
    run_inference(args.test_csv, MODEL_PATH, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
