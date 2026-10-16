"""Offline inference: python inference.py path/to/test.csv -> predictions.csv + pv_extensions_predictions.csv in cwd."""

from __future__ import annotations

import argparse
from pathlib import Path

from pvdetect.io import load_timeseries
from pvdetect.model import load_model
from pvdetect.output import write_extensions, write_predictions
from pvdetect.pipeline import build_feature_table, predict

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test_csv", help="path to the test time-series CSV file")
    args = ap.parse_args(argv)
    model = load_model(MODEL_PATH)
    loaded = load_timeseries(args.test_csv)
    table = build_feature_table(loaded)
    pred, ext = predict(model, table, loaded.customer_ids)
    write_predictions(pred, Path.cwd())
    write_extensions(ext, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
