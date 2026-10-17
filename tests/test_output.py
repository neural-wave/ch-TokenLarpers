from __future__ import annotations

import numpy as np
import pandas as pd

from pvdetect.output import write_extensions, write_predictions


def test_prediction_file_exact_text(tmp_path):
    pred = pd.DataFrame(
        {
            "customer_id": ["000123", "000124"],
            "pv_installed": [True, False],
            "pv_installation_date": [pd.Timestamp("2023-04-10"), pd.Timestamp("2023-01-01")],
            "pv_capacity_kwp": [8.0, 3.0],
        }
    )
    write_predictions(pred, tmp_path)
    text = (tmp_path / "predictions.csv").read_bytes().decode("utf-8")
    assert text == (
        "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
        "000123;TRUE;2023-04-10;8.00\n"
        "000124;FALSE;;\n"
    )


def test_empty_extension_file_is_header_only(tmp_path):
    write_extensions(pd.DataFrame(columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]), tmp_path)
    assert (tmp_path / "pv_extensions_predictions.csv").read_text() == (
        "customer_id;extension_date;pv_capacity_after_extension_kwp\n"
    )


def test_extension_rows_formatted(tmp_path):
    ext = pd.DataFrame(
        {"customer_id": ["007"], "extension_date": [pd.Timestamp("2024-06-01")], "pv_capacity_after_extension_kwp": [np.float64(12.345)]}
    )
    write_extensions(ext, tmp_path)
    assert (tmp_path / "pv_extensions_predictions.csv").read_text().splitlines()[1] == "007;2024-06-01;12.35"
