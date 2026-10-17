from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect import diagnose
from pvdetect.scoring import main_score


def _frame(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"])
    df["pv_installation_date"] = pd.to_datetime(df["pv_installation_date"])
    return df


@pytest.fixture
def pair():
    truth = _frame(
        [
            ("001", True, "2023-06-01", 10.0),
            ("002", True, "2023-12-01", 200.0),
            ("003", True, "2024-03-01", 5.0),
            ("004", False, None, np.nan),
            ("005", False, None, np.nan),
        ]
    )
    pred = _frame(
        [
            ("001", True, "2023-06-16", 5.0),  # 15 days late, half the capacity
            ("002", True, "2023-12-01", 500.0),  # exact date, at the ceiling
            ("003", False, None, np.nan),  # missed
            ("004", True, "2024-01-01", 7.0),  # false alarm
            ("005", False, None, np.nan),
        ]
    )
    return pred, truth


def test_counts_rates_and_score_match_the_official_scorer(pair):
    pred, truth = pair
    s = diagnose.summarize(diagnose.error_table(pred, truth))
    assert s["counts"] == {"TP": 2, "FP": 1, "FN": 1, "TN": 1}
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    official = main_score(pred, truth)
    assert s["score"] == pytest.approx(official["score"])
    assert s["D"] == pytest.approx(official["D"]) and s["P"] == pytest.approx(official["P"])
    assert sum(s["lost"].values()) == pytest.approx(100 - s["score"])


def test_date_and_capacity_breakdowns(pair):
    pred, truth = pair
    s = diagnose.summarize(diagnose.error_table(pred, truth))
    assert s["dates"]["median_signed"] == pytest.approx(7.5)
    assert s["dates"]["within"][15] == 1.0 and s["dates"]["within"][7] == 0.5
    assert s["capacity"]["at_ceiling"] == 1
    assert s["capacity"]["median_ratio"] == pytest.approx((0.5 + 2.5) / 2)
    assert s["by_size"]["10-30"] == {"n": 1, "missed": 0, "D": 0.5, "P": 0.0, "ratio": 0.5}
    assert s["by_size"]["<10"]["missed"] == 1 and np.isnan(s["by_size"]["<10"]["D"])


def test_missing_prediction_counts_as_not_pv(pair):
    pred, truth = pair
    s = diagnose.summarize(diagnose.error_table(pred[pred["customer_id"] != "001"], truth))
    assert s["counts"]["FN"] == 2 and s["counts"]["TP"] == 1


def test_oof_file_selects_one_repeat(tmp_path):
    rows = [
        "customer_id;repeat;fold;prob;pv_installed;pv_installation_date;pv_capacity_kwp",
        "001;0;0;0.9;TRUE;2023-06-01;10.00",
        "001;1;2;0.2;FALSE;;",
    ]
    (tmp_path / "oof.csv").write_text("\n".join(rows) + "\n")
    p0, n = diagnose.load_predictions(tmp_path / "oof.csv", 0)
    p1, _ = diagnose.load_predictions(tmp_path / "oof.csv", 1)
    assert n == 2 and p0["pv_installed"].tolist() == [True] and p1["pv_installed"].tolist() == [False]
    with pytest.raises(ValueError, match="repeat 5"):
        diagnose.load_predictions(tmp_path / "oof.csv", 5)


def test_out_dir_inside_repository_is_refused(tmp_path, pair):
    pred, truth = pair
    for name, df in (("p.csv", pred), ("t.csv", truth)):
        out = df.assign(
            pv_installed=df["pv_installed"].map({True: "TRUE", False: "FALSE"}),
            pv_installation_date=df["pv_installation_date"].dt.strftime("%Y-%m-%d"),
        )
        out.to_csv(tmp_path / name, sep=";", index=False)
    with pytest.raises(SystemExit, match="refusing"):
        diagnose.main([str(tmp_path / "p.csv"), str(tmp_path / "t.csv"), "--out-dir", str(diagnose.REPO / "x")])
    assert not (diagnose.REPO / "x").exists()
    assert diagnose.main([str(tmp_path / "p.csv"), str(tmp_path / "t.csv"), "--out-dir", str(tmp_path / "priv")]) == 0
    assert (tmp_path / "priv" / "errors.csv").exists()
