"""Shared split-file and out-of-fold prediction contract; fabricated inputs only."""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from neuralwave_experiment.oof_contract import (
    OOF_COLUMNS, OOFRow, load_labels, load_split_file, load_split_files, main, read_oof_predictions,
    score_oof_file, write_oof_predictions,
)
from neuralwave_experiment.scoring import score_predictions

IDS = ["A01", "A02", "B03", "B04"]


def _labels() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": IDS, "pv_installed": [True, True, False, False],
        "pv_installation_date": [date(2024, 3, 1), date(2023, 7, 15), None, None],
        "pv_capacity_kwp": [10.0, 5.0, None, None],
    })


def test_split_file_parquet_and_csv_give_the_same_folds(tmp_path: Path):
    frame = pd.DataFrame({"customer_id": IDS + ["EXTRA"], "fold": [0, 1, 0, 1, 0]})
    frame.to_parquet(tmp_path / "s.parquet", index=False)
    (tmp_path / "s.csv").write_text("customer_id;fold\n" + "".join(f"{c};{f}\n" for c, f in zip(frame.customer_id, frame.fold)))
    (tmp_path / "s_comma.csv").write_text("customer_id,fold\n" + "".join(f"{c},{f}\n" for c, f in zip(frame.customer_id, frame.fold)))
    expected = {"A01": 0, "A02": 1, "B03": 0, "B04": 1}
    assert load_split_file(tmp_path / "s.parquet", IDS) == expected
    assert load_split_file(tmp_path / "s.csv", IDS) == expected
    assert load_split_file(tmp_path / "s_comma.csv", IDS) == expected


def test_split_file_missing_a_labelled_customer_raises_with_a_count_only(tmp_path: Path):
    pd.DataFrame({"customer_id": IDS[:-1], "fold": [0, 1, 0]}).to_parquet(tmp_path / "s.parquet", index=False)
    with pytest.raises(ValueError, match=r"does not cover labelled customers: 1 missing$") as info:
        load_split_file(tmp_path / "s.parquet", IDS)
    assert "B04" not in str(info.value)


def test_split_file_with_duplicate_or_bad_folds_raises(tmp_path: Path):
    pd.DataFrame({"customer_id": IDS + ["A01"], "fold": [0, 1, 0, 1, 1]}).to_parquet(tmp_path / "dup.parquet", index=False)
    with pytest.raises(ValueError, match="several folds"):
        load_split_file(tmp_path / "dup.parquet", IDS)
    (tmp_path / "bad.csv").write_text("customer_id;fold\nA01;0\nA02;x\nB03;0\nB04;1\n")
    with pytest.raises(ValueError, match="non-integer folds"):
        load_split_file(tmp_path / "bad.csv", IDS)
    (tmp_path / "cols.csv").write_text("customer_id;group\nA01;0\n")
    with pytest.raises(ValueError, match="lacks columns"):
        load_split_file(tmp_path / "cols.csv", IDS)


def _rows() -> list[OOFRow]:
    return [
        OOFRow("A01", 0, 0, 0.91, True, date(2024, 3, 10), 9.5),
        OOFRow("A02", 0, 1, 0.20, False, date(2023, 7, 15), 5.0),   # FALSE rows get empty date and capacity
        OOFRow("B03", 0, 0, 0.05, False),
        OOFRow("B04", 0, 1, 0.70, True),                            # detection-only: TRUE with empty fields
        OOFRow("A01", 1, 1, 0.88, True, date(2024, 3, 1), 10.0),
        OOFRow("A02", 1, 0, 0.60, True, date(2023, 7, 1), 4.0),
        OOFRow("B03", 1, 1, 0.10, False),
        OOFRow("B04", 1, 0, 0.30, False),
    ]


def test_oof_file_has_exact_text_and_round_trips(tmp_path: Path):
    path = write_oof_predictions(reversed(_rows()), tmp_path / "oof.csv")
    text = path.read_text()
    assert text.splitlines()[0] == ";".join(OOF_COLUMNS)
    assert text.splitlines()[1] == "A01;0;0;0.910000;TRUE;2024-03-10;9.50"
    assert "A02;0;1;0.200000;FALSE;;" in text.splitlines()
    assert "B04;0;1;0.700000;TRUE;;" in text.splitlines()
    assert "\r" not in text
    frame = read_oof_predictions(path)
    assert list(frame.columns) == ["customer_id", "repeat", "fold", "prob", "pv_installed",
                                   "predicted_installation_date", "predicted_capacity_kwp"]
    # sorted by repeat, fold, customer: r0 -> A01, B03, A02, B04; r1 -> A02, B04, A01, B03
    assert frame.customer_id.tolist() == ["A01", "B03", "A02", "B04", "A02", "B04", "A01", "B03"]
    assert frame.pv_installed.tolist() == [True, False, False, True, True, False, True, False]
    assert frame.predicted_capacity_kwp.isna().tolist() == [False, True, True, True, False, True, False, True]
    assert frame.predicted_installation_date.isna().tolist() == [False, True, True, True, False, True, False, True]


def test_scoring_per_repeat_matches_the_official_scorer(tmp_path: Path):
    path = write_oof_predictions(_rows(), tmp_path / "oof.csv")
    labels = _labels()
    result = score_oof_file(path, labels)
    assert result["repeats"] == 2
    frame = read_oof_predictions(path)
    for block in result["per_repeat"]:
        rows = frame[frame.repeat == block["repeat"]].drop(columns=["repeat", "fold", "prob"])
        direct = score_predictions(labels, rows)
        assert block["main_score"] == pytest.approx(direct["main_score"])
        assert block["f1"] == pytest.approx(direct["f1"])
    # repeat 0: TP A01 (9 days off, 5% capacity error), FP B04, FN A02 -> precision = recall = f1 = 0.5
    r0 = result["per_repeat"][0]
    assert r0["f1"] == pytest.approx(0.5)
    assert r0["D"] == pytest.approx(1 - 9 / 30)       # B04 is a FP, so only A01 enters the D and P means
    assert r0["P"] == pytest.approx(1 - 0.05 / 0.5)
    assert r0["main_score"] == pytest.approx(0.5 * (70 + 15 * 0.7 + 15 * 0.9))
    # repeat 1: both PV customers found; A02 is 14 days and 20% off
    r1 = result["per_repeat"][1]
    assert r1["f1"] == pytest.approx(1.0)
    assert r1["D"] == pytest.approx((1 + (1 - 14 / 30)) / 2)
    assert r1["P"] == pytest.approx((1 + (1 - 0.2 / 0.5)) / 2)
    assert result["score_mean"] == pytest.approx((r0["main_score"] + r1["main_score"]) / 2)


def test_scoring_requires_full_coverage_per_repeat(tmp_path: Path):
    path = write_oof_predictions(_rows()[:3], tmp_path / "oof.csv")
    with pytest.raises(ValueError, match="does not cover the labelled customers exactly once"):
        score_oof_file(path, _labels())




def test_module_entry_point_runs(tmp_path: Path):
    labels = tmp_path / "labels.parquet"
    _labels().to_parquet(labels, index=False)
    oof = write_oof_predictions(_rows(), tmp_path / "oof.csv")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "neuralwave_experiment.oof_contract", "--labels", str(labels), str(oof)],
                            cwd=repo, env={"PYTHONPATH": str(repo / "src"), "PATH": ""}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("file=oof.csv repeats=2 ")


def test_pairing_rejects_changed_customer_fold_membership(tmp_path):
    from dataclasses import replace
    from neuralwave_experiment.oof_contract import paired_line
    first = write_oof_predictions(_rows(), tmp_path / "first.csv")
    changed = _rows()
    changed[0] = replace(changed[0], fold=1)
    second = write_oof_predictions(changed, tmp_path / "second.csv")
    result = paired_line(score_oof_file(first, _labels()), score_oof_file(second, _labels()))
    assert "skipped" in result
    assert "mean_diff=" not in result
    assert all(cid not in result for cid in IDS)


def test_pairing_accepts_identical_membership_in_different_row_order(tmp_path):
    from neuralwave_experiment.oof_contract import paired_line
    first = write_oof_predictions(_rows(), tmp_path / "first.csv")
    second = write_oof_predictions(reversed(_rows()), tmp_path / "second.csv")
    result = paired_line(score_oof_file(first, _labels()), score_oof_file(second, _labels()))
    assert "skipped" not in result
    assert "mean_diff=0.00" in result


@pytest.mark.parametrize("column", ["repeat", "fold"])
def test_fractional_membership_is_not_silently_truncated(tmp_path, column):
    path = write_oof_predictions(_rows(), tmp_path / "oof.csv")
    frame = pd.read_csv(path, sep=";")
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = .5
    frame.to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError):
        read_oof_predictions(path)
