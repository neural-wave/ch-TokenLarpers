"""Paired OOF comparison on hand-built out-of-fold files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvdetect import compare
from pvdetect.diagnose import error_table, summarize
from pvdetect.output import write_oof

REPO = Path(__file__).resolve().parents[1]


def _labels(path: Path, n: int) -> Path:
    on = np.arange(n) % 2 == 0
    rows = ["customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"]
    rows += [f"{i:03d};{'TRUE' if o else 'FALSE'};{'2023-05-01' if o else ''};{'10' if o else ''}" for i, o in enumerate(on)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _oof(path: Path, pred_pv, repeats: int = 2, folds=None, date="2023-05-01", kwp=10.0) -> Path:
    rows = []
    for r in range(repeats):
        for i, p in enumerate(pred_pv):
            fold = (i + r) % 2 if folds is None else folds[r][i]
            rows.append((f"{i:03d}", r, fold, 0.9 if p else 0.1, bool(p), pd.Timestamp(date), kwp))
    cols = ["customer_id", "repeat", "fold", "prob", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
    write_oof(pd.DataFrame(rows, columns=cols), path)
    return path


def test_counts_rates_and_paired_changes_are_exact(tmp_path):
    labels = _labels(tmp_path / "labels.csv", 6)  # PV: 000, 002, 004
    base = _oof(tmp_path / "base.csv", [1, 1, 0, 0, 0, 0])  # TP FP FN TN FN TN
    cand = _oof(tmp_path / "cand.csv", [1, 0, 1, 0, 0, 1])  # TP TN TP TN FN FP
    lines, changes = compare.compare(compare.load_oof(base), compare.load_oof(cand), compare.load_labels(labels))
    assert lines[0] == "compare: repeats=2 customers=6 (aggregates only)"
    assert lines[1] == (
        "repeat 0 base: TP=1 FP=1 FN=2 TN=2 precision=0.500 recall=0.333 f1=0.400 D=1.000 P=1.000 score=40.00"
    )
    assert lines[2] == (
        "repeat 0 cand: TP=2 FP=1 FN=1 TN=2 precision=0.667 recall=0.667 f1=0.667 D=1.000 P=1.000 score=66.67"
    )
    assert lines[3] == "repeat 0 paired: pv wrong->right=1 right->wrong=0 | nonpv wrong->right=1 right->wrong=1"
    assert lines[7].startswith("total base (pooled over repeats): TP=2 FP=2 FN=4 TN=4 ")
    assert lines[9] == "total paired: pv wrong->right=2 right->wrong=0 | nonpv wrong->right=2 right->wrong=2"
    assert lines[10].startswith("bootstrap: score_diff=+26.67 ")
    assert len(changes) == 12 and set(changes["repeat"]) == {0, 1}
    row = changes[changes["repeat"] == 0].loc["002"]
    assert (row["base_outcome"], row["cand_outcome"], row["base_pred_pv"], row["cand_pred_pv"]) == ("FN", "TP", False, True)


def test_bootstrap_point_estimate_matches_the_scorer(tmp_path):
    labels = compare.load_labels(_labels(tmp_path / "labels.csv", 10))
    base = compare.load_oof(_oof(tmp_path / "b.csv", [1, 1, 0, 0, 1, 0, 0, 0, 1, 1], date="2023-05-11", kwp=8.0))
    cand = compare.load_oof(_oof(tmp_path / "c.csv", [1, 0, 1, 0, 1, 0, 1, 0, 0, 1], date="2023-05-21", kwp=13.0))
    pairs = []
    for r in (0, 1):
        pairs.append(tuple(error_table(f[f["repeat"] == r], labels) for f in (base, cand)))
    diff, lo, hi = compare.paired_bootstrap(pairs)
    expect = np.mean([summarize(tc)["score"] - summarize(tb)["score"] for tb, tc in pairs])
    assert diff == pytest.approx(expect)
    assert lo <= diff <= hi


def test_single_flip_is_within_noise_and_a_large_gain_is_an_improvement(tmp_path):
    labels = compare.load_labels(_labels(tmp_path / "labels.csv", 40))
    truth = [i % 2 == 0 for i in range(40)]
    one_wrong = list(truth)
    one_wrong[0] = False
    base = compare.load_oof(_oof(tmp_path / "b.csv", one_wrong))
    cand = compare.load_oof(_oof(tmp_path / "c.csv", truth))
    lines, _ = compare.compare(base, cand, labels)
    assert lines[-1].endswith("verdict=within_noise")
    poor = compare.load_oof(_oof(tmp_path / "p.csv", [not t if i < 12 else t for i, t in enumerate(truth)]))
    assert compare.compare(poor, cand, labels)[0][-1].endswith("verdict=improvement")
    assert compare.compare(cand, poor, labels)[0][-1].endswith("verdict=worse")


def test_mismatched_folds_are_refused(tmp_path, capsys):
    labels = _labels(tmp_path / "labels.csv", 6)
    base = _oof(tmp_path / "b.csv", [1, 0, 1, 0, 1, 0])
    other = [[0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0]]
    cand = _oof(tmp_path / "c.csv", [1, 0, 1, 0, 1, 0], folds=other)
    with pytest.raises(ValueError, match="different folds"):
        compare.compare(compare.load_oof(base), compare.load_oof(cand), compare.load_labels(labels))
    short = _oof(tmp_path / "s.csv", [1, 0, 1, 0, 1, 0], repeats=1)
    with pytest.raises(SystemExit, match="refusing: .*different"):
        compare.main([str(base), str(short), str(labels)])
    assert capsys.readouterr().out == ""


def test_out_dir_inside_repo_refused_and_changes_never_printed(tmp_path, capsys):
    labels = _labels(tmp_path / "labels.csv", 6)
    base = _oof(tmp_path / "b.csv", [1, 1, 0, 0, 0, 0])
    cand = _oof(tmp_path / "c.csv", [1, 0, 1, 0, 0, 1])
    with pytest.raises(SystemExit, match="inside the repository"):
        compare.main([str(base), str(cand), str(labels), "--out-dir", str(REPO / "private_out")])
    assert not (REPO / "private_out").exists()
    out_dir = tmp_path / "private"
    assert compare.main([str(base), str(cand), str(labels), "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "002" not in printed and "changes.csv" in printed
    changes = pd.read_csv(out_dir / "changes.csv", sep=";", dtype={"customer_id": str})
    assert len(changes) == 12 and changes["customer_id"].str.len().eq(3).all()
