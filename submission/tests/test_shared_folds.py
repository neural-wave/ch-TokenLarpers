from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import train
from pvdetect.io import load_labels
from pvdetect.output import OOF_HEADER, PRED_HEADER
from pvdetect.scoring import main_score
from pvdetect.synthetic import generate_dataset, write_dataset


@pytest.fixture(scope="module")
def syn_small(tmp_path_factory) -> Path:
    """12 dev-seed customers (variant A) for the shared-fold contract."""
    out = tmp_path_factory.mktemp("syn_folds")
    write_dataset(out, *generate_dataset(12, seed=3))
    return out


def _write_split(path: Path, ids: list[str], folds: list[int], sep: str = ",") -> Path:
    pd.DataFrame({"customer_id": ids, "fold": folds}).to_csv(path, sep=sep, index=False)
    return path


def test_split_file_missing_a_labelled_customer_is_rejected(syn_small, tmp_path):
    ids = load_labels(syn_small / "labels.csv")["customer_id"].tolist()
    split = _write_split(tmp_path / "f0.csv", ids[1:], [i % 2 for i in range(len(ids) - 1)])
    argv = ["--timeseries", str(syn_small / "energy.csv"), "--labels", str(syn_small / "labels.csv"),
            "--out", str(tmp_path / "model.json"), "--cv", "--folds", str(split)]
    with pytest.raises(ValueError, match="does not cover labelled customers: 1 missing") as e:
        train.main(argv)
    assert ids[0] not in str(e.value)


def test_oof_file_matches_split_files_and_nested_scores(syn_small, tmp_path, monkeypatch):
    """Per repeat: one row per labelled customer, fold from its split file, and the same main score.

    train.py's per-repeat scores are captured by wrapping `train.nested_cv` with a spy that calls
    the real function and keeps its result: the exact numbers train.py computed, without a second
    nested run and without adding per-repeat lines to stdout. The tolerance only covers the
    contract's two-decimal capacity rounding (P is computed from the written kWp).
    """
    labels_path = syn_small / "labels.csv"
    ids = sorted(load_labels(labels_path)["customer_id"].tolist())
    maps = [{c: i % 3 for i, c in enumerate(ids)}, {c: (i // 2) % 2 + 5 for i, c in enumerate(ids)}]
    f0 = _write_split(tmp_path / "f0.csv", list(maps[0]), list(maps[0].values()))
    f1 = _write_split(tmp_path / "f1.csv", list(maps[1]), list(maps[1].values()), sep=";")
    real_nested_cv, results = train.nested_cv, []

    def spy(*args, **kwargs):
        results.append(real_nested_cv(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(train, "nested_cv", spy)
    oof = tmp_path / "oof.csv"
    argv = ["--timeseries", str(syn_small / "energy.csv"), "--labels", str(labels_path),
            "--out", str(tmp_path / "model.json"), "--cv", "--folds", f"{f0},{f1}", "--oof-out", str(oof)]
    assert train.main(argv) == 0
    assert oof.read_text(encoding="utf-8").splitlines()[0] == ";".join(OOF_HEADER)
    raw = pd.read_csv(oof, sep=";", dtype=str, keep_default_na=False)
    assert sorted(set(raw["repeat"])) == ["0", "1"]
    truth = load_labels(labels_path)
    for r, fold_map in enumerate(maps):
        block = raw[raw["repeat"] == str(r)]
        assert sorted(block["customer_id"]) == ids
        assert block["fold"].astype(int).tolist() == block["customer_id"].map(fold_map).tolist()
        block[PRED_HEADER].to_csv(tmp_path / f"pred{r}.csv", sep=";", index=False)
        score = main_score(load_labels(tmp_path / f"pred{r}.csv"), truth)["score"]
        assert score == pytest.approx(results[0].repeat_scores[r]["score"], abs=0.05)


def test_relabelling_folds_does_not_create_a_new_outer_repeat(tmp_path):
    from neuralwave_experiment.oof_contract import load_split_files
    ids = ["PODSYN_A", "PODSYN_B", "PODSYN_C", "PODSYN_D"]
    first = _write_split(tmp_path / "first.csv", ids, [0, 0, 1, 1])
    relabelled = _write_split(tmp_path / "relabelled.csv", ids, [9, 9, 4, 4])
    with pytest.raises(ValueError, match="genuinely different"):
        load_split_files([first, relabelled], ids)
    distinct = _write_split(tmp_path / "distinct.csv", ids, [0, 1, 0, 1])
    assert load_split_files([first, distinct], ids)[1] == dict(zip(ids, [0, 1, 0, 1]))
