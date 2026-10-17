from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import train
from pvdetect.io import load_labels
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
