from __future__ import annotations

from pathlib import Path

import pytest

from pvdetect.synthetic import generate_dataset, write_dataset


@pytest.fixture(scope="session")
def syn_train(tmp_path_factory) -> Path:
    """24 dev-seed customers (variant A, hard negatives, extensions) written with commas."""
    out = tmp_path_factory.mktemp("syn_train")
    write_dataset(out, *generate_dataset(24, seed=1, with_extensions=True))
    return out


@pytest.fixture(scope="session")
def syn_test(tmp_path_factory) -> Path:
    """12 dev-seed customers written with semicolons."""
    out = tmp_path_factory.mktemp("syn_test")
    write_dataset(out, *generate_dataset(12, seed=2, with_extensions=True), sep=";")
    return out
