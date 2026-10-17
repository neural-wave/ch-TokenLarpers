import numpy as np
from dataclasses import dataclass
import pandas as pd

from neuralwave_experiment.experiment import _initial_batch, _initial_indices


@dataclass
class _Batch:
    values: np.ndarray
    observed: np.ndarray
    calendar: np.ndarray
    magnitude: np.ndarray
    customer_ids: list[str]
    center_dates: list[object]
    start_epochs: np.ndarray


def _batch():
    n = 3
    return _Batch(np.arange(n * 4, dtype=np.float32).reshape(n, 4, 1),
                  np.ones((n, 4, 1), dtype=bool), np.zeros((n, 4, 4), dtype=np.float32),
                  np.zeros((n, 4), dtype=np.float32), ["a", "a", "b"],
                  [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date(), pd.Timestamp("2024-01-01").date()],
                  np.arange(n))


def test_initial_capacity_indices_align_batch_and_embedding():
    batch = _batch()
    extensions = pd.DataFrame({"customer_id": ["a"], "extension_date": ["2024-02-01"]})
    indices = _initial_indices(batch, extensions)
    trimmed = _initial_batch(batch, extensions)
    assert len(indices) == len(trimmed.values)
    assert np.array_equal(trimmed.values, batch.values[indices])
    embedding = np.arange(len(batch.values) * 2).reshape(len(batch.values), 2)
    assert np.array_equal(embedding[indices], embedding[[0, 2]])


