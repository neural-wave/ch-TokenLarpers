from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvdetect.embedding import ProjectionUnavailable, fit_projection, transform_projection


def test_projection_is_four_components_and_excludes_heldout_rows():
    ids = ["a", "b", "c", "d", "heldout"]
    raw = pd.DataFrame(np.arange(25, dtype=float).reshape(5, 5), index=ids)
    p1 = fit_projection(raw, ids[:4])
    changed = raw.copy()
    changed.loc["heldout"] = 1e12
    p2 = fit_projection(changed, ids[:4])
    assert p1 == p2
    got = transform_projection(raw, ids, p1)
    assert list(got.columns) == ["embedding_pc1", "embedding_pc2", "embedding_pc3", "embedding_pc4"]
    assert got.shape == (5, 4)


def test_projection_rejects_insufficient_training_rows():
    raw = pd.DataFrame(np.ones((3, 8)), index=["a", "b", "c"])
    with pytest.raises(ProjectionUnavailable):
        fit_projection(raw, ["a", "b", "c"])


def test_missing_raw_customer_is_explicitly_imputed():
    raw = pd.DataFrame(np.eye(4), index=["a", "b", "c", "d"])
    p = fit_projection(raw, list(raw.index))
    out = transform_projection(raw, ["missing"], p)
    assert np.isfinite(out.to_numpy()).all()
