from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from pvdetect.embedding import fit_projection, transform_projection
from pvdetect.model import CURRENT_VERSION, load_model, save_model


def _artifact_source() -> dict:
    return {
        "population": "unlabelled_only",
        "labels_used": False,
        "train_customer_count": 4,
        "val_customer_count": 2,
        "train_exclusion_hash": "a" * 64,
        "val_exclusion_hash": "b" * 64,
    }


def test_baseline_model_does_not_copy_encoder(tmp_path):
    model = {"version": CURRENT_VERSION, "_objects": {}}
    out = tmp_path / "baseline" / "model.json"
    save_model(model, out)
    assert not (out.parent / "encoder").exists()
    assert load_model(out)["version"] == CURRENT_VERSION


def test_real_saved_encoder_load_and_projection_reload(tmp_path):
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.checkpoint import export_encoder_artifact
    from neuralwave_experiment.encoder import EncoderConfig
    from neuralwave_experiment.encoder_factory import build_encoder

    config = EncoderConfig(context_length=32, patch_length=8, d_model=8, n_heads=2, n_layers=1)
    encoder = build_encoder(config, backend="official_patchtst")
    source = tmp_path / "source"
    export_encoder_artifact(
        source,
        encoder,
        encoder_config=asdict(config),
        window_config={"context_length": 32, "stride": 16, "min_observed_fraction": 0.5},
        source=_artifact_source(),
    )
    raw = np.arange(32, dtype=float).reshape(4, 8)
    projection = fit_projection(raw, ["0", "1", "2", "3"])
    out = tmp_path / "model"
    model = {
        "version": CURRENT_VERSION,
        "embedding": {
            "feature_names": ["embedding_pc1", "embedding_pc2", "embedding_pc3", "embedding_pc4"],
            "projection": projection,
            "encoder_files": {"weights": "encoder/encoder.pt", "metadata": "encoder/encoder.json"},
            "encoder_hashes": {},
        },
        "_encoder_source": str(source),
        "_objects": {},
    }
    save_model(model, out / "model.json")
    loaded = load_model(out / "model.json")
    assert loaded["embedding"]["projection"] == projection
    x = transform_projection(raw, ["0", "1", "2", "3"], projection)
    assert np.isfinite(x.to_numpy()).all()
    assert hashlib.sha256((out / "encoder" / "encoder.pt").read_bytes()).hexdigest() == loaded["embedding"]["encoder_hashes"]["encoder/encoder.pt"]
