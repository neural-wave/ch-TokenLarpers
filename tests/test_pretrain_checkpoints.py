from __future__ import annotations
import json
import shutil
import torch
import pytest
from neuralwave_experiment.encoder import EncoderConfig
from neuralwave_experiment.encoder_factory import build_encoder
from neuralwave_experiment.checkpoint import export_encoder_artifact, load_encoder_artifact


def _model():
    return build_encoder(EncoderConfig(context_length=16, patch_length=4, d_model=8, n_heads=2, n_layers=1, dropout=0.0), backend="official_patchtst")


def _source():
    return {"population":"unlabelled_only", "labels_used":False, "train_customer_count":2, "val_customer_count":1, "train_exclusion_hash":"a"*64, "val_exclusion_hash":"b"*64}


def test_export_load_is_exact_and_frozen(tmp_path):
    model = _model(); model.eval()
    out = export_encoder_artifact(tmp_path / "encoder", model, encoder_config={"context_length":16,"patch_length":4,"d_model":8,"n_heads":2,"n_layers":1,"dropout":0.0}, window_config={"context_length":16,"stride":8,"min_observed_fraction":.75,"max_windows_per_customer":4}, source=_source())
    assert {p.name for p in out.iterdir()} == {"encoder.pt", "encoder.json"}
    loaded, metadata = load_encoder_artifact(out)
    assert metadata["source"]["labels_used"] is False
    assert all(not p.requires_grad for p in loaded.parameters()) and not loaded.training


def test_artifact_rejects_hash_corruption_and_extra_files(tmp_path):
    out = export_encoder_artifact(tmp_path / "encoder", _model(), encoder_config={"context_length":16,"patch_length":4,"d_model":8,"n_heads":2,"n_layers":1,"dropout":0.0}, window_config={"context_length":16}, source=_source())
    (out / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="exactly"):
        load_encoder_artifact(out)
    (out / "extra.txt").unlink(); (out / "encoder.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_encoder_artifact(out)


def test_export_refuses_nonempty_directory(tmp_path):
    out = tmp_path / "encoder"; out.mkdir(); (out / "keep").write_text("x")
    with pytest.raises(FileExistsError):
        export_encoder_artifact(out, _model(), encoder_config={"context_length":16,"patch_length":4,"d_model":8,"n_heads":2,"n_layers":1,"dropout":0.0}, window_config={"context_length":16}, source=_source())
