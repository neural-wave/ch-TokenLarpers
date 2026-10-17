from __future__ import annotations

from datetime import date
import shutil
import copy
import numpy as np
import pandas as pd
import pytest
import torch

from neuralwave_experiment.encoder import EncoderConfig, NumericPatchEncoder
from neuralwave_experiment.streaming import EpochSource, WindowConfig, WindowStreamConfig, WindowSamplerState, assert_unlabelled_only
from neuralwave_experiment.windows import WindowBatch
from neuralwave_experiment.pretraining import _visible_inputs, PretrainConfig, pretrain_encoder_stream
from neuralwave_experiment.encoder_factory import build_encoder


def _rows(customer: str, start: int, count: int) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [customer] * count,
        "timestamp_epoch": np.arange(start, start + count * 900, 900, dtype=np.int64),
        "energy_kwh": np.arange(count, dtype=float) + 1.0,
        "value_status": ["observed"] * count,
    })


def _stream_cfg() -> WindowStreamConfig:
    return WindowStreamConfig(WindowConfig(context_length=16, stride=8, min_observed_fraction=.5), max_windows_per_customer=8, windows_per_epoch=4, batch_size=2, seed=3)


def test_customer_parts_span_interleaved_parquet_files(tmp_path):
    first = pd.concat([_rows("POD-B", 0, 16), _rows("POD-A", 0, 16)], ignore_index=True)
    second = pd.concat([_rows("POD-A", 16 * 900, 16), _rows("POD-B", 16 * 900, 16)], ignore_index=True)
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    first.to_parquet(left, index=False); second.to_parquet(right, index=False)
    source = EpochSource([left, right], role="unlabelled_measurements", cache_dir=tmp_path / "cache")
    customers = list(source.iter_customers())
    assert [item.customer_id for item in customers] == ["POD-A", "POD-B"]
    assert [len(item.frame) for item in customers] == [32, 32]
    index = source.index_windows(_stream_cfg())
    assert index.n_customers == 2 and index.n_windows > 0
    batches = list(source.iter_window_batches(_stream_cfg()))
    starts = [(customer, int(start)) for batch, _ in batches for customer, start in zip(batch.customer_ids, batch.start_epochs)]
    assert len(starts) == len(set(starts))


def test_unlabelled_exclusion_and_train_validation_disjointness(tmp_path):
    train_path, val_path = tmp_path / "train.parquet", tmp_path / "val.parquet"
    _rows("POD-TRAIN", 0, 16).to_parquet(train_path, index=False)
    _rows("POD-VAL", 0, 16).to_parquet(val_path, index=False)
    train = EpochSource(train_path, role="unlabelled_measurements", labelled_ids={"POD-LABELLED"})
    val = EpochSource(val_path, role="unlabelled_measurements", labelled_ids={"POD-LABELLED"})
    assert set(train.index_windows(_stream_cfg())._customer_ids).isdisjoint(val.index_windows(_stream_cfg())._customer_ids)
    with pytest.raises(ValueError, match="overlaps"):
        EpochSource(train_path, role="unlabelled_measurements", labelled_ids={"POD-TRAIN"})
    with pytest.raises(ValueError, match="unlabelled_measurements"):
        assert_unlabelled_only(EpochSource(train_path, role="labelled_measurements"), set())


def test_source_bytes_identity_prevents_stale_shared_cache(tmp_path):
    original = tmp_path / "one.parquet"; duplicate = tmp_path / "two.parquet"
    _rows("POD-A", 0, 16).to_parquet(original, index=False); shutil.copyfile(original, duplicate)
    cache = tmp_path / "cache"
    first = EpochSource(original, role="unlabelled_measurements", cache_dir=cache)
    second = EpochSource(duplicate, role="unlabelled_measurements", cache_dir=cache)
    first_index = first.index_windows(_stream_cfg()); second_index = second.index_windows(_stream_cfg())
    assert first.source_bytes_sha256 != second.source_bytes_sha256
    assert first_index._shards != second_index._shards


def test_visible_normalization_ignores_hidden_target_values():
    model = NumericPatchEncoder(EncoderConfig(context_length=16, patch_length=4, d_model=8, n_heads=2, n_layers=1, dropout=0.0))
    values = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    changed = values.copy(); changed[:, :4, 0] += 10000
    common = dict(observed=np.ones((1, 16, 1), dtype=bool), calendar=np.zeros((1, 16, 4), dtype=np.float32), magnitude=np.array([[0, 1, 0, 1]], dtype=np.float32), customer_ids=["POD-A"], center_dates=[date(2022, 1, 1)], start_epochs=np.array([0]))
    target = torch.tensor([[True, False, False, False]])
    a = WindowBatch(values, **common); b = WindowBatch(changed, **common)
    masked_a, observed_a, _, _, _ = _visible_inputs(model, a, target, "cpu")
    masked_b, observed_b, _, _, _ = _visible_inputs(model, b, target, "cpu")
    assert torch.equal(masked_a, masked_b) and torch.equal(observed_a, observed_b)


@pytest.mark.parametrize("stop_step", [1, 2])
def test_interrupted_cpu_resume_reproduces_model_and_sampler(tmp_path, stop_step):
    path = tmp_path / "measurements.parquet"; _rows("POD-A", 0, 32).to_parquet(path, index=False)
    source = EpochSource(path, role="unlabelled_measurements", cache_dir=tmp_path / "cache")
    validation = EpochSource.from_frame(_rows("PODSYNVAL", 0, 32))
    stream_cfg = _stream_cfg(); train_cfg = PretrainConfig(epochs=2, batch_size=2, learning_rate=.001, amp=False, seed=11)
    enc_cfg = EncoderConfig(context_length=16, patch_length=4, d_model=8, n_heads=2, n_layers=1, dropout=0.)
    torch.manual_seed(22); full_model = build_encoder(enc_cfg, backend="official_patchtst")
    final_snapshots = []
    full = pretrain_encoder_stream(full_model, source, stream_cfg, train_cfg, device="cpu",
        validation_source=validation, checkpoint=lambda payload: final_snapshots.append(copy.deepcopy(payload)))
    saved = []
    torch.manual_seed(22); interrupted_model = build_encoder(enc_cfg, backend="official_patchtst")
    def stop(payload):
        saved.append(payload)
        if payload["state"]["progress"]["global_step"] == stop_step: raise RuntimeError("intentional interruption")
    with pytest.raises(RuntimeError, match="intentional"):
        pretrain_encoder_stream(interrupted_model, source, stream_cfg, train_cfg, device="cpu", checkpoint=stop, validation_source=validation)
    torch.manual_seed(22); resumed_model = build_encoder(enc_cfg, backend="official_patchtst")
    resumed = pretrain_encoder_stream(resumed_model, source, stream_cfg, train_cfg, device="cpu", resume=saved[-1], validation_source=validation)
    assert resumed.sampler_state.as_dict() == full.sampler_state.as_dict()
    assert all(torch.equal(full_model.state_dict()[name], resumed_model.state_dict()[name]) for name in full_model.state_dict())
    assert resumed.history == full.history
    assert resumed.progress == full.progress
    completed = pretrain_encoder_stream(resumed_model, source, stream_cfg, train_cfg,
        device="cpu", resume=final_snapshots[-1], validation_source=validation)
    assert completed.history == full.history and completed.progress == full.progress
    assert all(torch.equal(full_model.state_dict()[name], resumed_model.state_dict()[name]) for name in full_model.state_dict())
def test_customer_round_robin_sampling_coverage(tmp_path):
    parts = []
    for i in range(6):
        parts.append(_rows(f"POD-CUST-{i}", 0, 32))
    path = tmp_path / "measurements.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    source = EpochSource(path, role="unlabelled_measurements", cache_dir=tmp_path / "cache")
    stream_cfg = WindowStreamConfig(WindowConfig(context_length=16, stride=8, min_observed_fraction=.5), max_windows_per_customer=8, windows_per_epoch=6, batch_size=2, seed=7)
    batches = list(source.iter_window_batches(stream_cfg))
    cids = [cid for b, _ in batches for cid in b.customer_ids]
    assert len(cids) == 6
    assert set(cids) == {f"POD-CUST-{i}" for i in range(6)}


def test_early_stopping_and_best_checkpoint_saved(tmp_path):
    path = tmp_path / "measurements.parquet"; _rows("POD-TRAIN", 0, 32).to_parquet(path, index=False)
    source = EpochSource(path, role="unlabelled_measurements", cache_dir=tmp_path / "cache")
    val_source = EpochSource.from_frame(_rows("POD-VAL", 0, 32))
    stream_cfg = WindowStreamConfig(WindowConfig(context_length=16, stride=8, min_observed_fraction=.5), max_windows_per_customer=8, windows_per_epoch=2, batch_size=2, seed=3)
    train_cfg = PretrainConfig(epochs=4, batch_size=2, learning_rate=.001, amp=False, seed=11, early_stopping_patience=2)
    enc_cfg = EncoderConfig(context_length=16, patch_length=4, d_model=8, n_heads=2, n_layers=1, dropout=0.)
    model = build_encoder(enc_cfg, backend="official_patchtst")
    saved_payloads = []
    result = pretrain_encoder_stream(model, source, stream_cfg, train_cfg, device="cpu",
                                    validation_source=val_source, checkpoint=lambda p: saved_payloads.append(p))
    assert result.progress.get("best_validation_loss") < float("inf")
    assert any(p.get("is_best") for p in saved_payloads)
