from __future__ import annotations

import socket

import pytest

import inference
import train
from pvdetect.io import load_labels
from pvdetect.scoring import main_score
from pvdetect.synthetic import generate_dataset, write_dataset


@pytest.fixture(scope="module")
def trained(syn_train, tmp_path_factory):
    model = tmp_path_factory.mktemp("model") / "model.json"
    rc = train.main(
        ["--timeseries", str(syn_train / "energy.csv"), "--labels", str(syn_train / "labels.csv"), "--out", str(model)]
    )
    assert rc == 0
    return model


def _block_network(*args, **kwargs):
    raise RuntimeError("network blocked")


def test_offline_inference_contract(trained, syn_test, tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_PATH", trained)
    monkeypatch.setattr(socket, "socket", _block_network)
    monkeypatch.chdir(tmp_path)
    assert inference.main([str(syn_test / "energy.csv")]) == 0
    lines = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"
    pred = load_labels(tmp_path / "predictions.csv")
    truth = load_labels(syn_test / "labels.csv")
    assert pred["customer_id"].tolist() == sorted(truth["customer_id"].tolist())
    assert pred["customer_id"].is_unique and len(pred) == 12
    on = pred["pv_installed"]
    assert pred.loc[on, "pv_installation_date"].notna().all()
    assert (pred.loc[on, "pv_capacity_kwp"] > 0).all()
    assert pred.loc[~on, "pv_installation_date"].isna().all() and pred.loc[~on, "pv_capacity_kwp"].isna().all()
    assert main_score(pred, truth)["f1"] >= 0.8
    ext_lines = (tmp_path / "pv_extensions_predictions.csv").read_text().splitlines()
    assert ext_lines[0] == "customer_id;extension_date;pv_capacity_after_extension_kwp"


@pytest.mark.parametrize("argv", [[], ["a.csv", "b.csv"]])
def test_exactly_one_positional_argument(argv):
    with pytest.raises(SystemExit) as e:
        inference.main(argv)
    assert e.value.code == 2


def test_missing_model_is_reported(tmp_path, monkeypatch, syn_test):
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        inference.main([str(syn_test / "energy.csv")])


def test_run_inference_makes_no_network_calls(tmp_path_factory, monkeypatch):
    """inference.run_inference must never open a socket (proves no network at inference)."""
    data_dir = tmp_path_factory.mktemp("offline_data")
    write_dataset(data_dir, *generate_dataset(6, seed=77))
    model_path = tmp_path_factory.mktemp("offline_model") / "model.json"
    assert train.main(["--timeseries", str(data_dir / "energy.csv"), "--labels", str(data_dir / "labels.csv"), "--out", str(model_path)]) == 0

    out_dir = tmp_path_factory.mktemp("offline_out")
    monkeypatch.setattr(socket, "socket", _block_network)
    result = inference.run_inference(data_dir / "energy.csv", model_path, out_dir)
    pred = result.pred
    assert len(pred) == 6
    assert set(pred.columns) >= {"customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"}
    on = pred["pv_installed"]
    assert pred.loc[on, "pv_installation_date"].notna().all()
    assert (pred.loc[on, "pv_capacity_kwp"] > 0).all()
