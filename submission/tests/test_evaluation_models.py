import numpy as np
import pandas as pd
import pytest


def _batch(customer, value):
    from neuralwave_experiment.windows import WindowBatch
    return WindowBatch(
        values=np.full((2, 8, 1), value, dtype=np.float32),
        observed=np.ones((2, 8, 1), dtype=bool),
        calendar=np.zeros((2, 8, 4), dtype=np.float32),
        magnitude=np.ones((2, 4), dtype=np.float32),
        customer_ids=[customer, customer],
        center_dates=[pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-02").date()],
        start_epochs=np.array([1577836800, 1577923200], dtype=np.int64),
    )


def _inputs():
    train = {"negative": _batch("negative", -1.0), "positive": _batch("positive", 1.0)}
    labels = pd.DataFrame({
        "customer_id": ["negative", "positive"],
        "pv_installed": [False, True],
        "pv_installation_date": [pd.NaT, pd.Timestamp("2019-01-01")],
    })
    evaluation = {"heldout": _batch("heldout", 0.25)}
    config = {
        "context_length": 8, "patch_length": 4, "d_model": 8,
        "n_heads": 2, "n_layers": 1, "dropout": 0.0,
        "epochs": 1, "pretrain_epochs": 1, "batch_size": 2,
        "learning_rate": 1e-3, "weight_decay": 1e-2, "mask_ratio": .3,
        "model_seed": 17,
    }
    return train, labels, evaluation, config


def test_all_arms_return_deterministic_valid_probabilities(tmp_path):
    pytest.importorskip("torch")
    from neuralwave_experiment.evaluation_models import ARMS, fit_predict
    train, labels, evaluation, config = _inputs()
    for arm in ARMS:
        first = fit_predict(arm, train, labels, evaluation, config, 17, tmp_path / arm)
        second = fit_predict(arm, train, labels, evaluation, config, 17, tmp_path / (arm + "-repeat"))
        assert set(first) == {"heldout"}
        assert first["heldout"].shape == (2,)
        assert np.isfinite(first["heldout"]).all()
        assert np.all((first["heldout"] >= 0) & (first["heldout"] <= 1))
        assert np.allclose(first["heldout"], second["heldout"])

def test_training_and_evaluation_ids_must_be_disjoint(tmp_path):
    from neuralwave_experiment.evaluation_models import fit_predict
    train, labels, evaluation, config = _inputs()
    with pytest.raises(ValueError, match="disjoint"):
        fit_predict("hourly_logistic", train, labels, {"negative": evaluation["heldout"]}, config, 17, tmp_path)


def test_single_physical_training_class_is_rejected(tmp_path):
    from neuralwave_experiment.evaluation_models import fit_predict
    train, labels, evaluation, config = _inputs()
    one_class = labels.copy()
    one_class["pv_installed"] = False
    with pytest.raises(ValueError, match="both physical training classes"):
        fit_predict("hourly_logistic", train, one_class, evaluation, config, 17, tmp_path)
