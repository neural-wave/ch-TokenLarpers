import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from neuralwave_data.prepare import prepare
from neuralwave_data.pretraining_split import partition
from neuralwave_data.schema import arrow_table


def fixture(tmp_path):
    raw = tmp_path / "input.csv"
    raw.write_text("customer_id;timestamp_epoch;energy_consumption_kwh\n" + "".join(
        f"PODSYN{i:03d};0;{i - 2}\nPODSYN{i:03d};1800;\n" for i in range(12)))
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"sources": [{"path": str(raw), "role": "unlabelled_measurements"}]}))
    source = tmp_path / "canonical"
    prepare(manifest, source)
    labels = tmp_path / "labels.parquet"
    frame = pd.DataFrame({"customer_id": ["PODSYN_LABELLED"], "pv_installed": [False],
                          "pv_installation_date": [None], "pv_capacity_kwp": [None]})
    pq.write_table(arrow_table(frame, "labels"), labels)
    return source, labels


def test_partition_conserves_values_and_keeps_whole_customers(tmp_path):
    source, labels = fixture(tmp_path)
    a = partition(source, labels, tmp_path / "a", validation_fraction=.25, seed=17)
    b = partition(source, labels, tmp_path / "b", validation_fraction=.25, seed=17)
    train = pq.read_table(tmp_path / "a/train.parquet").to_pandas()
    val = pq.read_table(tmp_path / "a/validation.parquet").to_pandas()
    assert len(set(train.customer_id)) == 9 and len(set(val.customer_id)) == 3
    assert not set(train.customer_id) & set(val.customer_id)
    expected = pq.read_table(source / "measurements.parquet").to_pandas()
    actual = pd.concat([train, val]).sort_values(["customer_id", "timestamp_epoch"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))
    assert a["train_customer_hash"] == b["train_customer_hash"]
    assert a["validation_customer_hash"] == b["validation_customer_hash"]
    assert sum(a["rows"].values()) == 36
    assert sum(a["observed_rows"].values()) == 12
    private_free_report = json.dumps(a)
    assert all(cid not in private_free_report for cid in expected.customer_id.unique())


def test_partition_rejects_labelled_overlap_and_wrong_source_role(tmp_path):
    source, labels = fixture(tmp_path)
    frame = pd.DataFrame({"customer_id": ["PODSYN000"], "pv_installed": [False],
                          "pv_installation_date": [None], "pv_capacity_kwp": [None]})
    pq.write_table(arrow_table(frame, "labels"), labels)
    with pytest.raises(ValueError, match="overlap"):
        partition(source, labels, tmp_path / "bad")
    metadata = source / "run_manifest.json"
    payload = json.loads(metadata.read_text()); payload["source_roles"] = ["development_measurements"]
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unlabelled-only"):
        partition(source, labels, tmp_path / "wrong-role")
