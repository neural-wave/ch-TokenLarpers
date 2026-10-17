"""Partition canonical unlabelled measurements by customer, without using PV labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .io import source_hash
from .schema import MEASUREMENT_ARROW_SCHEMA


def partition(source_dir: str | Path, labels_path: str | Path, output_dir: str | Path,
              *, validation_fraction: float = .15, seed: int = 17) -> dict:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    provenance = json.loads((source / "run_manifest.json").read_text())
    if provenance.get("source_roles") != ["unlabelled_measurements"]:
        raise ValueError("pretraining partitions require unlabelled-only source provenance")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("partition output directory must be empty")
    parquet_path = source / "measurements.parquet"
    parquet = pq.ParquetFile(parquet_path)
    stored_schema = parquet.schema_arrow
    timestamp_index = stored_schema.get_field_index("timestamp_local")
    if timestamp_index >= 0:
        timestamp_type = stored_schema.field(timestamp_index).type
        if pa.types.is_timestamp(timestamp_type) and timestamp_type.tz == "Europe/Zurich":
            # Parquet stores second-resolution Arrow timestamps as milliseconds.
            stored_schema = stored_schema.set(timestamp_index, MEASUREMENT_ARROW_SCHEMA.field(timestamp_index))
    if not stored_schema.equals(MEASUREMENT_ARROW_SCHEMA, check_metadata=False):
        raise ValueError("measurements do not match the canonical schema")
    excluded = set(pq.read_table(labels_path, columns=["customer_id"]).column(0).to_pylist())
    customers = set()
    for batch in parquet.iter_batches(columns=["customer_id"], batch_size=100_000):
        customers.update(pc.unique(batch.column(0)).to_pylist())
    if customers & excluded:
        raise ValueError("unlabelled customers overlap labelled customers")
    if len(customers) < 2:
        raise ValueError("at least two unlabelled customers are required")
    ordered = np.asarray(sorted(customers), dtype=object)
    order = np.random.default_rng(seed).permutation(len(ordered))
    n_validation = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation = set(ordered[order[:n_validation]].tolist())
    training = customers - validation
    output.mkdir(parents=True, exist_ok=True)
    validation_ids = pa.array(sorted(validation), type=pa.string())
    counts = {"train": 0, "validation": 0}
    observed = {"train": 0, "validation": 0}
    writers = {name: pq.ParquetWriter(output / f"{name}.parquet", MEASUREMENT_ARROW_SCHEMA)
               for name in counts}
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch]).cast(MEASUREMENT_ARROW_SCHEMA)
            mask = pc.is_in(table["customer_id"], value_set=validation_ids)
            for name, keep in (("train", pc.invert(mask)), ("validation", mask)):
                part = table.filter(keep)
                if part.num_rows:
                    writers[name].write_table(part)
                    counts[name] += part.num_rows
                    observed[name] += pc.sum(pc.cast(pc.equal(part["value_status"], "observed"), pa.int64())).as_py()
    finally:
        for writer in writers.values():
            writer.close()
    if counts["train"] + counts["validation"] != parquet.metadata.num_rows:
        raise RuntimeError("partition row conservation failed")
    def audit(ids):
        return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    report = {
        "schema_version": 1, "seed": seed, "validation_fraction": validation_fraction,
        "train_customers": len(training), "validation_customers": len(validation),
        "labelled_customers_excluded": len(excluded), "customer_overlap": 0,
        "rows": counts, "observed_rows": observed,
        "train_customer_hash": audit(training), "validation_customer_hash": audit(validation),
        "exclusion_customer_hash": audit(excluded), "source_sha256": source_hash(parquet_path),
        "partition_sha256": {name: source_hash(output / f"{name}.parquet") for name in counts},
    }
    (output / "qa_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {"sources": [{"role": "unlabelled_measurements", "split": name,
                              "path": f"{name}.parquet"} for name in counts]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--exclude-labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    report = partition(args.source_dir, args.exclude_labels, args.out,
                       validation_fraction=args.validation_fraction, seed=args.seed)
    print(json.dumps({key: report[key] for key in ("train_customers", "validation_customers", "rows", "customer_overlap")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
