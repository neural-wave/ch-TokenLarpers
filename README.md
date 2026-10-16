# NeuralWave data pipelines

The repository implements Pipeline A, strict, privacy-safe conversion of authorized CSV manifests into the versioned canonical Parquet contract, and Pipeline B, deterministic paired synthetic PV scenarios. Neither pipeline reads, packages, or logs organizer files during development; use fabricated fixtures for checks.

## Install and run

```bash
python -m pip install -e .
python -m neuralwave_data.prepare \
  --manifest /private/manifest.json \
  --out /private/artifacts/run \
  --config config.preprocess.json \
  --folds 5 --seed 17
```

A manifest is JSON with a `sources` list. Every source has an explicit `role`, `path`, `delimiter`, `decimal`, and `encoding`; paths are relative to the manifest. Measurement roles are `development_measurements`, `unlabelled_measurements`, and `inference_measurements`. Label roles are `development_labels` and `extension_labels`. Defaults are UTF-8, semicolon, and `.` when fields are omitted. Timestamps must be whole Unix seconds on a 15-minute grid or timezone-aware timestamp strings; a naive `timestamp_local` is accepted only alongside epoch seconds and is checked as a Europe/Zurich wall clock.
The output directory is never overwritten when populated. It contains `measurements.parquet`, `daily_features.parquet`, `run_manifest.json`, and `qa_report.json`; labels, extensions, and splits are written when supplied/requested. `measurements.parquet` preserves null energy and distinguishes `observed`, `missing_input`, and reindexed `absent_interval` records. Daily features use the exact observed-sum field name `energy_observed_kwh`. IDs remain strings, including leading zeroes. Raw inputs and all generated artifacts must stay under private paths.
Unlabelled rows are filtered before parsing: only non-null, case-sensitive literal IDs beginning with `POD` remain. Null and non-POD exclusions are counted separately in the private `qa_report.json`. This filter is not applied to labelled or inference roles.

The output directory is never overwritten when populated. It contains `measurements.parquet`, `daily_features.parquet`, `run_manifest.json`, and `qa_report.json`; labels, extensions, and splits are written when supplied/requested. `measurements.parquet` preserves null energy and distinguishes `observed`, `missing_input`, and reindexed `absent_interval` records. IDs remain strings, including leading zeroes. Raw inputs and all generated artifacts must stay under private paths.

Run the regression suite with `PYTHONPATH=src python -m pytest -q tests/test_preprocessing.py tests/test_augmentation.py` after installing dependencies. The tests construct only temporary artificial data and exercise preprocessing and synthesis.

## Synthetic data

See [the synthetic data guide](docs/synthetic_data.md) for the synthesis CLI, validation API, output contract, and the fully fabricated local dataset example. Generated measurements, labels, provenance, and latent components stay under ignored private directories; they are not distributed through Git.
