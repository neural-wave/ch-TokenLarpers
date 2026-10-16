# NeuralWave data pipelines

This branch combines canonical preprocessing, constrained synthetic PV generation, and offline numerical time-series embedding experiments. Raw organizer data and all data-derived artifacts remain private. Regression tests use independently fabricated inputs; authorized local evaluation can use AIL data.

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


Run the regression suite with `python -m pytest -q tests/test_preprocessing.py` after installing dependencies. The tests construct only temporary artificial CSV/JSON data and exercise the actual module CLI.

## Synthetic data

See [the synthesis usage guide](docs/synthetic_data.md). Use only confirmed non-PV training backgrounds and explicitly choose the meter mode. Synthetic capacity targets are deliberately invalid.

## Numerical embedding experiment

The [implementation plan](docs/time_series_experiment_plan.md) was written before parallel implementation. See [the experiment guide](docs/embedding_experiment.md) for the three-arm customer-held-out comparison and [encoder details](docs/representation_encoder.md).

```bash
python -m pip install -e '.[experiment]'
python -m neuralwave_experiment.experiment \
  --canonical-dir /private/artifacts/run \
  --out /private/artifacts/embedding-experiment \
  --config configs/embedding_experiment.json \
  --seed 17 --device cpu
```

The experiment compares raw-feature baseline, frozen numerical embeddings, and limited supervised fine-tuning. It does not download pretrained weights or use a language model. Optional synthetic inputs must be organized by held-out fold.

## Official PatchTST versus custom encoder

The [comparison guide](docs/official_patchtst_comparison.md) explains the attributed official-source adapter, its missingness/calendar adaptations, and matched evaluation.

```bash
python -m neuralwave_experiment.comparison \
  --canonical-dir /private/artifacts/run \
  --out /private/artifacts/patchtst-comparison \
  --config configs/embedding_experiment.json \
  --seed 17 --device cpu
```

This runs both backends with identical folds and training budgets, writes per-backend predictions/checkpoints and `comparison.json`, and requires identical raw-feature baseline outputs. Set `encoder_backend` to `custom` or `official_patchtst` to run just one backend through the experiment CLI. The upstream source is pinned and distributed with Apache-2.0 license and modification notices; no source checkout or download is needed at runtime.
