# NeuralWave data pipelines

This branch combines canonical preprocessing, constrained synthetic PV generation, offline numerical time-series embedding experiments, and the Task 1 PatchTST window classifier. Raw organizer data and all data-derived artifacts remain private. Regression tests use independently fabricated inputs; authorized local evaluation can use AIL data.

## Data hygiene

The repository is open-sourced after the event, so nothing derived from the AIL data may enter git history: raw files and generated artifacts stay under private, git-ignored paths (`data_private/`, `artifacts_private/`), and reports such as [benchmarks.md](benchmarks.md) hold aggregate metrics only, never customer identifiers, per-customer readings, probabilities or label dates.

Install the commit-time check once per clone:

```bash
sh scripts/install_hooks.sh
```

It copies `scripts/check_hygiene.py` into `.git/hooks/pre-commit` (a copy, so the hook also works on branches without the script; re-run after editing it). The hook checks the staged files and refuses the commit when any of them is a data or model dump (`.csv`, `.parquet`, `.cbm`, `.pt`, ...), lives under an artifacts directory, or contains a real customer identifier (the letters `POD` immediately followed by digits). Fabricated fixtures use identifiers such as `PODSYN001`, which start with `POD` as the organizer filter requires but are not matched. The check prints offending paths only, never their content. `python scripts/check_hygiene.py --all` audits the whole tracked tree and is meant for CI; `tests/test_hygiene.py` runs it too. The hand-in is an export of the frozen tree (`git archive <commit>`), never this history.

Run the full regression suite with `python -m pytest -q`; `pyproject.toml` points pytest at `tests/` and puts `src/` on the path, so no install is needed for the tests.

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

## Frozen matched evaluation

Use the [evaluation protocol](docs/evaluation_protocol.md) for the current matched comparison. Architecture settings are fixed; the arms are hourly-profile logistic regression, PatchTST from scratch, and the same PatchTST with masked pretraining. Thresholds use inner customer-OOF predictions; outer customer assignments genuinely change across repetitions. Persistence uses timestamps and observed coverage, not positions in subsampled arrays.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m neuralwave_experiment.evaluation \
  --canonical-dir data_private/patchtst_comparison_canonical \
  --out artifacts_private/evaluation_v1 \
  --config configs/evaluation.json \
  --data-kind development
```

Install the `experiment` extra first. This retains all split assignments, inner/outer predictions, calibration candidates, models, histories, and input/config identities. It reports presence and temporal diagnostics, not a capacity or blind-test score. The [benchmark status](benchmarks.md) distinguishes historical development numbers from the new protocol. Verification so far is an artificial-data smoke, not a new AIL benchmark.

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

## Task 1 window classifier on the shared comparison contract

[benchmarks.md](benchmarks.md) reports the PatchTST window classifier (aggregate metrics only). Every pipeline in the team is scored on identical customer folds and writes the same out-of-fold prediction file, so results stay comparable across branches without a package merge:

* **Split files.** `python -m neuralwave_data.prepare ... --folds 5 --seed S` writes `splits.parquet` (`customer_id`, `fold`). Seeds 0, 1 and 2 give three repeat files; a CSV with the same two columns is accepted too. Every labelled customer must appear in exactly one fold, otherwise the run stops with `split file ... does not cover labelled customers: N missing`. Extra customers are ignored.
* **Out-of-fold file.** One semicolon CSV over all repeats with the columns `customer_id;repeat;fold;prob;pv_installed;pv_installation_date;pv_capacity_kwp`, using the submission formatting (`TRUE`/`FALSE`, `YYYY-MM-DD`, two-decimal capacities, empty date and capacity when `FALSE`). It carries no labels, features or regimes. A detection-only pipeline, which this classifier is, leaves date and capacity empty and therefore scores `D = P = 0`; compare its F1 separately.

```bash
python -m neuralwave_experiment.task1_detector \
  --canonical-dir /private/artifacts/run \
  --splits /private/splits_seed0.parquet,/private/splits_seed1.parquet,/private/splits_seed2.parquet \
  --oof-out /private/oof_patchtst.csv \
  --out-dir /private/task1_results --epochs 8
python -m neuralwave_experiment.oof_contract --labels /private/artifacts/run/labels.parquet /private/oof_patchtst.csv [OTHER_OOF ...]
```

Each split file is one outer repeat; thresholds are tuned on the training customers of each fold only; checkpoints of the first repeat are written as `task1_model_fold_<k>.pt`. The scorer prints one aggregate line per file (`file=... repeats=k score=M±SD f1=... D=... P=...`) and a paired line for files that share repeats. Any design choice made from such a comparison is a Tier B decision and must be logged in the decision file of the main pipeline.

Scoring unlabelled customers as a plausibility check (no ground truth, so no accuracy is claimed):

```bash
python -m neuralwave_experiment.wild_inference \
  --unlabelled-csv /private/energy_consumption_unlabelled_dev.csv \
  --weights /private/task1_results/task1_model_fold_0.pt \
  --id-prefix POD --id-start 100 --id-count 20 \
  --out /private/task1_results/wild_batch.json
```

The terminal shows decision counts and score ranges; the per-customer table goes only to the private `--out` JSON (`--show-customers` prints it locally, never commit it).
