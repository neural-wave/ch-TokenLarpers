# AIL PV detection from consumption profiles

Detects photovoltaic installations, their commissioning date and nominal capacity (kWp) from 15-minute grid-withdrawal time series (AIL NeuralWave challenge). The authoritative plan is `IMPLEMENTATION_HANDOFF.md`.

## Rules

- No LLM in the training or inference pipeline; no weather, irradiation or location data; calendar features from the timestamps only.
- Inference runs fully offline.
- Agents develop against synthetic data (`pvdetect/synthetic.py`); real-data commands are run by the user.

## Setup

Python 3.10–3.12 (`requires-python >=3.10,<3.13`; developed on 3.12).

```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
```

or plain `pip install -r requirements.txt`. `requirements.txt` is compiled from `requirements.in` with exact pins (`uv pip compile requirements.in -o requirements.txt --python-version 3.12`).

## Train

```
python train.py --timeseries energy.csv --labels labels.csv --out artifacts/model.json [--cv] [--outer-repeats 4] [--report-dir DIR]
```

`--cv` runs nested customer-level cross-validation of the whole training procedure and prints aggregate lines only (`data:`, `nested:`, `dates:`, `regime_confusion:`, `selection:`); the final model is always fit on all labelled customers. `--report-dir` writes the repeat-0 out-of-fold table and selection records.

## Inference

```
python inference.py path/to/test.csv
```

Loads `artifacts/model.json` (plus sibling files) automatically and writes to the current directory:

- `predictions.csv`: `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`, one row per customer in the input, UTF-8, `;` separator, decimal point, `TRUE`/`FALSE`, `YYYY-MM-DD`, empty date/capacity when FALSE. Every TRUE row has a date and a positive capacity.
- `pv_extensions_predictions.csv`: `customer_id;extension_date;pv_capacity_after_extension_kwp` (header only until extensions are implemented).

The loader sniffs `;`/`,` separators and comma decimals, accepts epoch seconds or milliseconds, drops duplicate timestamps, and maps every reading to fixed UTC+1 so DST days (92/100 rows) become ordinary 96-slot days.

## Synthetic smoke run

```
python -m pvdetect.synthetic /tmp/pvsyn_train --n 24 --seed 1
python -m pvdetect.synthetic /tmp/pvsyn_test --n 12 --seed 2 --sep ";"
python train.py --timeseries /tmp/pvsyn_train/energy.csv --labels /tmp/pvsyn_train/labels.csv --out artifacts/model.json --cv
cd /tmp/pvsyn_test && python <repo>/inference.py energy.csv && PYTHONPATH=<repo> python -m pvdetect.scoring predictions.csv labels.csv
```

Remove `artifacts/` afterwards so no synthetic model is left where the real one goes.

## Validation policy

All data-driven choices happen inside the training procedure and are chosen by inner CV; outer CV estimates the whole procedure. Decisions taken after looking at outer results are logged in `DECISIONS.md` and the estimate is then labelled `selection-biased`. Evidence tiers: unit tests, synthetic regression tests (dev seeds), a held-out synthetic benchmark run only at phase gates (`benchmarks/LOG.md`), and real-data nested CV (primary evidence).

## Status

submit-v0: baseline (physical features, `ruptures` changepoint, logistic regression, ridge capacity); synthetic-validated only; extensions header-only; unlabelled data unused.
