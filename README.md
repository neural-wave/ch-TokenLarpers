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

## Data hygiene

The repository is open-sourced after the event, so nothing derived from the AIL data may enter git history: raw files and every data-derived artifact stay in private, git-ignored paths, and reports hold aggregate metrics only (no customer identifiers, per-customer values, probabilities or label dates).

Install the commit-time check once per clone:

```
sh scripts/install_hooks.sh
```

It copies `scripts/check_hygiene.py` into `.git/hooks/pre-commit` (a copy, so the hook also works on branches without the script; re-run after editing it). The hook checks the staged files and refuses the commit when any of them is a data or model dump (`.csv`, `.parquet`, `.cbm`, `.pt`, ...), lives under an artifacts directory, or contains a real customer identifier (the letters `POD` immediately followed by digits). Fabricated fixtures use identifiers such as `PODSYN001`, which are not matched. The check prints offending paths only, never their content. `python scripts/check_hygiene.py --all` audits the whole tracked tree (for CI); `tests/test_hygiene.py` runs it too. The hand-in is an export of the frozen tree (`git archive <commit>`), never this history.

## Train

```
python train.py --timeseries energy.csv --labels labels.csv --out artifacts/model.json [--cv] [--outer-repeats 4] [--report-dir DIR]
```

`--cv` runs nested customer-level cross-validation of the whole training procedure and prints aggregate lines only: `data:` (including whether readings are signed, which switches export features on), `nested:` (score ± SD over repeats, bootstrap 90% interval, F1/D/P, and `outer_informed_decisions=k`), `dates:` (date accuracy per regime), `regime_confusion:`, `selection:` (how often each inner-CV choice was made). The final model is always fit on all labelled customers. `--report-dir` writes the repeat-0 out-of-fold table (`oof_repeat0.csv`) and selection records.

Inside each fit, inner CV on the training customers chooses: logistic regression vs CatBoost (CatBoost needs ≥2.0 points and a stable bootstrap), the decision threshold, ridge vs CatBoost capacity, a capacity factor κ, and the date rule for weak-evidence customers. The week-of-year climatology (from labelled non-PV training customers) deseasonalizes the changepoint signal; candidates whose step recurs at the same time in other years are rejected.

Dates come from one of three regimes: `event` (detected changepoint + learned offset), `present_at_start` (PV already present when data starts: first date − learned lag), `insufficient` (weak candidate or first date, chosen by inner CV).

### Review plots (dev requirement: matplotlib)

```
python -m pvdetect.review REPORT_DIR/oof_repeat0.csv --timeseries energy.csv --out-dir PLOTS_DIR
```

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

All data-driven choices happen inside the training procedure and are chosen by inner CV; outer CV estimates the whole procedure. Decisions taken after looking at outer results are logged in `DECISIONS.md` and the estimate is then labelled `selection-biased`. Evidence tiers: unit tests, synthetic regression tests (dev seeds), a held-out synthetic benchmark run only at phase gates, and real-data nested CV (primary evidence).

### Synthetic benchmarks

```
python -m pvdetect.benchmark --pilot                         # times a small run, projects the full one
python -m pvdetect.benchmark --suite dev [--scenarios a,b] [--n 40] [--repeats 2]
python -m pvdetect.benchmark --suite holdout --log benchmarks/LOG.md   # phase gates only
```

Every full run starts with the pilot and is refused if the projection exceeds `--budget-min` (default 30) or memory looks too high, unless `--force`. The holdout suite uses generator variant B (different PV, cloud and load models plus confounders absent from the dev generator) and seeds from `benchmarks/HOLDOUT_SEED_BLOCK`; a code change made in response to a holdout result burns the block.

## Status

submit-v1: season-aware detection, date regimes, confounder features, inner-CV selection (logistic regression vs CatBoost, ridge vs CatBoost capacity, threshold, κ); synthetic-validated only; extensions header-only; unlabelled data unused.
