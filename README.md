# AIL PV detection from consumption profiles

Detects photovoltaic installations, their commissioning date and nominal capacity (kWp) from 15-minute grid-withdrawal time series (AIL NeuralWave challenge). The authoritative plan is `IMPLEMENTATION_HANDOFF.md`.

## Team

- Alessandro Tognoni — data hygiene (check, scrubbing, commit hook), shared-fold comparability contract across branches, project overview and plan review.
- Jeffre Daniel Martinez Marroquin — the `pvdetect` pipeline (loading, scoring, season-aware detection, date regimes, inner-CV selection, nested CV, synthetic generator, benchmarks) and the implementation plan.
- Tarun Deepak — canonical preprocessing pipeline, synthetic PV data pipeline, time-series embeddings and the PatchTST window classifier, hackathon guidelines.

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

It copies `scripts/check_hygiene.py` into `.git/hooks/pre-commit` (a copy, so the hook also works on branches without the script; re-run after editing it). The hook checks the staged files and refuses the commit when any of them is a data or model dump (`.csv`, `.parquet`, `.cbm`, `.pt`, ...), lives under an artifacts directory, or contains a real customer identifier (the letters `POD` in any case, starting a word and immediately followed by digits, even when letters or `_` follow the digits). Fabricated fixtures use identifiers such as `PODSYN001`, which are not matched. The check prints offending paths only, never their content. `python scripts/check_hygiene.py --all` audits the whole tracked tree (for CI); `tests/test_hygiene.py` runs it too. The hand-in is an export of the frozen tree (`git archive <commit>`), never this history.

History and visibility: commits `e303f6d`, `1f875b6`, `a65b552` and `086eda8` on the `time-series-embeddings` branch contain real customer identifiers. Until the team decides how to remove them (delete that branch once its content is carried over, or a coordinated `git filter-repo` rewrite), `ch-TokenLarpers` stays private, and a repo admin confirms the visibility setting. The NW_26 hand-in is always a clean export (Step 2.5-G) that does not carry this history.

## Train

```
python train.py --timeseries energy.csv --labels labels.csv --out <private>/model.json [--cv] [--outer-repeats 4] [--report-dir DIR] [--folds F0,F1,...] [--oof-out PATH]
```

`--cv` runs nested customer-level cross-validation of the whole training procedure and prints aggregate lines only: `data:` (including whether readings are signed, which switches export features on), `nested:` (score ± SD over repeats, bootstrap 90% interval, F1/D/P, `inner=`/`gap=` for the inner-vs-outer score gap, and `outer_informed_decisions=k`), `dates:` (date accuracy per regime), `regime_confusion:`, `selection:` (how often each inner-CV choice was made). The final model is always fit on all labelled customers. `--report-dir` writes the repeat-0 out-of-fold table (`oof_repeat0.csv`) and selection records. A model trained on real data goes to a private path outside this repository; it reaches the NW_26 repository only through the export below (`## Submission export`).

Inside each fit, inner CV on the training customers chooses: logistic regression vs CatBoost (CatBoost needs ≥2.0 points and a stable bootstrap), the decision threshold, ridge vs CatBoost capacity, a capacity factor κ, and the date rule for weak-evidence customers. The week-of-year climatology (from labelled non-PV training customers) deseasonalizes the changepoint signal; candidates whose step recurs at the same time in other years are rejected.

`inner=` is the mean inner-CV score of the final selected configuration over the outer fits, and `gap=` is `inner` minus the nested score. After the capacity model, κ and the date rule are chosen, the decision threshold is rescored once on those final inner out-of-fold outputs (same grid, same tie-breaking); `threshold_initial`/`inner_score_initial` in the selection records keep the pre-rescoring values. The gap is a warning signal, not a measure of overfitting: inner and outer scores come from different customer sets and training sizes. A gap above about 5 points calls for a look at selection stability (the `selection:` line); it proves nothing alone.

Dates come from one of three regimes: `event` (detected changepoint + learned offset), `present_at_start` (PV already present when data starts: first date − learned lag), `insufficient` (weak candidate or first date, chosen by inner CV).

### Shared-fold comparison

Pipelines on different branches are compared through one split file set, one prediction contract and one scorer, not a package merge. `--folds F0,F1,...` replaces the internal repeated stratified folds: each split file (`splits.parquet` from `neuralwave_data.prepare`, or a `;`/`,` CSV with `customer_id,fold`) is one outer repeat, in the given order, and must assign every labelled customer to exactly one integer fold (extra IDs are ignored; errors report counts only). `--oof-out PATH` writes the shared out-of-fold file `customer_id;repeat;fold;prob;pv_installed;pv_installation_date;pv_capacity_kwp` (`repeat` = position of the split file, `fold` = its fold value), with no labels, features or regimes. The canonical comparison scorer is `python -m neuralwave_experiment.oof_contract` on the `time-series-embeddings` branch; `pvdetect.scoring` stays pvdetect's internal scorer.

Matched run (data holder, private directory only):

```
python -m neuralwave_data.prepare <manifest> --out <private>/prep_s{0,1,2} --folds 5 --seed {0,1,2}
python train.py --timeseries <dev energy> --labels <dev labels> --out <private>/pvdetect/model.json --cv \
  --folds <private>/prep_s0/splits.parquet,<private>/prep_s1/splits.parquet,<private>/prep_s2/splits.parquet \
  --oof-out <private>/pvdetect_oof.csv
python -m neuralwave_experiment.task1_detector ... --splits <same three files> --oof-out <private>/patchtst_oof.csv
python -m neuralwave_experiment.oof_contract --labels <dev labels> <private>/pvdetect_oof.csv <private>/patchtst_oof.csv
```

Only the printed aggregate lines leave the private directory. PatchTST writes empty date and capacity (D = P = 0), so compare F1 separately from the full score. The comparison is a Tier B decision and gets a `DECISIONS.md` line.

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

## Submission export

The trained model never enters this repository's history: the hygiene hook blocks `artifacts/` and every `.cbm`. It reaches the NW_26 repository only through the export script below or through the organizer-approved private channel (files placed into `artifacts/` next to `inference.py`, which loads them automatically).

```
sh scripts/export_submission.sh COMMIT DEST [MODEL_DIR]
python scripts/check_release.py DEST --allow artifacts/model.json [--allow artifacts/catboost_clf.cbm ...]
```

`export_submission.sh` extracts `git archive COMMIT` into `DEST` (clean tree, no history). With `MODEL_DIR` it copies `MODEL_DIR/model.json` and exactly the files that model references into `DEST/artifacts/`; without it (private-channel route) no model file is copied. The listing (`check_release.model_files`, mirroring `pvdetect.model.referenced_files`) and the audit use the standard library only, so any `python3` works; set `PYTHON` to choose another interpreter. It then runs `check_release.py` with an allow list generated from `model.json`, never a wildcard, and aborts non-zero on any failure; an aborted `DEST` must not be committed.

`check_release.py` walks every file under `DEST` on the filesystem (the NW_26 repository has no hook; `.git` directories are skipped) and exits 1 when a file not on the allow list is a data or model dump, when an allowed path is missing, or when any file, allowed model files included, contains a real customer identifier. It prints paths and reasons only. Passing it is necessary, not sufficient: the export also needs human review and organizer approval (Step 2.5-G).

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

## Organizer clarifications

Questions for the organizers (Step 2.5-H, sent by the team); answers get recorded here.

1. May semi-synthetic augmentation of the provided series be used in the competitive model? *Awaiting answer.*
2. May the trained model files (JSON parameters with aggregate statistics over the 40 development customers, and CatBoost model files) be published in the open-sourced NW_26 repo? *Awaiting answer.*
3. If not, which private channel should deliver them (e.g. direct upload or Swisstransfer), with the organizers placing them into `artifacts/` next to `inference.py`? *Awaiting answer.*

Until answer 2 arrives we plan for the private channel. `inference.py` loads `artifacts/model.json` automatically, so that route needs no code change.

## Status

Procedure: season-aware changepoint and confounder features, three date regimes, inner-CV selection of classifier, threshold, capacity model, κ and date rule, scored by nested CV; `outer_informed_decisions=1` (selection-biased). Synthetic-validated only; real-data nested CV pending (P3); extensions header-only; unlabelled data unused.
