# Model Card — AIL PV Detection (NeuralWave challenge)

## Intended Use

Detects photovoltaic (PV) installations from 15-minute grid-withdrawal time series
(energy_consumption_kwh) recorded by smart metering systems. For each customer the
model predicts whether PV is installed, the commissioning date (±30 days matters to
the score), and the nominal capacity (kWp). Intended for the AIL NeuralWave challenge
evaluation only. Not validated for deployment outside that context.

**Not intended for:** individual energy auditing, real-time grid management, or any
application requiring per-customer outputs to be shared (see Privacy Notes below).

## Data

**Training source:** 40 labelled development customers provided by the challenge
organizers (AIL NeuralWave). The labelled period spans 2022-01-01 to 2025-12-31 with
varying per-customer coverage. Some customers have shorter histories.

**Signal:** 15-minute interval grid-withdrawal energy in kWh. Negative values (export)
are retained when a dataset-wide signed-data test confirms at least 10% of customers
have signed readings.

**Features derived from:** timestamps (UTC+1 clock windows) only. No weather, solar
irradiation, GPS location, or external data of any kind.

**No per-customer data is published.** The trained model stores aggregate statistics
only (week-of-year climatology curves, median offsets, capacity band quantiles). An
organizer-approved human review of the exact model files precedes any publication.

## Training Procedure

### Nested cross-validation

All data-driven choices live inside a single `Procedure.fit` call and are selected by
inner cross-validation on that fit's training customers. The outer nested-CV score
(4 repeats × 5 folds) is an unbiased estimate of the *whole procedure*, including its
option pool. Any human design decision made after inspecting outer results is logged in
`DECISIONS.md` and causes the estimate to be labelled **selection-biased** (`outer_informed_decisions=k`).

### Inner-CV Tier A option pool

The following choices are made automatically inside each `fit` by inner CV on the
training customers only:

| Option | Candidates |
|---|---|
| Classifier | Logistic regression (C = 0.5, balanced) · CatBoost (depth 3, balanced) |
| Threshold | Grid 0.20–0.80, step 0.05 |
| Capacity model | Ridge regression · CatBoost regressor |
| Capacity factor κ | Grid 0.80–1.10, step 0.05 |
| Date rule (weak evidence) | `weak_candidate` · `first_date` |
| Extra feature blocks | `recent` (late-install detection) · `smallpv` (small-PV detection) |
| Output correction | `date_debias` (per-regime median date error) · `cap_ceiling` (capacity band) |

A candidate replaces the current selection only if its inner score is at least 2.0
points higher **and** a paired inner bootstrap (B = 500) gives P(diff > 0) ≥ 0.8.
The bootstrap is a stability heuristic on inner predictions; it is never computed on
outer predictions and never cited as validation evidence.

### Season-aware changepoint detection

The midday/night log-ratio `s` is deseasonalized using a week-of-year climatology
(`ds = s − clim.s[woy]`) derived from labelled non-PV training customers. A Binseg
changepoint is refined and filtered by persistence (fraction of later months below the
pre-changepoint level), year-over-year recurrence (candidates that recur annually are
discarded as seasonal confounders), and a minimum evidence threshold.

### Date regimes

The commissioning date is assigned by one of three regimes chosen per customer:

| Regime | Date rule |
|---|---|
| `event` | Refined changepoint + learned median offset (clipped to ±120 days) |
| `present_at_start` | First date in data − learned start lag |
| `insufficient` | Weak candidate or first date (chosen by inner CV) |

### Final model

After nested CV reports the estimate, a final `Procedure.fit` is run on all 40
labelled customers. Its Tier A selections are independent of the outer evaluation.

### Lock-box protocol

Evaluation protocol v2 (`docs/evaluation_lockbox_plan.md`): nested CV on 30
development customers, one sealed lock-box of 10 opened once after `freeze`.
Pre-registered in `configs/lockbox.json` (seed 2026, at most 3 comparisons). Results:
**<filled by the data holder>**. Whether the lock-box score falls inside, below, or
above the nested `ci90` is stated in the model's `lockbox_result.json`.

## Evaluation Protocol

| Metric | What it measures |
|---|---|
| `score = F1 × (70 + 15D + 15P)` | Overall: detection + date accuracy + capacity accuracy |
| `F1` | PV detection (binary, pooled across all customers) |
| `D = mean max(0, 1 − |days|/30)` | Date accuracy over true positives |
| `P = mean max(0, 1 − APE/0.5)` | Capacity accuracy over true positives |
| `ci90` | 90% customer-bootstrap interval of the mean score across CV repeats (1000 draws, `default_rng(0)`); describes sampling uncertainty, not model-selection uncertainty |

### What each number means

- **Nested score ± SD**: mean and standard deviation of `main_score` across the outer
  repeats. When `outer_informed_decisions > 0`, the estimate is labelled
  **selection-biased**.
- **`ci90`**: resamples the fixed out-of-fold predictions 1000 times (bootstrap), not
  new data. It does not account for model-selection uncertainty. With 40 customers the
  interval is wide (~10–20 points) and should be interpreted as a rough guide only.
- **Lock-box result**: a single look at 10 held-out customers after the model is
  frozen. With 10 customers the interval is very wide. A result below the nested
  interval suggests possible overfitting but does not by itself justify re-tuning.

## Limitations

- **Small labelled set:** only 40 labelled customers. Each cross-validation repeat
  trains on ~32 customers. The hidden test set has 20 customers.
- **Selection-biased estimate:** design decisions informed by real-data outer results
  are logged (`DECISIONS.md`). When `k > 0`, the nested score is labelled
  `selection-biased`.
- **Capacity is not uniquely identifiable** from grid-withdrawal data alone: battery
  storage, export clipping, building load profile, and panel orientation all confound
  the midday dip that drives the capacity estimate.
- **Synthetic benchmarks are not accuracy evidence.** Generator variant A and B
  synthetic runs (Tier R and G) verify algorithmic contracts and detect regressions.
  They do not measure performance on the challenge data.
- **Weekly windows are not independent labelled cases.** The bootstrap CI resamples
  customers, not weeks. It does not account for within-customer correlation.
- **No weather or location data.** The calendar-only feature set cannot distinguish a
  cloud-covered PV customer from a non-PV customer during a cloudy period.
- **Extensions remain header-only.** The extension predictor is not yet implemented;
  `pv_extensions_predictions.csv` contains only the header.

## Ethical and Privacy Notes

- **No per-customer outputs are published.** Review plots, out-of-fold predictions,
  and per-customer error tables remain with the data holder in private directories.
- **Aggregate results** (nested score, F1, D, P, selection counts) may be shared only
  with team members, not with coding assistants, per organizer answer 6.
- The model stores only aggregate statistics (climatology curves, quantile band,
  median offsets) derived from the 40 training customers. No individual customer
  reading or identifier is present in the model file (verified by `check_release.py`
  before every export).
- The repository remains private until the team decides how to handle prior commits
  that contained real identifiers on a now-unused branch.

## Reproducibility

```bash
# Nested CV on the 40 development customers (data holder, private):
python train.py --timeseries <private>/energy.csv --labels <private>/labels.csv \
  --out <private>/model.json --cv --folds <splits> --oof-out <private>/oof.csv

# Export rehearsal on synthetic data (any environment, no real data):
python -m pvdetect.synthetic /tmp/pvsyn20 --n 20 --seed 9
python train.py --timeseries /tmp/pvsyn20/energy.csv --labels /tmp/pvsyn20/labels.csv \
  --out /tmp/syn_model/model.json
sh scripts/export_submission.sh HEAD /tmp/syn_export /tmp/syn_model
python inference.py /tmp/pvsyn20/energy.csv   # from /tmp/syn_export

# Nested CV score (synthetic, for CI only — not accuracy evidence):
# outer_informed_decisions=<filled by the data holder>
# nested: score=<filled by the data holder>
```

Real-data nested CV score: **<filled by the data holder>**
Lock-box score: **<filled by the data holder>**
Python version used: **<filled by the data holder>**
