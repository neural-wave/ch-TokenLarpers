# AIL PV detection from consumption profiles

Detects photovoltaic installations, commissioning dates and nominal capacity (kWp) from 15-minute grid-withdrawal time series (AIL NeuralWave challenge). `IMPLEMENTATION_HANDOFF.md` describes the original baseline; `goals/hybrid-pv-integration/plan.md` records the approved integration.

The repository also retains Pipeline A, strict conversion of authorized CSV manifests into canonical Parquet, and Pipeline B, deterministic paired synthetic PV scenarios. Use fabricated fixtures for agent-run development checks.

For a readable, step-by-step explanation of the implemented pipeline, open [`docs/PIPELINE_EXPLAINED.html`](docs/PIPELINE_EXPLAINED.html) in a browser. The model card (intended use, data, training procedure, evaluation protocol, limitations, privacy notes, reproducibility commands) is in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Team

- Alessandro Tognoni — data hygiene (check, scrubbing, commit hook), shared-fold comparability contract across branches, project overview and plan review.
- Jeffre Daniel Martinez Marroquin — the `pvdetect` pipeline (loading, scoring, season-aware detection, date regimes, inner-CV selection, nested CV, synthetic generator, benchmarks) and the implementation plan.
- Tarun Deepak — canonical preprocessing pipeline, synthetic PV data pipeline, time-series embeddings and the PatchTST window classifier, hackathon guidelines.

## Rules

- No LLM in the training or inference pipeline; no weather, irradiation or location data; calendar features from the timestamps only.
- Inference runs fully offline.
- Agents develop against synthetic data (`pvdetect/synthetic.py`); real-data commands are run by the user.

## Setup

Both pretrained encoder sizes are included under `pretrained/`. For clone-contained offline setup and matched baseline/hybrid experiments, see [`docs/hybrid_team_setup.md`](docs/hybrid_team_setup.md). After installing dependencies, run `python scripts/check_pretrained.py --models small medium` to verify both.


Python 3.11–3.12 (`requires-python >=3.11,<3.13`; verified on 3.12).

```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
```

or plain `pip install -r requirements.txt`. `requirements.txt` is compiled from `requirements.in` with exact pins (`uv pip compile requirements.in -o requirements.txt --python-version 3.12`).

PyTorch is included in the standard requirements for selected CNN inference.
The `experiment` extra remains accepted by existing editable-install commands.
The physical-only path still avoids importing Torch.

The CNN-presence handoff corrects the KDE branch to use normalized hourly
profiles, matching the CNN branch instead of evaluating raw kWh against a
residential-scale grid. Refit CNN weights before evaluating this correction.
Date and capacity estimators are unchanged. Fabricated-data checks establish
software behavior, not real-world accuracy or absence of overfitting.

## Data hygiene

Raw AIL inputs, private metrics, customer identifiers, optimizer checkpoints, and unapproved derived artifacts must not enter Git history. The user explicitly approved the small/medium pretrained encoder pairs and the CNN-only presence package under `pretrained/pv_presence/`; the hygiene hook permits only the exact checksum-pinned files in `scripts/check_hygiene.py`. No general model/data-file exemption is enabled. Evaluation files, logs and customer records are not included.

Install the commit-time check once per clone:

```
sh scripts/install_hooks.sh
```

It copies `scripts/check_hygiene.py` into `.git/hooks/pre-commit` (a copy, so the hook also works on branches without the script; re-run after editing it). The hook checks the staged files and refuses the commit when any of them is a data or model dump (`.csv`, `.parquet`, `.cbm`, `.pt`, ...), lives under an artifacts directory, or contains a real customer identifier (the letters `POD` in any case, starting a word and immediately followed by digits, even when letters or `_` follow the digits). Fabricated fixtures use identifiers such as `PODSYN001`, which are not matched. The check prints offending paths only, never their content. `python scripts/check_hygiene.py --all` audits the whole tracked tree (for CI); `tests/test_hygiene.py` runs it too. The hand-in is an export of the frozen tree (`git archive <commit>`), never this history.

History and visibility: commits `e303f6d`, `1f875b6`, `a65b552` and `086eda8` on the `time-series-embeddings` branch contain real customer identifiers. Until the team decides how to remove them (delete that branch once its content is carried over, or a coordinated `git filter-repo` rewrite), `ch-TokenLarpers` stays private, and a repo admin confirms the visibility setting. The NW_26 hand-in is always a clean export (Step 2.5-G) that does not carry this history.

## Train

```
python train.py --timeseries energy.csv --labels labels.csv --out <private>/model.json [--cv] [--outer-repeats 4] [--report-dir DIR] [--folds F0,F1,...] [--oof-out PATH] [--unlabelled <private>/unlabelled.csv] [--split-dir <private>/split [--split-frac 0.75] [--split-seed 0]]
```

`--cv` runs nested customer-level cross-validation of the whole training procedure and prints aggregate lines only: `data:` (including whether readings are signed, which switches export features on), `nested:` (score ± SD over repeats, `ci90`: a 90% customer-bootstrap interval of the score averaged over all repeats — each of 1000 draws resamples the customers once and averages every repeat's score on them, F1/D/P, `inner=`/`gap=` for the inner-vs-outer score gap, and `outer_informed_decisions=k`), `points_lost:` (the exact split of `100 − score` into detection `100·(1−F1)`, dates `15·F1·(1−D)` and capacity `15·F1·(1−P)`, with the largest named; it picks the Step 2.5-E branch), `dates:` (date accuracy per regime), `regime_confusion:`, `selection:` (how often each inner-CV choice was made). These lines are AIL-derived: they stay with the team and never go into an agent session (organizer answer 6). The final model is always fit on all labelled customers. `--report-dir` writes the repeat-0 out-of-fold table (`oof_repeat0.csv`) and selection records. A model trained on real data goes to a private path outside this repository; it reaches the NW_26 repository only through the export below (`## Submission export`).

Inside each fit, inner CV on the training customers chooses: logistic regression vs CatBoost (CatBoost needs ≥2.0 points and a stable bootstrap), then the average of both (`avg`, Step 2.5-E2, same margin rule against that winner), the decision threshold, ridge vs CatBoost capacity, a capacity factor κ, the date rule for weak-evidence customers, and whether to add the persistence feature block (E1: `summer_suppressed_week_frac`, the share of summer weeks in the last year whose midday/night ratio stays clearly below the winter level, which separates PV from temporary drops such as holidays; same margin rule). The week-of-year climatology (from labelled non-PV training customers) deseasonalizes the changepoint signal; candidates whose step recurs at the same time in other years are rejected.

`--unlabelled PATH` (optional, Steps 17–18) streams the unlabelled series in typed chunks (`--unlabelled-chunksize`, default 500000) into per-day sums with exact keep-first deduplication; it drops null ids, ids that don't start with `--unlabelled-id-prefix` (default `POD`) and every customer present in `--timeseries`, and prints one aggregate `unlabelled:` line. From these label-free daily frames it builds a fold-independent prior: a week-of-year climatology from the unlabelled customers with `seasonal_contrast` at or above the median (`unlabelled_0.5`), and the spread of behaviour across households per week. Two Tier A options are then scored on each fit's common inner folds: climatology `labelled_nonpv` vs `unlabelled_0.5` (only when the filter-sensitivity test passes: the climatologies for q = 0, 0.25, 0.5, 0.75 differ by at most 0.05 log units in every week; otherwise disabled with the reason printed), then changepoint evidence scaled by the weekly spread or not. An option replaces the baseline only with ≥2.0 inner points and a paired bootstrap ≥0.8; the `selection:` line counts the choices. Unlabelled customers are never treated as PV-free or as negatives. `clim_unlabelled.json` (aggregate curves and the sensitivity numbers) is written next to `--out` and is not shipped by the export. Without `--unlabelled`, training is byte-identical to before. `python -m pvdetect.unlabelled_audit --unlabelled P --timeseries DEV` prints dev-vs-unlabelled KS statistics and peak memory (synthetic gate: 100 customers × 4 years, 13.9M rows, 866 MB peak, 10 s).

Pseudo-labelling (organizer answer 4): with `--unlabelled`, `--pseudo-pool N` (default 300, `0` disables) streams the file a second time to rebuild full day matrices (float32) for at most N unlabelled customers with ≥365 valid days, chosen by a salted hash of the id (independent of file order). As the last Tier A option of each fit, on the same inner splits, the already chosen classifier scores the pool; customers with probability ≥0.95 or ≤0.05 become pseudo-positives/negatives (each class capped at 3× the labelled count, weight 0.5) and only the classifier is refit; dates, capacity and climatology never see pseudo-labels, and unconfident customers are left out, never treated as negatives. Adopted only with the margin rule; one round. The `selection:` line shows `pseudo off=k/n on=k/n`.

Option switches for single-change experiments: `--no-e1e2` disables the E1/E2 options (training is then byte-identical to commit `fbab11e`); `--schedule` adds the `solar_shape_consistency` feature-block option (computed at the changepoint cut from June–August days before vs ≥14 days after, weekdays and weekends separately with ≥20 days each, value = the smaller of the two midday-trough deepenings, NaN otherwise; adopted only with the margin rule; off by default). `scripts/train_and_launch.sh` passes `TRAIN_FLAGS` through, e.g. `TRAIN_FLAGS="--no-e1e2 --schedule"`.

`--sun` adds the `sun_coupling` feature-block option (off by default; adopted only with the margin rule). PV is the only load-side effect that lowers midday consumption more on sunny days. Inside every fit, each labelled PV training customer from 14 days after its installation date contributes its daily midday anomaly (`mid` minus a centred rolling median within its own day type, 15 days for weekdays and 29 for weekends, scaled by 1.4826·MAD; label-free and cached); per date, the mean over ≥5 contributors after dropping the lowest and the highest is the fit's sunshine index, so every value mixes ≥3 customers. It is derived only from the provided consumption data, never from weather, irradiation or location data. A customer's `sun_coupling` is the Spearman correlation of its own anomaly with that index over the April–September days of its last 365 days (≥40 shared days, else NaN): PV clearly positive, weather-independent seasonal or schedule dips near 0, sun-driven loads such as cooling negative. A contributor's training value uses the index without itself; the index is refitted in every inner and outer fit from that fit's training labels only, and an adopted index is saved as `climatology.sun` (one value per date, no per-customer copies). The dev generator gains `shared_weather` (one dataset-wide daily sunshine series drives PV output and AC load; non-PV hard negatives become sun-driven AC or a recurring summer weekday daytime absence); the holdout suite is unchanged.

**Recall-targeted options (`recent`, `smallpv`; in the default pool, `--no-recent` / `--no-smallpv` remove them).** The frozen protocol run's out-of-fold diagnosis names missed PV (recall) as the largest detection loss (the failure type only; no numbers). Mechanisms by which a PV customer gets a low probability: little post-install data (installation in the last weeks, so the 365-day and persistence features barely see it), a shallow midday dip (small PV relative to load), a median day clouded over the dip, a threshold chosen high, and rare positives for the classifiers. Two label-free feature blocks target the first two, each adopted per fit only by the usual inner-CV margin rule (≥2.0 inner points, bootstrap ≥0.8, the same `inner_splits`): `recent` adds `recent_matched_step` (median `s` of the same ISO weeks ≥300 days before the tail minus the tail's, the tail being the last 84 valid days), `recent_prior_step` (the 84 days before the tail minus the tail) and `recent_zero_frac` (share of near-zero midday slots in the tail); `smallpv` adds `deficit_contrast` and `deficit_q10_contrast` (summer minus winter of `log(mid/mean(night, eve))` over the last 365 days, median and 10th percentile; scale-free). Windows that are too short give NaN. Class weights are already balanced for logreg and CatBoost and the threshold is chosen by inner CV, so no extra weighting option was added. Dev scenarios `late_install` (installation 8–16 weeks before the end) and `small_pv` (residential 0.8–2.0 kWp) reproduce the first two mechanisms (variant A only). On small synthetic nested runs (32 customers, three dev seeds) the baseline misses 0/48 late installs and 4/48 small-PV customers; the options leave these counts unchanged, so the synthetic data show no recall gain, only that the `recent` features separate late installs (AUC ≥ 0.9) where the 365-day seasonal contrast does not. Whether they help on the real data is for the data holder's run to show.

Weekly-evidence head (Step 2.5-E, on by default): each customer's history becomes clock-aligned ISO weeks with ≥5 valid days (missing readings masked, never zeros), with 11 per-week features (deseasonalized midday/night ratio and its change against the customer's own median, midday zero share and noon zero-run length, evening contrast, night change as a confounder channel, within-week midday variability, log amplitude, week-of-year sin/cos, coverage). Weeks of PV customers ending ≥7 days before the labelled date are negative, weeks starting ≥7 days after are positive, the band between is dropped; each customer's weeks share weight 1. A logistic head with C = 0.1 is trained inside each fit on the training customers only, and a customer's score is the maximum of 4-week rolling medians of its week probabilities (late installs and PV present from the start both score high; isolated short drops do not). Inner CV then offers `weekly` and `avg_weekly` (mean with logreg) as classifier candidates, and logreg C ∈ {0.1, 0.5, 2.0} (`ctune`), each adopted only with the margin rule. `--no-weekly` and `--no-ctune` switch them off; the exact pre-E1/E2 baseline of a single-change experiment is therefore `--no-e1e2 --no-weekly --no-ctune`.

Fraunhofer-style CNN+MLP candidate (opt-in): install `pip install -e ".[experiment]"`, then add `--fraunhofer` to `train.py`. It combines a 1D CNN on 168-hour load windows with an MLP on a 100-bin Gaussian KDE of hourly consumption, training the two branches separately before a frozen-branch combined head. Each fit evaluates the candidate on the **same customer-level inner folds** as the other options; no customer's windows cross a fold boundary. CNN selection uses the existing ≥2-point margin and inner bootstrap ≥0.8 rule, and its decision threshold is selected from inner out-of-fold customer probabilities, not fixed. The baseline remains eligible; `--fraunhofer` does not force CNN adoption and requires inner CV.

With `--fraunhofer`, the classifier kind `avg_cnn` is one more inner candidate: the mean of the CNN customer score and the classifier chosen before the CNN step (logreg, catboost, avg, weekly or avg_weekly), with its own inner threshold. It is scored on the same inner OOF frame and adopted only by the same ≥2-point margin and bootstrap ≥0.8 rule against the best of the other kinds. The model then stores both components (`fraunhofer.pt` plus the base classifier's logreg/CatBoost/weekly parts) with `classifier.kind = "avg_cnn"` and `classifier.base_kind`; inference computes the same average. Older models, including `fraunhofer`-kind ones, predict exactly as before. The `selection:` line counts `fraunhofer` and `avg_cnn` fits.

Windows need ≥80% complete hours; installation-straddling weeks and undated positive customers are excluded from window training. Customer evidence is the strongest run of **four consecutive calendar weeks**, scored by its weakest weekly probability. Missing/rejected weeks break a run; fewer than four consecutive usable weeks score 0. Earlier non-PV history no longer dilutes a recent supported run. The four-week duration is fixed, not tuned on evaluation results; the existing inner-CV customer threshold remains fitted on training customers only. This trades sensitivity to shorter installations for resistance to isolated high-score weeks, and does not eliminate persistent confounders.

All three training stages weight each window by `N / (C * n_customer)`, giving each customer equal total base weight regardless of history length. Positive-class balancing uses the same customer-weighted class mass. Only labelled training customers train the CNN; unlabelled customers are not treated as negatives. Refit CNN weights **and** reselect the inner-CV threshold after these changes; previously fitted packages do not validate the new behavior. Date/capacity estimators are unchanged.

When selected, the model saves `fraunhofer.pt` beside `model.json`; normal offline inference loads it, and `scripts/export_submission.sh` includes it through the model's artifact references. Training and inference select MPS when available, otherwise CPU. Fabricated checks prove software behavior, not predictive accuracy. Except for the explicitly approved checksum-pinned CNN presence package below, real trained weights stay outside this repository and use the private submission export.

### Reuse the fitted CNN without retraining

`pretrained/pv_presence/model.json` and `fraunhofer.pt` contain the approved
trained presence classifier and its fitted customer threshold. A normal clone
includes both files. This is **not** a complete date/capacity inference model
and must not be passed directly to `gui.py` or `inference.py`.

```python
from pvdetect.model import load_model
from pvdetect.fraunhofer import fraunhofer_customer_scores

presence = load_model("pretrained/pv_presence/model.json")
# table is built with build_feature_table(load_timeseries(path)).
scores = fraunhofer_customer_scores(
    presence["_objects"]["fraunhofer"], table, table.ids
)
pv_present = scores >= presence["classifier"]["threshold"]
```

To attach this classifier to an existing fitted pipeline without changing its
date/capacity estimators, replace only its `classifier` dictionary with
`presence["classifier"].copy()` and set
`model["_objects"]["fraunhofer"] = presence["_objects"]["fraunhofer"]`.
Keep its other fitted components, then persist to a new folder with
`save_model(model, path)`; this packages the CNN weights with that model.
This recombination is a new pipeline configuration: the original evaluated
score does not transfer automatically. Do not use this supervised checkpoint
to produce held-out evidence for customers included in its training population.
Nested CV must still train a separate CNN inside each training fold.

The package intentionally excludes labels, customer identifiers, OOF
predictions, evaluation scores, diagnosis, logs, optimizer state and fitted
date/capacity parameters. Its manifest records the inference contract,
training code revision and weight checksum.


Dates and capacity (on by default): κ is re-chosen on the true positives at the final (rescored) threshold (`kappa_initial` keeps the first pass). Option `offset_tri` replaces the shrunk-median event offset and start lag by the integer offset that maximises the summed date score Σ max(0, 1 − |e − x|/30) (the scoring triangle), and lets one or two present-at-start training customers move the start lag partially (× n/(n+5)) instead of leaving it at 0. Option `recur_strict` requires persistence ≥ 0.75 before a changepoint counts as found when no ±1–3-year recurrence window fits inside the series (short histories), stored in the model so inference detects the same way. Both adopted only with the margin rule; `--no-offset-tri` and `--no-recur-strict` switch them off, so the full single-change baseline is `--no-e1e2 --no-weekly --no-ctune --no-offset-tri --no-recur-strict`. The dev generator gains `label_lag_60` (labelled date 30–90 days after the visible onset) and `meter_spikes` (isolated single-slot spikes); the holdout suite is unchanged.

Options `onset_first` and `tail_guard` target spring-ramp ambiguity and end-of-series noise in onset dating. `onset_first`: recurrence is computed using only earlier-year windows (k < 0), falling back to all windows when none exist; among qualifying candidates (persistent, evidence ≥ threshold), the earliest whose evidence ≥ 0.3 × best evidence is chosen. `tail_guard`: candidates require ≥ 45 valid days after τ, and `_persist` requires ≥ 2 post months; applied to both `found` and `weak_date` candidates. Both stored in the model so inference detects the same way. Both adopted only with the margin rule; `--no-onset-first` and `--no-tail-guard` switch them off. The dev generator gains `winter_install_ramp` (installs in 1 Oct–28 Feb with a gradual production ramp) and `holiday_tail` (December vacation events in the last 30–45 days of every customer's series); both are variant-A-only and the holdout suite is unchanged. `onset_first` is the fixed dating rule (dates and regimes only; features unchanged): it is applied in every fit, including fits without inner CV, and no longer re-decided per fit by the margin rule; `--no-onset-first` switches it off.

Output corrections (on by default, scored last on the selected inner out-of-fold outputs, after pseudo-labelling): option `date_debias` shifts each regime's dates (`event`, `present_at_start`, `insufficient`) by the median signed date error (predicted − labelled) of that regime's inner out-of-fold true positives at the chosen threshold (≥3 of them, else no shift; clipped to ±365 days). In the inner comparison each inner validation fold is shifted by the other folds' bias, so a customer's own label never moves its date; the final model stores the bias over all its training customers as `date.date_bias`. Option `cap_ceiling` learns a capacity band from the fit's labelled kWp, `[q05 / 1.5, q95 × 1.5]`, scored per inner fold from that fold's training labels; a capacity (before κ) outside the band is replaced by the training median kWp, and the global [0.5, 500] clip stays as the outer safety net (`capacity.band`). Both adopted only with the margin rule; `--no-date-debias` and `--no-cap-ceiling` switch them off, and models without the keys predict exactly as before. The dev generator gains `label_lead_60` (labelled date 30–90 days before the visible onset: late dates), `winter_install` (installs in 15 Nov–31 Jan) and `cap_outlier` (every twentieth customer's load × 60–120: its log-space capacity lands at a clip); the holdout suite is unchanged.

`inner=` is the mean inner-CV score of the final selected configuration over the outer fits, and `gap=` is `inner` minus the nested score. After the capacity model, κ and the date rule are chosen, the decision threshold is rescored once on those final inner out-of-fold outputs (same grid, same tie-breaking); `threshold_initial`/`inner_score_initial` in the selection records keep the pre-rescoring values. The gap is a warning signal, not a measure of overfitting: inner and outer scores come from different customer sets and training sizes. A gap above about 5 points calls for a look at selection stability (the `selection:` line); it proves nothing alone.

Dates come from one of three regimes: `event` (detected changepoint + learned offset), `present_at_start` (PV already present when data starts: first date − learned lag), `insufficient` (weak candidate or first date, chosen by inner CV).

`--split-dir DIR` (optional) first splits `--timeseries` by customer: the unique customer ids are shuffled with `--split-seed` (default 0), a `--split-frac` share (default 0.75, rounded) goes to `DIR/training.csv` and the rest to `DIR/test.csv`, each file holding every row of its customers as text (same separator and columns, leading zeros kept; rows with an empty id go to neither). Training, `--cv` included, then runs on `training.csv` only: labelled and extension customers in `test.csv` are dropped from `--labels`/`--ext-labels`, and `--unlabelled` excludes every customer of the original input. It prints one `split:` line with counts. `DIR` must be outside the repository. Score the held-out customers by running inference on `DIR/test.csv` with the saved model.

### Shared-fold comparison

The integrated pipelines share split files, a prediction contract and one scorer. `--folds F0,F1,...` replaces pvdetect's internal repeated stratified folds: each file (`splits.parquet` or a `;`/`,` CSV with `customer_id,fold`) is one outer repeat and must assign every labelled customer to one integer fold. `--oof-out PATH` retains every repeat in `customer_id;repeat;fold;prob;pv_installed;pv_installation_date;pv_capacity_kwp`. Paired scoring checks each `(repeat, customer_id) → fold` assignment, not just repeat numbers. Use genuinely different customer partitions across repeats.

Matched run (data holder, private directory only):

```
python -m neuralwave_data.prepare <manifest> --out <private>/prep_s{0,1,2} --folds 5 --seed {0,1,2}
python train.py --timeseries <dev energy> --labels <dev labels> --out <private>/pvdetect/model.json --cv \
  --folds <private>/prep_s0/splits.parquet,<private>/prep_s1/splits.parquet,<private>/prep_s2/splits.parquet \
  --oof-out <private>/pvdetect_oof.csv --report-dir <private>/pvdetect_report
python -m neuralwave_experiment.evaluation --canonical-dir <private>/canonical --out <private>/patchtst \
  --data-kind development --splits <private>/prep_s0/splits.parquet,<private>/prep_s1/splits.parquet,<private>/prep_s2/splits.parquet
python -m neuralwave_experiment.oof_contract --labels <dev labels> <private>/pvdetect_oof.csv <private>/patchtst/oof_patchtst_scratch.csv
```

Only the printed aggregate lines leave the private directory, and only to teammates, never to an agent. PatchTST writes empty date and capacity (D = P = 0), so compare F1 separately from the full score. The comparison is a Tier B decision and gets a `DECISIONS.md` line.

### One-command private run and error analysis (data holder)

```
sh scripts/train_and_launch.sh DEV_ENERGY_CSV DEV_LABELS_CSV [UNLABELLED_CSV]
python -m pvdetect.compare BASE_OOF CAND_OOF LABELS [--out-dir <private>]
python -m pvdetect.diagnose PRED_OR_OOF.csv LABELS.csv [--repeat 0] [--out-dir <private>]
```

The script refuses a `PRIVATE_DIR` inside the repository (default `~/pvdetect_private/final`, mode 700). It runs `train.py --cv --oof-out`, saves the final model and `train_log.txt`, then runs `pvdetect.diagnose` on the out-of-fold predictions (every customer predicted by a model that never saw it) into `diagnosis.txt`, `diagnosis.json` and `errors.csv`, and finally opens the GUI with the new model. `CV=0` fits the final model only; `NO_GUI=1` skips the GUI; `PORT=` picks the GUI port.

`pvdetect.diagnose` prints aggregates only: score with F1/D/P and the exact `points_lost:` split; precision, recall and the TP/FP/FN/TN counts; date errors (median absolute and signed days, share late, shares within 0/7/15/30 days); capacity errors (median APE, median predicted/true ratio, share underestimated, shares within 10/25/50 %, predictions at the 500 kWp clip); the same by true size bin, install season and install year; a rule-based `failure_mode:` per component; and `next_step:`. The per-customer table goes only to `--out-dir`, which must be outside the repository. On a submission-format file of customers the model was trained on, every number is optimistic. All of it is AIL-derived on real data: share only the `next_step:` word or a decision with an agent.

`pvdetect.compare` is the paired check for one pre-registered change: two out-of-fold files from the same splits (it refuses differing folds or coverage), per repeat and in total the confusion matrices, precision, recall, F1, D, P and score for both, the paired counts of customers that went wrong→right and right→wrong, and a customer bootstrap of the score difference with a 90 % interval and a verdict (`improvement`, `within_noise`, `worse`). Numbers reuse `pvdetect.diagnose`; the per-customer `changes.csv` goes only to the private `--out-dir`. Typical run: `TRAIN_FLAGS=--no-e1e2` and `TRAIN_FLAGS="--no-e1e2 --schedule"` into two private folders, then compare their `oof.csv`. Each comparison is a Tier B decision (one `DECISIONS.md` line).

### Lock-box protocol (data holder)

Evaluation protocol v2 (`docs/evaluation_lockbox_plan.md`): nested CV on 30 development customers, one sealed lock-box of 10 opened once, then a refit on all 40. The choices are pre-registered in `configs/lockbox.json` (seed 2026; 10 lock-box customers stratified by `pv_installed`; outer split seeds 17, 42, 100 with 5 folds; at most 3 comparisons). Paste one line at a time; `P` is a private folder outside the repository:

```
P=~/pvdetect_private/lockbox
sh scripts/lockbox.sh seal --dir $P --energy ENERGY.csv --labels LABELS.csv
sh scripts/lockbox.sh develop --dir $P base
sh scripts/lockbox.sh develop --dir $P cand -- --schedule
sh scripts/lockbox.sh compare --dir $P base cand
sh scripts/lockbox.sh status --dir $P
sh scripts/lockbox.sh freeze --dir $P base
sh scripts/lockbox.sh open --dir $P
sh scripts/lockbox.sh final --dir $P --energy ENERGY.csv --labels LABELS.csv
python -m pvdetect.stability $P/final/model.json OLD/model.json UNLABELLED.csv
python gui.py --model $P/runs/base/model.json
```

`scripts/lockbox.sh` runs `python -m pvdetect.lockbox` with the same interpreter choice as `train_and_launch.sh` (`PYTHON`, else `.venv/bin/python`, else `python3`). The state lives in `$P/state.json` (stage, timestamps, git commit, config hash, runs, decisions used, forced overrides); each stage refuses to run out of order and prints aggregate lines only.

- `seal` (once per folder; refuses when `state.json` exists): stratified split of the customers into `dev.csv` / `lockbox.csv` (rows copied as text) with `dev_labels.csv` / `lockbox_labels.csv`; the shared outer split files `splits_s17.csv`, `splits_s42.csv`, `splits_s100.csv` (stratified 5-fold over the dev customers, built like `neuralwave_data.prepare`, usable as `train.py --folds` and by the hybrid arms); `seal.json` with the sha256 of the sorted lock-box ids, `lockbox.csv`, `lockbox_labels.csv` and the config. Prints counts only.
- `develop NAME [--unlabelled U] [-- TRAIN_FLAGS]`: `train.py --cv` on the dev customers with the three split files, `--oof-out` and the model into `$P/runs/NAME/`, then `pvdetect.diagnose` on its out-of-fold file. Flags after `--` go to `train.py` (the protocol's own flags are refused). The `nested:` numbers are recorded. Refused after `freeze`.
- `compare A B`: `pvdetect.compare` on the two runs' out-of-fold files into `$P/runs/B/compare_vs_A/`; uses one decision of the budget and prints the verdict and the `DECISIONS.md` line to add. Beyond the budget only with `--force`, which is recorded. Adopt a candidate only on `improvement`; once the budget is spent, freeze.
- `freeze NAME`: records the git commit (and whether tracked files were modified), the run's flags and the hash of its model; prints the `git tag freeze-vN` command. No code change after this.
- `open`: checks the seal hashes and the frozen model's hash, predicts `lockbox.csv` with the frozen run's model (fit on all dev customers) through `inference.run_inference`, scores it like `pvdetect.diagnose` with a 90 % customer-bootstrap interval (2000 draws) and states whether the score is inside, below or above the run's nested `ci90`. `lockbox_result.json` (aggregates only) goes to `$P` and next to the frozen model, where the GUI shows it as "Lock-box · single look"; the predictions stay in `$P/lockbox_predictions/`. A second `open` refuses unless `--force`, which is recorded and shown. Below the interval means possible overfitting: report it and still submit the frozen model; never re-tune on the lock-box. With 10 customers the interval is wide.
- `final --energy E --labels L [--cv]`: after `open` (or with a recorded `--force`), refits the frozen run's flags on all customers into `$P/final/model.json` (`--cv` adds nested CV and its diagnosis) and prints the `scripts/export_submission.sh` command.
- `status`: stage, runs with their nested scores, decisions used of the budget, lock-box opened or not.

`python -m pvdetect.stability MODEL_A MODEL_B UNLABELLED [--pool 300]` is the label-free check of the report stage: it builds the same bounded unlabelled pool as pseudo-labelling (≥365 valid days, salted-hash choice), predicts it with both models and prints the predicted PV share of each, the number and share of customers whose decision flips, and the mean absolute change of the PV probability.

### Unlabelled pretraining and optional hybrid features

Pretraining learns masked-patch reconstruction without PV labels. Supervised customer-level fitting remains necessary: the frozen encoder is not itself a PV classifier, date estimator or capacity estimator.

The pretraining manifest explicitly separates customer-disjoint training and validation populations:

```json
{"sources":[
  {"role":"unlabelled_measurements","split":"train","path":"train.parquet"},
  {"role":"unlabelled_measurements","split":"validation","paths":["validation.parquet"]}
]}
```

These are canonical Parquet files with `customer_id`, `timestamp_epoch`, `energy_kwh` and `value_status`. Sources may contain interleaved customers across files. Labelled customers must be excluded from both populations; overlap is an error. The data holder prepares this manifest and runs real-data commands privately.

```
python -m neuralwave_experiment.pretrain --manifest <private>/manifest.json \
  --exclude-labels <private>/labels.csv --config configs/pretraining_unlabelled.json \
  --device cpu --out <private>/pretrain
```

Use `--device cuda` only on a verified CUDA host. `--max-steps N` pauses cleanly with `run.json` status `paused`; resume with the same configuration and inputs, `--resume <private>/pretrain/checkpoint_latest.pt`, and a new output directory. Private resumable checkpoints include optimizer, RNG, sampler and progress; only the compact `encoder/encoder.pt` and `encoder/encoder.json` pair is used for inference. Training uses fresh observed-patch masks and visible-only normalization; validation uses fixed masks on separate customers. Reconstruction loss is not evidence of PV accuracy.

```
python train.py --timeseries <private>/energy.csv --labels <private>/labels.csv \
  --out <private>/hybrid/model.json --cv --folds <same split files> --oof-out <private>/hybrid_oof.csv \
  --embedding-checkpoint <private>/pretrain/encoder --feature-block auto --report-dir <private>/hybrid_report
```

The frozen encoder produces label-free customer summaries. Imputation, scaling and the four-component projection are fit only on the current training customers, including inside inner folds. Physical features, date detection and capacity estimation remain intact. `auto` compares baseline and embedding candidates on identical inner folds using complete predictions and the existing margin/bootstrap rule; outer results never choose the feature block. `--feature-block embedding` is a forced diagnostic, not an adoption decision. A selected baseline saves no encoder dependency; a selected hybrid saves the exact hash-checked `encoder/` pair with model schema v3.

The integrated `new-stack` options remain available with the encoder: E1 persistence
and E2 probability averaging are enabled by default (`--no-e1e2` disables them),
`--schedule` adds the schedule-consistency candidate, and `--unlabelled` enables
the upstream prior and pseudo-pool preparation. Pseudo-labelling and frozen embeddings
are separate candidates: the compact pseudo pool has no epoch frame for encoder
windows. Automatic hybrid selection compares against the selected physical candidate,
including pseudo-labelling when it wins. A selected hybrid records pseudo-labelling
as off with an explicit reason; it is not a combined pseudo-plus-embedding refit.
Previously saved evaluation results describe their recorded source version, not this
updated option pool.

Before any scale increase, compare the same transformer trained from scratch and with masked pretraining, with matched windows, targets, aggregation, splits and tuning budgets. Evaluate calendar-time persistence/coverage, pre-installation alarms, onset error, late installations and non-PV alarms separately. Fresh blind customers are required for a final independent claim when available; more development seeds cannot replace them.

#### Resumable labelled-sample learning curve

Run privately as the data holder, with the same three outer split files for every arm:

```bash
PYTHONPATH=.:src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/run_learning_curve.py \
  --timeseries <private>/energy.csv --labels <private>/labels.csv \
  --splits <private>/split0.parquet,<private>/split1.parquet,<private>/split2.parquet \
  --small-encoder <private>/small/encoder --medium-encoder <private>/medium/encoder \
  --subsets 8,16,32 --out <private>/learning_curve_results.json
```

The CLI prints preprocessing stages and each fold's start/completion. It preserves
the encoder's saved window configuration, including an uncapped window count.
Calendar features are computed once per customer timeline and sliced into windows;
raw frozen embeddings are reused across folds, while fitted projections stay fold-local.
Missing commissioning dates receive zero date reward in fast bootstrap scoring,
matching the official scorer.

Every finished fold is saved atomically in `<out>.progress.json`; that file contains
private predictions and must remain private. Aggregate results are saved after each
completed arm. After interruption, repeat the identical command with `--resume`.
Resume checks input/encoder hashes, source code, settings and runtime versions, and
refuses mismatches. An already-completed run returns its saved results. Previous
versions saved only at the end, so a timed-out old run cannot be resumed unless it
actually produced a progress file. The CLI uses one Torch CPU thread by default
(`--torch-threads` overrides it); no deadline is imposed by the script.


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

## Local GUI (optional)

A browser interface for running the same detector without the command line. It is optional: `python inference.py path/to/test.csv` remains the submission interface and does not depend on it.

```
python gui.py                                    # default model location (artifacts/model.json)
python gui.py --model /private/final/model.json  # explicit model
python gui.py --model /private/final/model.json --labels /private/labels.csv  # diagnose every analysed CSV
python gui.py --port 8765 --no-browser
```

The terminal prints `PV detector GUI: http://127.0.0.1:PORT/`; the browser opens there unless `--no-browser`. Stop with Ctrl-C.

- **Run synthetic check** needs no files and no model. It generates fabricated variant-A customers (train seed 101, n = 24; test seed 102, n = 12), fits a demo model with the normal training procedure, runs inference, repeats it, runs `inference.py` in an isolated copy of the app, and compares both CSVs byte for byte. The synthetic score is labelled as a regression demonstration, not real-world accuracy.
- **Analyze local CSV** uploads a file through the browser to a private temporary folder and runs it with the model configured at launch only. The demo model is never used for your file; without a configured model this action is disabled.
- **Automatic diagnosis:** with `--labels` (`scripts/train_and_launch.sh` passes the training labels), every completed analysis is scored against the labels of the customers it contains, with the same aggregates as `pvdetect.diagnose`, shown at the top of the sidebar ("Diagnosis"). Customers without a label are left out and counted. On the customers the model was trained on the numbers are optimistic; the cross-validated panel below it is the honest estimate. The synthetic check is diagnosed against its known labels the same way.
- **Model evaluation panel:** when `diagnosis.json` (written by `pvdetect.diagnose --out-dir`) sits next to the configured `model.json`, the sidebar shows it before any run: score and F1, points lost per part, precision/recall with false-alarm and missed counts, date D and error days, capacity P and the predicted/true ratio, the failure modes and the per-size breakdown. It is labelled out-of-fold or in-sample, and marked stale when the model file is newer. Only the aggregate file is read; `errors.csv` is never served.
- **Lock-box panel:** when `lockbox_result.json` (written by `pvdetect.lockbox open`) sits next to the configured `model.json`, a "Lock-box · single look" card appears above the model evaluation panel: the lock-box score with its 90 % interval, F1, precision, recall, D and P, and whether the score is inside, below or above the run's nested interval. A forced re-open, any forced override or a code change after freeze is flagged in orange. Only the aggregate file is read; the lock-box predictions are never served.
- Layout: header actions (Analyze CSV, Run synthetic check, download menu), a sidebar of run figures (customers, PV share, median/total capacity, date regimes, synthetic score, output checks, files, input quality), the selected customer's weekly/daily withdrawal or midday power with the estimated (and, for synthetic data, known) installation marker, a customer table with search and PV filter, and a before/after typical-day comparison (observed withdrawal, not seasonally adjusted). Times use the fixed UTC+1 day matrix of the model.
- **Download** returns the exact `predictions.csv` and `pv_extensions_predictions.csv` written by `pvdetect.output`, byte-identical to the CLI for the same model and input.
- **Clear session data** deletes uploads, outputs, synthetic data and the demo model; your original files, the configured model and downloaded files are untouched. Stopping the app deletes the session folder as well. A hard kill can leave the folder (`pvdetect-gui-*`) in the OS temp directory; it is not securely erased.

Privacy: the server binds to 127.0.0.1 only, rejects other Host headers and cross-origin requests, requires a per-launch token for every API call, logs no data, and loads no external resources. Do not use real customer data in screenshots, videos or reports; demo material comes from the synthetic check.

Design credit: the visual design (dark Quartz layout, palette, chart styling and icons) is ported from Open Climate Fix's MIT-licensed [quartz-frontend](https://github.com/openclimatefix/quartz-frontend); see `pvdetect/gui_assets/Quartz-frontend-MIT.txt`. The bundled Inter font is under the SIL Open Font License (`pvdetect/gui_assets/Inter-OFL.txt`).

## Submission export

The trained model never enters this repository's history: the hygiene hook blocks `artifacts/` and every `.cbm`. It reaches the NW_26 repository through the export script below, which the organizers allow (answer 2): the exported repository carries `artifacts/model.json` and its sibling files next to `inference.py`, which loads them automatically.

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

Questions for the organizers (Step 2.5-H and the team's Discord questions); answers get recorded here.

1. May semi-synthetic augmentation of the provided series be used in the competitive model? *Awaiting answer* (not covered by the answers below; Step 19 stays gated).
2. May the trained model files be published in the NW_26 repo? **Yes** (2026-10-17): a pre-trained model may be included in the repository. `scripts/export_submission.sh COMMIT DEST MODEL_DIR` copies it into `DEST/artifacts/`; the private channel is no longer needed.
3. Sunrise/sunset or solar elevation computed for an assumed location? **No**: it counts as location data, and the customers are spread across Switzerland. Calendar features from the timestamps only (the pipeline uses fixed UTC+1 clock windows).
4. Pseudo-labelling / self-training on the unlabelled series? **Yes**, semi-supervised and hybrid approaches are allowed.
5. Are the 20 hidden test customers in the unlabelled set? **No**, there is no overlap.
6. May per-customer or aggregate statistics derived from AIL data be shared with an LLM coding assistant? **No**, not even aggregates. Real-data outputs (`train.py --cv` lines, review plots, OOF files) stay with the data holder and are never pasted into an agent session or a file an agent reads.
7. Test data: same 2022–2025 period, already filtered to the evaluated POD customers.
8. Evaluation: no internet; any Python version (document it); no formal time or memory limit, fast is appreciated.
9. Date scoring: linear, 1.0 at 0 days, 0.5 at 15 days, 0 at ≥ 30 days (matches `pvdetect.scoring`).
10. Installs before 2022: none in the labelled set; possible in the unlabelled set (never checked).

## Status

The baseline and optional frozen-encoder pipeline are integrated, including upstream unlabelled-derived climatology and evidence-scale options (`--unlabelled`) and inner-CV model selection. Local checks use wholly artificial data only; they establish execution and leakage-contract behavior, not competition accuracy. Real-data nested CV and blind-customer evaluation remain data-holder tasks. Extensions remain header-only. Upstream organizer restrictions apply to all AIL outputs, including aggregates: they must not be shared with coding agents.

Upstream baseline export rehearsal (reported on `Jeffremm10/new-stack`, Step 2.5-X2, synthetic model, 2026-10-17): passed with default `python3` and fresh Python 3.12.14 dependencies; network-blocked inference on 20 synthetic customers produced repeatable output. Hybrid integration verification is recorded separately in `docs/evaluation_protocol.md`.

Clean-environment rehearsal (release hardening, synthetic `generate_dataset(20, seed=9)`, 2026-10-17): **pass** — fresh `uv venv` Python 3.12.14 from `requirements.txt` (no torch), export of `bfc5507` passed `check_release.py`, socket-blocked inference 4 s for 20 customers, exact headers, 20 rows, every TRUE row dated with capacity > 0, two runs byte-identical (training 54 s).

## Canonical preprocessing checks

Run the regression suite with `PYTHONPATH=src python -m pytest -q tests/test_preprocessing.py tests/test_augmentation.py` after installing dependencies. The tests construct only temporary artificial data and exercise preprocessing and synthesis.

## Synthetic data

See [the synthetic data guide](docs/synthetic_data.md) for the synthesis CLI, validation API, output contract, and the fully fabricated local dataset example. Generated measurements, labels, provenance, and latent components stay under ignored private directories; they are not distributed through Git.
