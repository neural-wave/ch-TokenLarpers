# Official PatchTST comparison

`python -m neuralwave_experiment.comparison --canonical-dir CANONICAL --out RESULTS [--config CONFIG] [--seed N] [--device cpu]`

The command runs the complete three-arm protocol (`baseline`, `frozen`, and
`finetuned`) twice, using the same canonical folds, windows, seed, budgets,
and head settings. It refuses to overwrite a populated output directory and
writes `comparison.json` plus one self-contained output directory per backend.
Each run records backend identity, parameter count, runtime, configuration,
checkpoint state dictionaries, predictions, fold metrics, pooled metrics, and
lineage audits.

`encoder_backend` is `custom` by default and may be set to
`official_patchtst`; the comparison command overrides it for each run and
rejects unknown values. The official adapter vendors the upstream encoder
locally and adapts its inputs to the experiment's single-channel values,
observation masks, and calendar features; it is not claimed to be an
untouched upstream training implementation.

Capacity features deliberately do not use `extensions.parquet` during
prediction. Extension dates are held-out metadata and would leak information
if used to trim input windows. Thus initial-capacity evaluation uses all
available windows under the same raw-window rule for every customer. The
baseline arm is computed independently of either encoder and should be
identical between comparison runs; this is reported as `baseline_metrics_equal`.
No network access, pretrained weights, or external datasets are used.

## Source and adaptation

Upstream is `yuqinie98/PatchTST`, Apache-2.0, pinned at
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`. The minimal self-supervised
model/layer source subset is vendored under
`src/neuralwave_experiment/vendor/patchtst/`, with LICENSE, NOTICE,
modification notices and provenance included in the built wheel.

The adapter preserves upstream value/position projections, feed-forward layers,
channel-independent attention and residual attention scores. It uses LayerNorm,
threads observation padding masks into attention, adds calendar/observation
token projections, and pools a single measurement channel for the existing
downstream protocol. It is a real upstream-backbone adaptation, not a second
implementation using `torch.nn.TransformerEncoder`.

Both backends retain magnitude separately. The custom backend's earlier
initialization incorrectly set LayerNorm scales to zero; this is corrected to
one before the new comparison. Earlier custom smoke scores are not the control
for this comparison.

## What this enables

- Local masked pretraining of numeric consumption representations.
- Frozen-embedding PV classification, date-candidate scoring and conditional
  capacity regression with the same downstream heads as the custom model.
- Limited supervised adaptation of the actual final upstream encoder block.
- Reproducible backend selection, checkpointing and paired evaluation.

The upstream repository also implements forecasting, but a forecasting service,
PV generation disaggregation and an external-checkpoint import path are not
implemented here. Capacity is still statistical inference, not a direct
measurement of nameplate kWp.

## Evidence and interpretation

The regression suite passed 29 tests. A complete artificial-data CLI scenario
trained/evaluated both backends across two folds and wrote all six prediction
files. The package wheel was built and checked for license/provenance inclusion.

User authorization was updated to permit AIL data access with coding agents.
Authorized AIL evaluation remains local, and raw data, manifests, predictions,
checkpoints and customer-level reports remain ignored/private. No weather,
location, external datasets, external pretrained weights or LLM inference are
introduced by this adapter.

## Completed AIL development pilot

All 40 labelled development customers (22 PV, 18 non-PV) were evaluated using
five customer-disjoint folds, seed 17. Settings are in
`configs/patchtst_pilot.json`: 672-step windows, 16-step patches, 32-dimensional
two-layer encoders with four heads, at most 32 windows/customer, one masked
pretraining epoch and one supervised fine-tuning epoch. No synthetic augmentation
or unlabelled histories were used. No externally pretrained weights were used.
Scores below are pooled out-of-fold development results, not hidden-test scores.

| Method | PV F1 | Date D | Capacity P | Main score /100 |
| --- | ---: | ---: | ---: | ---: |
| Raw-feature baseline | 0.6667 | 0.0644 | 0.3329 | 50.6401 |
| Custom, frozen | 0.6829 | 0.0762 | 0.2555 | 51.2023 |
| Custom, fine-tuned | 0.6829 | 0.0405 | 0.2340 | 50.6170 |
| Official PatchTST, frozen | 0.6486 | 0.0417 | 0.3764 | 49.4730 |
| Official PatchTST, fine-tuned | 0.6486 | 0.0417 | 0.3775 | 49.4837 |

The baseline prediction files and metrics were exactly equal across backend
runs. Custom had 30,512 parameters and took 995.4 seconds for its whole
five-fold pipeline; official had 30,450 and took 991.1 seconds. These timings
include repeated CPU profile/date features and heads, not just encoder training.

The frozen custom gain over baseline was only 0.5621 points, with one seed and
40 customers: not evidence of a reliable improvement. Official PatchTST did
not outperform the custom backend under this limited protocol. All date scores
are poor. More training, use of permitted unlabelled histories, alternative
features, and independent tuning/validation could change the ranking; this
pilot does not establish model-family superiority.

Missing-nighttime feature windows produced All-NaN median warnings during the
run; the existing date-feature code converts missing derived values to zero.
The warnings were not suppressed. This missingness treatment and the weak date
candidate method are limitations of the measured pipeline and should be
investigated before a large compute sweep. The earlier custom LayerNorm fix
means these results supersede its artificial smoke numbers as the current
real-data comparison.

Full predictions, checkpoints, configs and lineage audits remain private under
`artifacts_private/patchtst_real_comparison/`; preprocessing outputs are under
`data_private/patchtst_comparison_canonical/`. Only aggregate development metrics
and data-independent pilot settings are documented here.
