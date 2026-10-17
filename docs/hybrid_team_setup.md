# Hybrid PV: teammate setup and offline use

## What is included

Both historical pretrained PatchTST encoders are ordinary Git files. A normal clone checks out their bytes; **no Git LFS, model download helper, GitHub release, API token, or GPU is needed** to load them locally.

```text
pretrained/
  small/encoder.pt
  small/encoder.json
  medium/encoder.pt
  medium/encoder.json
```

Small uses width 64/two layers; medium uses width 128/three layers; both have four attention heads, seven-day quarter-hourly input windows, and four-hour patches. These are the same historical tensor weights, with distribution metadata minimized to configuration, hashes and label-free provenance. They are not claims of improved PV accuracy.

`configs/pretrained_encoders.json` pins both file hashes for each model. The commit and export hygiene checks permit only those exact approved bytes at those exact paths. Other model dumps, datasets, private metrics and optimizer checkpoints remain blocked.

## Install once

Run from the repository root, using Python 3.12 (the package supports 3.11–3.12):

```bash
git switch main
git pull --ff-only origin main
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e '.[experiment]'
sh scripts/install_hooks.sh
python scripts/check_pretrained.py --models small medium
```

Dependency installation requires network access unless compatible wheels are already available locally. **Once dependencies and the clone are present, encoder loading, feature extraction and inference do not download anything.**

Frozen-feature CPU use does not require the full unlabelled dataset, reconstruction pretraining, CUDA or a Vast account. Only retraining needs pretraining inputs; CUDA is an explicit optional environment choice. Baseline-only use does not require Torch. Avoid stale installed wheels: use the editable installation above and run commands from this checkout.

## What the weights do—and do not do

The encoders convert numerical windows, observation masks and timestamp calendar features into vectors. They do **not**, by themselves, output PV presence, installation date or capacity.

The hybrid pipeline aggregates window embeddings as mean/std/last-minus-first, fits imputation/scaling/four-component PCA on training customers only, and adds those features to the presence classifier. Date and capacity estimators retain their physical-feature paths. The encoder stays frozen.

- `--embedding-checkpoint pretrained/small` selects the small representation.
- `--embedding-checkpoint pretrained/medium` selects the medium representation.
- `--feature-block auto` lets inner CV retain the baseline or select embeddings.
- `--feature-block embedding` forces embeddings for a diagnostic; it is not an adoption rule.
- No encoder checkpoint argument runs the physical-feature pipeline.

A complete offline judging package additionally requires the **fitted downstream model**: `model.json`, its fitted projection/parameters, referenced CatBoost files if any, and the exact encoder pair if selected. `save_model` packages those references. Use the existing clean submission export; do not mistake these encoder weights for the final supervised submission model.

## Agent-friendly synthetic exercise

Agents can use the checked-in approved encoders with independently fabricated inputs. They must not open AIL data, private labels, private checkpoints, reports, predictions, or real-data aggregate metrics. For a small runnable CLI exercise:

```bash
WORK="$(mktemp -d)"
python -m pvdetect.synthetic "$WORK/train" --n 12 --seed 17
python train.py --timeseries "$WORK/train/energy.csv" \
  --labels "$WORK/train/labels.csv" --out "$WORK/hybrid/model.json" \
  --embedding-checkpoint pretrained/small --feature-block embedding
```

Repeat with `pretrained/medium` and a different output directory to exercise that size. A synthetic score proves software behavior, not real-world accuracy. `scripts/check_pretrained.py` offers the faster model-integrity/forward-pass check for both sizes.

For final CSV inference, `python inference.py path/to/test.csv` loads `artifacts/model.json` in the exported submission. For an explicit model and output path during development, use the shared Python API:

```python
from inference import run_inference
run_inference("/private/test.csv", "/private/run/model.json", "/private/output")
```

Use fabricated files for agent-executed calls; the real-data forms below are for the human data holder.

## Shared real-data prerequisites: data holder only

Teammates need the same privately distributed:

1. Development measurement CSV and corresponding labels.
2. Predeclared customer-level outer split files, covering the same labelled customers.
3. Option settings and experiment seed/protocol.
4. Optional unlabelled CSV only if explicitly testing the upstream prior/pseudo-labelling options; pass it identically to competing arms.

They do not need raw unlabelled histories merely to reuse the frozen encoders. Keep data, logs, outputs and fitted models outside the tracked source tree. Never send their contents or metrics to a coding agent.

## Matched baseline versus hybrid comparison

Use identical data, folds and non-embedding options. Set these variables to private paths as the data holder:

```bash
DEV_ENERGY="/private/development/energy.csv"
DEV_LABELS="/private/development/labels.csv"
SPLITS="/private/splits/s0.parquet,/private/splits/s1.parquet,/private/splits/s2.parquet"
PRIVATE="/private/experiments/comparison-name"

python train.py --timeseries "$DEV_ENERGY" --labels "$DEV_LABELS" \
  --out "$PRIVATE/baseline/model.json" --cv --folds "$SPLITS" \
  --oof-out "$PRIVATE/baseline/oof.csv" --report-dir "$PRIVATE/baseline/report"

python train.py --timeseries "$DEV_ENERGY" --labels "$DEV_LABELS" \
  --out "$PRIVATE/hybrid-small/model.json" --cv --folds "$SPLITS" \
  --embedding-checkpoint pretrained/small --feature-block auto \
  --oof-out "$PRIVATE/hybrid-small/oof.csv" --report-dir "$PRIVATE/hybrid-small/report"

python train.py --timeseries "$DEV_ENERGY" --labels "$DEV_LABELS" \
  --out "$PRIVATE/hybrid-medium/model.json" --cv --folds "$SPLITS" \
  --embedding-checkpoint pretrained/medium --feature-block auto \
  --oof-out "$PRIVATE/hybrid-medium/oof.csv" --report-dir "$PRIVATE/hybrid-medium/report"

python -m pvdetect.compare "$PRIVATE/baseline/oof.csv" \
  "$PRIVATE/hybrid-small/oof.csv" "$DEV_LABELS" --out-dir "$PRIVATE/compare-small"
python -m pvdetect.compare "$PRIVATE/baseline/oof.csv" \
  "$PRIVATE/hybrid-medium/oof.csv" "$DEV_LABELS" --out-dir "$PRIVATE/compare-medium"
```

`pvdetect.compare` checks coverage and customer/fold assignments. The shared main score is `F1 * (70 + 15*D + 15*P)`; presence F1 is not percent accuracy. Inner CV makes automatic model/threshold/feature choices; do not adopt whichever outer score is largest. Record outer-informed redesign/admission decisions in `DECISIONS.md`.

E1 persistence/E2 probability averaging are enabled by default; `--no-e1e2` disables them, `--schedule` adds a candidate, and `--unlabelled` supplies optional priors/pseudo pools. Keep these settings matched across arms. Pseudo and embedding candidates are separate because the compact pseudo pool has no epoch frame for encoder windows.

## Independent teammate experiments

Create a branch from the shared mainline:

```bash
git switch main
git pull --ff-only origin main
git switch -c team/my-name-my-hypothesis
```

Use one explicit hypothesis per branch and fresh private output directories. Preserve existing study files. The new-stack merge expanded the option pool, so historical pre-merge results are not measurements of current main.

Known research caveat: training applies robust visible-only scale fallback/clipping, while current frozen extraction uses MAD normalization with a smaller floor. This discrepancy is untested as a cause of task results; investigate using fabricated flat/sparse/spike-heavy histories before changing it. Physical features use fixed UTC+1 day matrices; encoder calendar features use Europe/Zurich. Do not silently change either convention during a comparison.

Read `docs/hybrid_pv_handoff.md` for the engineering history and unresolved controls. Public availability of the explicitly approved encoder pairs does not authorize publication of AIL inputs, metrics, private training state or the source repository's old history.
