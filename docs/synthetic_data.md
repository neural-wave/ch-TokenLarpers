# Constrained synthetic PV data

Pipeline B generates deterministic, paired scenarios from the persisted canonical
contract. It does not use weather, location, irradiance, labels from unlabelled
customers, or a forecasting/model framework.

## Inputs

`--canonical-dir` must contain `run_manifest.json` with `schema_version: 1`,
`measurements.parquet`, and `labels.parquet`. `--splits` is a canonical
`splits.parquet` file containing exactly one `customer_id, fold` assignment for
each labelled development customer. Only labelled `pv_installed == false`
customers in training folds are eligible; the held-out fold is removed before any
scale is calculated. A source with missing split assignment, negative baseline
reading, insufficient pre/post coverage, or zero daytime scale is rejected or
excluded with a reason in `quality_report.json`.

```sh
PYTHONPATH=src python -m neuralwave_data.synthesize \
  --canonical-dir /path/to/canonical \
  --splits /path/to/canonical/splits.parquet \
  --held-out-fold 0 --meter-mode import_only --seed 17 \
  --config configs/augmentation_v1.json --out /path/to/private/synthetic
```

The command refuses to overwrite a populated output directory. `import_only`
clips the meter equation at zero; `signed_net` preserves signed net output.
The latent native load remains nonnegative by construction. Timestamps, local
metadata and source missingness are copied exactly. PV is zero before the known
injected onset and outside the compact standard-UTC+1 daytime support. DST
repeated wall-clock hours are therefore evaluated by epoch rather than by an
ambiguous local timestamp.

## Outputs

* `measurements.parquet`: generated canonical measurements, four scenarios per
  source (`original_negative`, `pv_positive`, `demand_change_negative`, and
  `pv_demand_change_positive`).
* `synthetic_labels.parquet`: scenario labels. Positive onset dates are the
  injected event; `pv_capacity_kwp` is always null and
  `capacity_target_valid` is always false.
* `provenance.parquet`: source fold, seed, scenario, meter mode, onset, ramp,
  scale and generator parameters. Generated IDs are stable hashes and do not
  expose source IDs.
* `latent.parquet`: private baseline, demand and PV components used for
  independent equation checks.
* `quality_report.json` and `run_manifest.json`: hard-gate diagnostics,
  artifact counts and schema/config identity.

The validator can be run by consumers after loading artifacts:

```python
from neuralwave_data.synthetic_validation import validate_synthetic_directory
validate_synthetic_directory("/path/to/private/synthetic", held_out_fold=0)
```

Diagnostic summaries are not a claim of physical realism or improved
real-data generalization. Keep generated artifacts and manifests private.
