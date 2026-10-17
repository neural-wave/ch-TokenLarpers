# Offline embedding experiment

Run the three real arms on a version-1 canonical directory:

```sh
python -m neuralwave_experiment.experiment --canonical-dir CANONICAL --out PRIVATE_OUT \
  --config configs/embedding_experiment.json --device cpu
```

The runner uses the existing `splits.parquet` customer folds. For each outer fold,
held-out customers are excluded from window fitting, pretraining, scaling and all
heads. Outputs are written only to a new empty directory: semicolon-separated
out-of-fold predictions for `baseline`, `frozen`, and `finetuned`, pooled/fold
metrics, local checkpoints and loss histories, feature definitions, and a lineage
manifest. The fixed presence threshold is 0.5; conditional date and capacity
fields are empty for negative predictions.

Synthetic augmentation is optional and must be a fold-specific root, never one
shared generated directory:

```
PRIVATE_SYNTHETIC/fold_0/{measurements.parquet,synthetic_labels.parquet,provenance.parquet,run_manifest.json}
PRIVATE_SYNTHETIC/fold_1/{...}
```

Each folder must have provenance columns `source_customer_id` and `source_fold`.
The runner rejects missing, held-out, non-training, or mismatched ancestry and
refuses to reuse `fold_0` for another fold. Synthetic capacity labels are never
used. No network access or external checkpoint is needed at runtime.

`score_predictions` implements the challenge formula exactly: `main_score = F1 *
(70 + 15D + 15P)`, with invalid or missing conditional targets contributing zero.
Fabricated smoke fixtures are not benchmark evidence.
