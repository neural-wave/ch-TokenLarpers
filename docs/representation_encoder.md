# Numerical representation encoder


For bounded self-supervised runs, `WindowConfig.max_windows_per_customer`
optionally selects uniformly spaced starts (including the final tail) before
materializing window arrays. The default remains uncapped.

`build_windows` converts measurement rows into regular 900-second windows. Windows are independent per customer: gaps remain missing, never zero observations. `observed` is true only for finite rows whose `value_status` is `observed`; `values` are robustly normalized per window (median and MAD), while `magnitude` retains center, robust scale, mean, and standard deviation in kWh. Calendar channels are Zurich local-time sine/cosines.

`NumericPatchEncoder` consumes float tensors shaped `[batch, context, 1]`, a boolean observation tensor of the same shape, and four calendar channels. Context must divide into nonoverlapping patches. Empty patches are excluded from attention pooling. `encode_windows` uses deterministic evaluation mode and returns float32 NumPy embeddings.

`pretrain_encoder` masks complete observed patches, including both values and observation indicators, and reconstructs only genuine observed values. It updates the supplied model in place and returns finite epoch losses. A checkpoint can be saved and restored with the ordinary PyTorch `state_dict` API:

```python
state = model.state_dict()
model2 = NumericPatchEncoder(config)
model2.load_state_dict(state)
```

No customer identifiers or external checkpoints are used by this package.
