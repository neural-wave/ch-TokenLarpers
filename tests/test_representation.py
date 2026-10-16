import numpy as np
import pandas as pd
import pytest

def _rows():
    rows=[]
    for customer, offset in [("a",0),("b",90000)]:
        for i in range(40):
            rows.append({"customer_id":customer,"timestamp_epoch":offset+i*900,
                         "energy_kwh":float(i+1),"value_status":"missing" if i == 7 else "observed"})
    return pd.DataFrame(rows)

def test_windows_mask_and_boundaries():
    from neuralwave_experiment.windows import WindowConfig, build_windows
    b=build_windows(_rows(), WindowConfig(context_length=16,stride=8,min_observed_fraction=.5))
    assert b.values.dtype == np.float32 and b.observed.dtype == np.bool_
    assert set(b.customer_ids) == {"a","b"}
    assert np.all(b.values[~b.observed] == 0)
    assert np.all(b.magnitude[:,2] > 0)


def test_stride_tail_and_strict_epoch_validation():
    from neuralwave_experiment.windows import WindowConfig, build_windows
    cfg = WindowConfig(context_length=8, stride=3, min_observed_fraction=.5)
    b = build_windows(_rows(), cfg)
    starts = sorted(set(int(x) for x, c in zip(b.start_epochs, b.customer_ids) if c == "a"))
    assert starts[-1] - starts[-2] != cfg.stride * 900
    bad = _rows()
    bad.loc[0, "timestamp_epoch"] += 1
    with pytest.raises(ValueError, match="900-second"):
        build_windows(bad, cfg)
    dup = _rows().iloc[[0, 1]].copy()
    dup.loc[1, "timestamp_epoch"] = dup.loc[0, "timestamp_epoch"]
    with pytest.raises(ValueError, match="duplicate"):
        build_windows(dup, cfg)
def test_encoder_and_pretraining_update():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.windows import WindowConfig, build_windows
    from neuralwave_experiment.encoder import EncoderConfig, NumericPatchEncoder, encode_windows
    from neuralwave_experiment.pretraining import pretrain_encoder
    b=build_windows(_rows(), WindowConfig(context_length=16,stride=16,min_observed_fraction=.5))
    m=NumericPatchEncoder(EncoderConfig(context_length=16,patch_length=4,d_model=16,n_heads=4,n_layers=1))
    before=encode_windows(m,b)
    old={k:v.detach().clone() for k,v in m.state_dict().items()}
    losses=pretrain_encoder(m,b,epochs=1,batch_size=4,seed=2)
    after=encode_windows(m,b)
    assert np.isfinite(losses).all() and not np.allclose(before,after)
    assert any(not torch.equal(v,m.state_dict()[k]) for k,v in old.items())

def test_masked_inputs_do_not_leak_and_reload_is_deterministic():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.windows import WindowConfig, build_windows
    from neuralwave_experiment.encoder import EncoderConfig, NumericPatchEncoder, encode_windows
    b = build_windows(_rows(), WindowConfig(context_length=16, stride=16, min_observed_fraction=.5))
    cfg = EncoderConfig(context_length=16, patch_length=4, d_model=16, n_heads=4, n_layers=1)
    m = NumericPatchEncoder(cfg)
    first = encode_windows(m, b)
    altered = type(b)(b.values.copy(), b.observed.copy(), b.calendar.copy(),
                      b.magnitude.copy(), b.customer_ids, b.center_dates, b.start_epochs)
    altered.values[~altered.observed] = 999999.0
    second = encode_windows(m, altered)
    assert np.isfinite(first).all() and np.allclose(first, second)
    restored = NumericPatchEncoder(cfg)
    restored.load_state_dict(m.state_dict())
    assert np.allclose(first, encode_windows(restored, b))

def test_reconstruction_uses_observed_positions_only():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.encoder import EncoderConfig, NumericPatchEncoder
    cfg = EncoderConfig(context_length=8, patch_length=4, d_model=16, n_heads=4, n_layers=1)
    m = NumericPatchEncoder(cfg)
    values = torch.tensor([[[1.0], [3.0], [5.0], [7.0], [100.0], [100.0], [100.0], [100.0]]])
    observed = torch.tensor([[[True], [True], [True], [True], [False], [False], [False], [False]]])
    calendar = torch.zeros(1, 8, 4)
    hidden_values = values.clone(); hidden_values[:, :4] = 0
    hidden_mask = observed.clone(); hidden_mask[:, :4] = False
    tokens, _ = m.token_outputs(hidden_values, hidden_mask, calendar)
    prediction = m.reconstruction_head(tokens)
    loss = ((prediction[:, 0, :] - values[:, :4, 0]) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss) and m.reconstruction_head.weight.grad.abs().sum() > 0

def test_bad_divisibility():
    from neuralwave_experiment.encoder import EncoderConfig, NumericPatchEncoder
    with pytest.raises(ValueError): NumericPatchEncoder(EncoderConfig(context_length=10,patch_length=4))
