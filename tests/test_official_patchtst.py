import pytest


def _inputs(torch, *, batch=2, context=8):
    values = torch.arange(batch * context, dtype=torch.float32).reshape(batch, context, 1)
    observed = torch.ones(batch, context, 1, dtype=torch.bool)
    calendar = torch.zeros(batch, context, 4)
    return values, observed, calendar


def _config():
    from neuralwave_experiment.encoder import EncoderConfig

    return EncoderConfig(
        context_length=8,
        patch_length=2,
        d_model=8,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
    )


def test_official_matches_vendored_reference_when_side_adaptations_are_neutral():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.official_patchtst import OfficialPatchTSTEncoder

    model = OfficialPatchTSTEncoder(_config()).eval()
    with torch.no_grad():
        for layer in (model.observation_proj, model.calendar_proj):
            layer.weight.zero_()
            layer.bias.zero_()
    values, observed, calendar = _inputs(torch)
    tokens, _ = model.token_outputs(values, observed, calendar)
    patches = values.reshape(2, 4, 1, 2)
    reference = model.backbone(patches)[:, 0].transpose(1, 2)
    assert torch.allclose(tokens, reference, atol=1e-6, rtol=1e-6)


def test_official_hidden_values_do_not_change_tokens_or_pool():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.encoder_factory import build_encoder

    model = build_encoder(_config(), "official_patchtst").eval()
    values, observed, calendar = _inputs(torch)
    observed[:, 2:4] = False
    altered = values.clone()
    altered[~observed] = 999999.0
    first, first_mask = model.token_outputs(values, observed, calendar)
    second, second_mask = model.token_outputs(altered, observed, calendar)
    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
    assert torch.allclose(model(values, observed, calendar), model(altered, observed, calendar))


def test_official_parameters_update_and_reload_deterministically():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.encoder_factory import build_encoder

    torch.manual_seed(7)
    model = build_encoder(_config(), "official_patchtst")
    values, observed, calendar = _inputs(torch)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(values, observed, calendar).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    restored = build_encoder(_config(), "official_patchtst").eval()
    restored.load_state_dict(model.state_dict())
    model.eval()
    assert torch.allclose(model(values, observed, calendar), restored(values, observed, calendar))


def test_official_all_missing_is_finite_and_reconstruction_has_gradients():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.encoder_factory import build_encoder

    model = build_encoder(_config(), "official_patchtst")
    values, observed, calendar = _inputs(torch)
    observed.zero_()
    tokens, unavailable = model.token_outputs(values, observed, calendar)
    assert torch.isfinite(tokens).all() and unavailable.all()
    loss = model.reconstruction_head(tokens).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and model.reconstruction_head.weight.grad is not None


def test_finetune_parameters_are_the_actual_last_layers():
    torch = pytest.importorskip("torch")
    from neuralwave_experiment.encoder_factory import build_encoder, finetune_parameters

    for backend in ("custom", "official_patchtst"):
        model = build_encoder(_config(), backend)
        actual = list((model.transformer.layers if backend == "custom" else model.backbone.encoder.layers)[-1].parameters())
        yielded = list(finetune_parameters(model))
        assert yielded and all(a is b for a, b in zip(actual, yielded))
