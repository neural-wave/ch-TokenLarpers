"""Backend selection for NeuralWave patch encoders."""
from __future__ import annotations

from collections.abc import Iterator

from torch import nn

from .encoder import EncoderConfig, NumericPatchEncoder
from .official_patchtst import OfficialPatchTSTEncoder


def build_encoder(config: EncoderConfig, backend: str = "custom") -> nn.Module:
    """Build the requested encoder without registering duplicate module paths."""
    if backend == "custom":
        model = NumericPatchEncoder(config)
    elif backend == "official_patchtst":
        model = OfficialPatchTSTEncoder(config)
    else:
        raise ValueError(f"unknown encoder backend: {backend!r}")
    model.backend = backend
    return model


def finetune_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield parameters in the final actual encoder layer.

    Both implementations retain their genuine layer stack; no synthetic
    ``transformer`` wrapper or second registration is introduced here.
    """
    if isinstance(model, NumericPatchEncoder):
        layers = model.transformer.layers
    elif isinstance(model, OfficialPatchTSTEncoder):
        layers = model.backbone.encoder.layers
    else:
        backend = getattr(model, "backend", None)
        if backend == "custom" and hasattr(model, "transformer"):
            layers = model.transformer.layers
        elif backend == "official_patchtst" and hasattr(model, "backbone"):
            layers = model.backbone.encoder.layers
        else:
            raise TypeError("model is not a supported encoder backend")
    if not layers:
        raise ValueError("encoder has no layers")
    yield from layers[-1].parameters()
