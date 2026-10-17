"""Vendored yuqinie98/PatchTST self-supervised source."""

UPSTREAM_REPOSITORY = "https://github.com/yuqinie98/PatchTST"
UPSTREAM_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"

from .models.patchTST import PatchTST, PatchTSTEncoder

__all__ = ["PatchTST", "PatchTSTEncoder", "UPSTREAM_REPOSITORY", "UPSTREAM_COMMIT"]
