"""Resolution of data files packaged inside the SemanticMemo distribution.

This module is intentionally dependency-light (no torch import) so that
``ClassifierConfig.bundled()`` can resolve the shipped checkpoint path without
pulling in the optional ML stack.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from semanticmemo.exceptions import SemanticMemoError

BUNDLED_CLASSIFIER_NAME = "equivalence-net-v1.pt"
_BUNDLED_MODELS_DIR = "_models"


def bundled_classifier_path() -> Path:
    """Return the on-disk path to the pretrained classifier shipped with SemanticMemo.

    Raises:
        SemanticMemoError: if the checkpoint is missing from this installation.
    """

    resource = resources.files("semanticmemo").joinpath(
        _BUNDLED_MODELS_DIR, BUNDLED_CLASSIFIER_NAME
    )
    path = Path(str(resource))
    if not path.is_file():
        msg = (
            f"The bundled classifier '{BUNDLED_CLASSIFIER_NAME}' is missing from "
            "this SemanticMemo installation. Reinstall the package, or train your own "
            "checkpoint and pass ClassifierConfig(model_path=...)."
        )
        raise SemanticMemoError(msg)
    return path
