"""Resolution of data files packaged inside the smartmemo distribution.

This module is intentionally dependency-light (no torch import) so that
``ClassifierConfig.bundled()`` can resolve the shipped checkpoint path without
pulling in the optional ML stack.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from smartmemo.exceptions import SmartMemoError

BUNDLED_CLASSIFIER_NAME = "classifier-v2.pt"
_BUNDLED_MODELS_DIR = "_models"


def bundled_classifier_path() -> Path:
    """Return the on-disk path to the pretrained classifier shipped with smartmemo.

    Raises:
        SmartMemoError: if the checkpoint is missing from this installation.
    """

    resource = resources.files("smartmemo").joinpath(_BUNDLED_MODELS_DIR, BUNDLED_CLASSIFIER_NAME)
    path = Path(str(resource))
    if not path.is_file():
        msg = (
            f"The bundled classifier '{BUNDLED_CLASSIFIER_NAME}' is missing from "
            "this smartmemo installation. Reinstall the package, or train your own "
            "checkpoint and pass ClassifierConfig(model_path=...)."
        )
        raise SmartMemoError(msg)
    return path
