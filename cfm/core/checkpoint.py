"""Saving and loading trained models.

Every example used to open-code this pair of operations, so the same six
lines appeared roughly eight times across ``examples/``.

Checkpoints are cloudpickled, which stores classes *by module path*. Models
trained before the ``cfm/`` package existed therefore reference
``architectures.flow`` and friends, so ``load`` remaps those paths onto their
new homes rather than making every previously trained model unloadable --
retraining the MNIST UNet costs about half an hour on a GPU.
"""

import pickle
from pathlib import Path
from typing import Tuple

import cloudpickle
from flax import nnx

from cfm.models.normalizer import Normalizer

# Old module path -> new module path, for checkpoints written before the
# package move. Safe to drop once no pre-refactor checkpoints remain.
LEGACY_MODULES = {
    "architectures": "cfm.models",
    "datasets": "cfm.datasets",
}


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that redirects pre-``cfm`` module paths."""

    def find_class(self, module: str, name: str):
        root = module.split(".", 1)[0]
        if root in LEGACY_MODULES:
            module = LEGACY_MODULES[root] + module[len(root):]
        return super().find_class(module, name)


def save(path, model: nnx.Module, normalizer: Normalizer) -> None:
    """Write a trained model and its normalizer to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        cloudpickle.dump({"model": model, "normalizer": normalizer}, f)


def load(path) -> Tuple[nnx.Module, Normalizer]:
    """Read a trained model and its normalizer from ``path``.

    Returns:
        model: The trained flow model.
        normalizer: The normalizer fitted during training.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one first, e.g. "
            f"`uv run -m cfm.cli train --problem <name>`."
        )
    with open(path, "rb") as f:
        data = _LegacyUnpickler(f).load()
    return data["model"], data["normalizer"]
