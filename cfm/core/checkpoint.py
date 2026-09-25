"""Saving and loading trained models."""

import pickle
from pathlib import Path
from typing import Tuple

import cloudpickle
from flax import nnx

from cfm.models.normalizer import Normalizer

# Remap old module paths so older cloudpickled checkpoints still load.
LEGACY_MODULES = {
    "architectures": "cfm.models",
    "datasets": "cfm.datasets",
}


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that applies ``LEGACY_MODULES``."""

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
    """Read a trained model and its normalizer from ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one first, e.g. "
            f"`uv run -m cfm.cli train --problem <name>`."
        )
    with open(path, "rb") as f:
        data = _LegacyUnpickler(f).load()
    return data["model"], data["normalizer"]
