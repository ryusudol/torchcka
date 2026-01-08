"""Utility functions for CKA computation."""

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn


def validate_batch_size(n: int) -> None:
    """Validate that batch size is sufficient for HSIC computation.

    Args:
        n: Batch size (number of samples).

    Raises:
        ValueError: If n <= 3 (unbiased HSIC requires n > 3).
    """
    if n <= 3:
        raise ValueError(
            f"HSIC requires batch size > 3, got {n}. "
            "Increase batch size to at least 4."
        )


def get_device(
    model: nn.Module,
    fallback: Optional[torch.device] = None,
) -> torch.device:
    """Get device from model parameters.

    Args:
        model: PyTorch model.
        fallback: Device to use if model has no parameters.

    Returns:
        Device the model is on.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback or torch.device("cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DistributedDataParallel wrapper.

    Args:
        model: Potentially wrapped model.

    Returns:
        Unwrapped model (accesses .module attribute if wrapped).
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


class FeatureCache:
    """Feature cache for hook outputs.

    Stores layer outputs with optional detaching for memory efficiency.

    Attributes:
        detach: Whether to detach tensors from computation graph.
    """

    def __init__(self, detach: bool = True) -> None:
        """Initialize feature cache.

        Args:
            detach: Whether to detach tensors from computation graph.
        """
        self._features: Dict[str, torch.Tensor] = {}
        self._detach = detach

    def store(self, name: str, tensor: torch.Tensor) -> None:
        """Store a feature tensor.

        Args:
            name: Layer name.
            tensor: Feature tensor to store.
        """
        if self._detach:
            tensor = tensor.detach()
        self._features[name] = tensor

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get a stored feature tensor.

        Args:
            name: Layer name.

        Returns:
            Stored tensor or None if not found.
        """
        return self._features.get(name)

    def clear(self) -> None:
        """Clear all stored features."""
        self._features.clear()

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over stored features.

        Yields:
            Tuples of (layer_name, tensor).
        """
        return iter(self._features.items())

    def keys(self) -> Iterator[str]:
        """Iterate over layer names.

        Yields:
            Layer names.
        """
        return iter(self._features.keys())

    def __len__(self) -> int:
        """Return number of stored features."""
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        """Check if a layer name is in cache."""
        return name in self._features


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Context manager to temporarily set model to eval mode.

    Restores original training state on exit.

    Args:
        model: PyTorch model.

    Yields:
        The model in eval mode.
    """
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(was_training)
