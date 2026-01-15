from typing import Dict, Iterator, Tuple

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
    fallback: torch.device | None = None,
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
        self._features: Dict[str, torch.Tensor] = {}
        self._detach = detach

    def store(self, name: str, tensor: torch.Tensor) -> None:
        if self._detach:
            tensor = tensor.detach()
        self._features[name] = tensor

    def get(self, name: str) -> torch.Tensor | None:
        return self._features.get(name)

    def clear(self) -> None:
        self._features.clear()

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return iter(self._features.items())

    def keys(self) -> Iterator[str]:
        return iter(self._features.keys())

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features
