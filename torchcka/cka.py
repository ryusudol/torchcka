"""Main CKA class for comparing neural network representations.

This module provides the CKA class for computing Centered Kernel Alignment
between layers of PyTorch models with proper hook management and memory safety.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CKAConfig
from .core import compute_gram_matrix, hsic
from .utils import (
    FeatureCache,
    auto_select_layers,
    get_device,
    unwrap_model,
    validate_batch_size,
    validate_layers,
)

BATCH_INPUT_KEYS = ("input", "inputs", "x", "image", "images", "input_ids", "pixel_values")


@dataclass
class ModelInfo:
    """Information about a model being compared."""

    name: str
    layers: List[str]


class CKA:
    """Centered Kernel Alignment for comparing neural network representations.

    This class provides a context-manager-based API for safe hook management
    and efficient CKA computation between model layers.

    Example:
        >>> config = CKAConfig(kernel="linear", unbiased=True)
        >>> with CKA(model1, model2, layers1=["layer1", "layer2"], config=config) as cka:
        ...     matrix = cka.compare(dataloader)
        ...     print(matrix)

    Attributes:
        model1: First model for comparison.
        model2: Second model for comparison.
        config: CKA computation configuration.
        model1_info: Metadata about first model.
        model2_info: Metadata about second model.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module] = None,
        layers1: Optional[List[str]] = None,
        layers2: Optional[List[str]] = None,
        model1_name: Optional[str] = None,
        model2_name: Optional[str] = None,
        config: Optional[CKAConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize CKA analyzer.

        Args:
            model1: First model to compare.
            model2: Second model. If None, compares model1 with itself.
            layers1: Layers to hook in model1. If None, auto-selects up to 50 layers.
            layers2: Layers to hook in model2. If None, uses layers1.
            model1_name: Display name for model1.
            model2_name: Display name for model2.
            config: CKA computation configuration.
            device: Device for computation. If None, auto-detects from model.

        Raises:
            ValueError: If no valid layers are found.
        """
        # Handle same-model comparison
        self._same_model = model2 is None or model1 is model2

        # Unwrap DataParallel/DDP
        self.model1 = unwrap_model(model1)
        self.model2 = unwrap_model(model2) if model2 is not None else self.model1

        # Configuration
        self.config = config or CKAConfig()
        if device is not None:
            self.config.device = torch.device(device) if isinstance(device, str) else device
        if self.config.device is None:
            self.config.device = get_device(self.model1)

        # Validate and store layers
        if layers1 is None:
            layers1, _ = auto_select_layers(self.model1, max_layers=50, model_name="Model1")

        if layers2 is None:
            if self._same_model:
                layers2 = layers1
            else:
                layers2, _ = auto_select_layers(self.model2, max_layers=50, model_name="Model2")

        # Validate layers exist
        valid1, invalid1 = validate_layers(self.model1, layers1)
        valid2, invalid2 = validate_layers(self.model2, layers2)

        if invalid1:
            warnings.warn(f"Layers not found in model1: {invalid1}")
        if invalid2 and not self._same_model:
            warnings.warn(f"Layers not found in model2: {invalid2}")

        if not valid1:
            raise ValueError(
                "No valid layers found in model1. "
                "Use model.named_modules() to see available layers."
            )
        if not valid2:
            raise ValueError(
                "No valid layers found in model2. "
                "Use model.named_modules() to see available layers."
            )

        self.layers1 = valid1
        self.layers2 = valid2

        # Model info
        self.model1_info = ModelInfo(
            name=model1_name or self.model1.__class__.__name__,
            layers=self.layers1,
        )
        self.model2_info = ModelInfo(
            name=model2_name or self.model2.__class__.__name__,
            layers=self.layers2,
        )

        # Feature storage
        # When _same_model is True, _features1 stores the union of layers1 and layers2,
        # and _features2 is aliased to _features1 in compare() for efficiency.
        self._features1 = FeatureCache(detach=True)
        self._features2 = FeatureCache(detach=True)

        # Hook handles for cleanup
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Training state for restoration
        self._model1_training: Optional[bool] = None
        self._model2_training: Optional[bool] = None

        # Hooks registered flag
        self._hooks_registered = False

    # =========================================================================
    # CONTEXT MANAGER PROTOCOL
    # =========================================================================

    def __enter__(self) -> "CKA":
        """Enter context: register hooks and set eval mode."""
        self._register_hooks()
        self._save_training_state()
        self.model1.eval()
        if not self._same_model:
            self.model2.eval()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit context: remove hooks and restore training state."""
        self._remove_hooks()
        self._restore_training_state()
        return False

    # =========================================================================
    # HOOK MANAGEMENT
    # =========================================================================

    def _make_hook(
        self,
        cache: FeatureCache,
        layer_name: str,
    ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Create a hook function for a specific layer.

        Args:
            cache: Feature cache to store outputs.
            layer_name: Name of the layer being hooked.

        Returns:
            Hook function.
        """

        def hook(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Handle tuple outputs (e.g., from attention layers)
            if isinstance(output, tuple):
                output = output[0]
            cache.store(layer_name, output)

        return hook

    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        if self._hooks_registered:
            return

        # When same model, hook union of layers1 and layers2 to _features1.
        # A single forward pass will populate all needed features, and
        # _features2 will be aliased to _features1 in compare().
        layers_to_hook = set(self.layers1)
        if self._same_model:
            layers_to_hook = layers_to_hook.union(set(self.layers2))

        # Register hooks for model1
        for name, module in self.model1.named_modules():
            if name in layers_to_hook:
                handle = module.register_forward_hook(
                    self._make_hook(self._features1, name)
                )
                self._hook_handles.append(handle)

        # Register hooks for model2 (only if different model)
        if not self._same_model:
            for name, module in self.model2.named_modules():
                if name in self.layers2:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features2, name)
                    )
                    self._hook_handles.append(handle)

        self._hooks_registered = True

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._hooks_registered = False

    def _save_training_state(self) -> None:
        """Save models' training state for later restoration."""
        self._model1_training = self.model1.training
        if not self._same_model:
            self._model2_training = self.model2.training

    def _restore_training_state(self) -> None:
        """Restore models' training state."""
        if self._model1_training is not None:
            self.model1.train(self._model1_training)
        if not self._same_model and self._model2_training is not None:
            self.model2.train(self._model2_training)

    # =========================================================================
    # MAIN API
    # =========================================================================

    def compare(
        self,
        dataloader: DataLoader,
        dataloader2: Optional[DataLoader] = None,
        progress: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Compute CKA similarity matrix between model layers.

        Args:
            dataloader: DataLoader for model1 (and model2 if dataloader2 is None).
            dataloader2: Optional separate DataLoader for model2.
            progress: Show progress bar.
            callback: Optional callback(batch_idx, total_batches, current_matrix).

        Returns:
            CKA similarity matrix of shape (len(layers1), len(layers2)).

        Raises:
            RuntimeError: If hooks are not registered (use context manager).
        """
        if not self._hooks_registered:
            raise RuntimeError(
                "Hooks not registered. Use 'with CKA(...) as cka:' context manager "
                "or call _register_hooks() first."
            )

        if dataloader2 is None:
            dataloader2 = dataloader

        n_layers1 = len(self.layers1)
        n_layers2 = len(self.layers2)

        # Accumulators for minibatch CKA
        hsic_xy = torch.zeros(
            n_layers1, n_layers2, device=self.config.device, dtype=self.config.dtype
        )
        hsic_xx = torch.zeros(n_layers1, device=self.config.device, dtype=self.config.dtype)
        hsic_yy = torch.zeros(n_layers2, device=self.config.device, dtype=self.config.dtype)

        total_batches = min(len(dataloader), len(dataloader2))
        iterator = zip(dataloader, dataloader2)

        if progress:
            iterator = tqdm(iterator, total=total_batches, desc="Computing CKA")

        with torch.no_grad():
            for batch_idx, (batch1, batch2) in enumerate(iterator):
                # Clear previous features
                self._features1.clear()
                self._features2.clear()

                x1 = self._extract_input(batch1)
                x2 = self._extract_input(batch2)

                validate_batch_size(x1.shape[0], self.config.unbiased)

                x1 = x1.to(self.config.device)
                x2 = x2.to(self.config.device)

                # Forward pass
                self.model1(x1)
                if not self._same_model:
                    self.model2(x2)
                else:
                    # Same model: _features1 contains union of layers1 and layers2
                    self._features2 = self._features1

                self._accumulate_hsic(hsic_xy, hsic_xx, hsic_yy)

                if callback is not None:
                    current_cka = self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)
                    callback(batch_idx, total_batches, current_cka)

        # Compute final CKA matrix
        cka_matrix = self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)

        return cka_matrix

    def _extract_input(self, batch: Any) -> torch.Tensor:
        """Extract input tensor from batch.

        Args:
            batch: Batch from DataLoader (tensor, tuple, list, or dict).

        Returns:
            Input tensor.

        Raises:
            TypeError: If batch type is not supported.
            ValueError: If cannot find input in dict batch.
        """
        if isinstance(batch, torch.Tensor):
            return batch
        elif isinstance(batch, (list, tuple)):
            return batch[0]
        elif isinstance(batch, dict):
            for key in BATCH_INPUT_KEYS:
                if key in batch:
                    return batch[key]
            raise ValueError(f"Cannot find input in dict batch. Keys: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _prepare_gram_and_self_hsic(
        self,
        feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare gram matrix and compute self-HSIC for a feature tensor.

        Handles flattening, dtype conversion, gram computation, and HSIC(K,K).

        Args:
            feat: Feature tensor from a layer, shape (B, ...).

        Returns:
            Tuple of (gram_matrix, self_hsic_value).
        """
        if feat.dim() > 2:
            feat = feat.flatten(1)

        feat = feat.to(dtype=self.config.dtype)

        gram = compute_gram_matrix(
            feat, self.config.kernel, self.config.sigma, self.config.epsilon
        )

        hsic_self = hsic(gram, gram, self.config.unbiased, self.config.epsilon)

        return gram, hsic_self

    def _accumulate_hsic(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> None:
        """Accumulate HSIC values for minibatch CKA.

        Args:
            hsic_xy: Accumulator for HSIC(K, L), shape (n_layers1, n_layers2).
            hsic_xx: Accumulator for HSIC(K, K), shape (n_layers1,).
            hsic_yy: Accumulator for HSIC(L, L), shape (n_layers2,).
        """
        gram1_cache: Dict[str, torch.Tensor] = {}
        gram2_cache: Dict[str, torch.Tensor] = {}

        # Cache gram matrices and HSIC(K, K) for model1
        for i, layer1 in enumerate(self.layers1):
            feat1 = self._features1.get(layer1)
            if feat1 is None:
                continue

            gram1, hsic_kk = self._prepare_gram_and_self_hsic(feat1)
            gram1_cache[layer1] = gram1
            hsic_xx[i] += hsic_kk

        # Cache gram matrices and HSIC(L, L) for model2
        for j, layer2 in enumerate(self.layers2):
            feat2 = self._features2.get(layer2)
            if feat2 is None:
                continue

            gram2, hsic_ll = self._prepare_gram_and_self_hsic(feat2)
            gram2_cache[layer2] = gram2
            hsic_yy[j] += hsic_ll

        # Compute cross-HSIC for all layer pairs
        for i, layer1 in enumerate(self.layers1):
            gram1 = gram1_cache.get(layer1)
            if gram1 is None:
                continue

            for j, layer2 in enumerate(self.layers2):
                gram2 = gram2_cache.get(layer2)
                if gram2 is None:
                    continue

                hsic_kl = hsic(gram1, gram2, self.config.unbiased, self.config.epsilon)
                hsic_xy[i, j] += hsic_kl

    def _compute_cka_matrix(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CKA matrix from accumulated HSIC values.

        Args:
            hsic_xy: Accumulated HSIC(K, L), shape (n_layers1, n_layers2).
            hsic_xx: Accumulated HSIC(K, K), shape (n_layers1,).
            hsic_yy: Accumulated HSIC(L, L), shape (n_layers2,).

        Returns:
            CKA matrix of shape (n_layers1, n_layers2).
        """
        # CKA[i,j] = HSIC_xy[i,j] / sqrt(HSIC_xx[i] * HSIC_yy[j])
        # Clamp to non-negative to handle potential negative unbiased HSIC values
        denominator = torch.sqrt(torch.clamp(hsic_xx.unsqueeze(1) * hsic_yy.unsqueeze(0), min=0.0)) + self.config.epsilon
        return hsic_xy / denominator

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def save_checkpoint(
        self,
        path: Union[str, Path],
        cka_matrix: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save CKA results and configuration to a checkpoint file.

        Args:
            path: Path to save checkpoint.
            cka_matrix: Computed CKA matrix.
            metadata: Optional additional metadata.
        """
        checkpoint = {
            "cka_matrix": cka_matrix.cpu(),
            "model1_info": {
                "name": self.model1_info.name,
                "layers": self.model1_info.layers,
            },
            "model2_info": {
                "name": self.model2_info.name,
                "layers": self.model2_info.layers,
            },
            "config": {
                "kernel": self.config.kernel,
                "sigma": self.config.sigma,
                "unbiased": self.config.unbiased,
                "epsilon": self.config.epsilon,
            },
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a saved CKA checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Dict containing cka_matrix, model info, config, and metadata.
        """
        return torch.load(path, weights_only=False)

    # =========================================================================
    # CALLABLE API
    # =========================================================================

    def __call__(
        self,
        dataloader: DataLoader,
        dataloader2: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute CKA with automatic hook management.

        This is a convenience method that handles context management automatically.

        Args:
            dataloader: DataLoader for model1.
            dataloader2: Optional DataLoader for model2.
            **kwargs: Additional arguments passed to compare().

        Returns:
            CKA similarity matrix.
        """
        with self:
            return self.compare(dataloader, dataloader2, **kwargs)
