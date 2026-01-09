"""Main CKA class for comparing neural network representations.

This module provides the CKA class for computing Centered Kernel Alignment
between layers of PyTorch models with proper hook management and memory safety.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.types import Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import EPSILON, compute_gram_matrix, hsic
from .utils import (
    FeatureCache,
    get_device,
    unwrap_model,
    validate_batch_size,
)


class CKA:
    """Centered Kernel Alignment for comparing neural network representations.

    This class provides a context-manager-based API for safe hook management
    and efficient CKA computation between model layers.

    Example:
        >>> with CKA(model1, model2, model1_layers=["layer1", "layer2"]) as cka:
        ...     matrix = cka.compare(dataloader)
        ...     print(matrix)
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_name: Optional[str] = None,
        model2_name: Optional[str] = None,
        model1_layers: Optional[List[str]] = None,
        model2_layers: Optional[List[str]] = None,
        device: Device = None,
    ) -> None:
        """Initialize CKA analyzer.

        Args:
            model1: First model to compare.
            model2: Second model to compare.
            model1_layers: Layers to hook in model1. If None, uses all layers.
            model2_layers: Layers to hook in model2. If None, uses all layers.
            model1_name: Display name for model1.
            model2_name: Display name for model2.
            device: Device for computation. If None, auto-detects from model.

        Raises:
            ValueError: If no valid layers are found.
        """
        # Unwrap DataParallel/DDP
        self.model1 = unwrap_model(model1)
        self.model2 = unwrap_model(model2)

        self.device = torch.device(device) if device else get_device(self.model1)

        # Get all layers if not specified
        if not model1_layers:
            model1_layers = [name for name, _ in self.model1.named_modules() if name]
            if len(model1_layers) > 150:
                warnings.warn(
                    f"Model1 has {len(model1_layers)} layers. "
                    "Consider specifying layers explicitly for faster computation."
                )
        if not model2_layers:
            model2_layers = [name for name, _ in self.model2.named_modules() if name]
            if len(model2_layers) > 150:
                warnings.warn(
                    f"Model2 has {len(model2_layers)} layers. "
                    "Consider specifying layers explicitly for faster computation."
                )

        self.model1_layers = model1_layers
        self.model2_layers = model2_layers

        self.model1_name = model1_name or self.model1.__class__.__name__
        self.model2_name = model2_name or self.model2.__class__.__name__

        self._is_same_model = self.model1 is self.model2
        self._is_same_layers = (
            self._is_same_model and self.model1_layers == self.model2_layers
        )

        self._features1 = FeatureCache(detach=True)
        # _features1 stores features for the union of model1_layers and model2_layers when _is_same_model is True
        self._features2 = self._features1 if self._is_same_model else FeatureCache(detach=True)

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._model1_training: Optional[bool] = None
        self._model2_training: Optional[bool] = None


    # =========================================================================
    # CONTEXT MANAGER PROTOCOL
    # =========================================================================

    def __enter__(self) -> "CKA":
        """Enter context: register hooks and set eval mode."""
        self._register_hooks()
        self._save_training_state()
        self.model1.eval()
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
            # Handle HuggingFace ModelOutput objects (e.g., BaseModelOutput)
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state

            if not isinstance(output, torch.Tensor):
                return
            cache.store(layer_name, output)

        return hook


    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        if self._hook_handles:
            return

        if self._is_same_model:
            all_layers = set(self.model1_layers) | set(self.model2_layers)
            found = set()
            for name, module in self.model1.named_modules():
                if name in all_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features1, name)
                    )
                    self._hook_handles.append(handle)
                    found.add(name)

            found1 = found & set(self.model1_layers)
            found2 = found & set(self.model2_layers)

            missing1 = set(self.model1_layers) - found1
            if missing1:
                warnings.warn(f"Layers not found in model: {sorted(missing1)}")

            if not found1:
                raise ValueError(
                    "No valid layers found in model1. "
                    "Use model.named_modules() to see available layers."
                )
            if not found2:
                raise ValueError(
                    "No valid layers found in model2. "
                    "Use model.named_modules() to see available layers."
                )
        else:
            # Register hooks for model1
            found1 = set()
            for name, module in self.model1.named_modules():
                if name in self.model1_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features1, name)
                    )
                    self._hook_handles.append(handle)
                    found1.add(name)

            missing1 = set(self.model1_layers) - found1
            if missing1:
                warnings.warn(f"Layers not found in model1: {sorted(missing1)}")

            # Register hooks for model2
            found2 = set()
            for name, module in self.model2.named_modules():
                if name in self.model2_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features2, name)
                    )
                    self._hook_handles.append(handle)
                    found2.add(name)

            missing2 = set(self.model2_layers) - found2
            if missing2:
                warnings.warn(f"Layers not found in model2: {sorted(missing2)}")

            # Raise if no valid layers found in either model
            if not found1:
                raise ValueError(
                    "No valid layers found in model1. "
                    "Use model.named_modules() to see available layers."
                )
            if not found2:
                raise ValueError(
                    "No valid layers found in model2. "
                    "Use model.named_modules() to see available layers."
                )


    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


    def _save_training_state(self) -> None:
        """Save models' training state for later restoration."""
        self._model1_training = self.model1.training
        self._model2_training = self.model2.training


    def _restore_training_state(self) -> None:
        """Restore models' training state."""
        if self._model1_training is not None:
            self.model1.train(self._model1_training)
        if self._model2_training is not None:
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
            CKA similarity matrix of shape (len(model1_layers), len(model2_layers)).

        Raises:
            RuntimeError: If hooks are not registered (use context manager).
        """
        if not self._hook_handles:
            raise RuntimeError(
                "Hooks not registered. Use 'with CKA(...) as cka:' context manager "
                "or call _register_hooks() first."
            )

        if dataloader2 is None:
            dataloader2 = dataloader

        n_layers1 = len(self.model1_layers)
        n_layers2 = len(self.model2_layers)

        hsic_xy = torch.zeros(n_layers1, n_layers2, device=self.device)
        hsic_xx = torch.zeros(n_layers1, device=self.device)
        hsic_yy = torch.zeros(n_layers2, device=self.device)

        total_batches = min(len(dataloader), len(dataloader2))
        iterator = zip(dataloader, dataloader2)

        if progress:
            iterator = tqdm(iterator, total=total_batches, desc="Computing CKA")

        with torch.no_grad():
            for batch_idx, (batch1, batch2) in enumerate(iterator):
                self._features1.clear()
                if not self._is_same_model:
                    self._features2.clear()

                x1 = self._extract_input(batch1)
                validate_batch_size(x1.shape[0])
                x1 = x1.to(self.device)

                self.model1(x1)

                if not self._is_same_model:
                    x2 = self._extract_input(batch2)
                    x2 = x2.to(self.device)
                    self.model2(x2)

                self._accumulate_hsic(hsic_xy, hsic_xx, hsic_yy)

                if callback is not None:
                    current_cka = self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)
                    callback(batch_idx, total_batches, current_cka)

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
            for key in ("input", "inputs", "x", "image", "images", "input_ids", "pixel_values"):
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

        Handles flattening and gram computation.

        Args:
            feat: Feature tensor from a layer, shape (B, ...).

        Returns:
            Tuple of (gram_matrix, self_hsic_value).
        """
        if feat.dim() > 2:
            feat = feat.flatten(1)

        gram = compute_gram_matrix(feat)
        hsic_self = hsic(gram, gram)

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
        if self._is_same_layers:
            self._accumulate_hsic_symmetric(hsic_xy, hsic_xx, hsic_yy)
            return

        if self._is_same_model:
            gram_cache: Dict[str, torch.Tensor] = {}
            hsic_cache: Dict[str, torch.Tensor] = {}

            all_layers = set(self.model1_layers) | set(self.model2_layers)
            for layer in all_layers:
                feat = self._features1.get(layer)
                if feat is None:
                    continue

                gram, hsic_self = self._prepare_gram_and_self_hsic(feat)
                gram_cache[layer] = gram
                hsic_cache[layer] = hsic_self

            for i, layer1 in enumerate(self.model1_layers):
                if layer1 in hsic_cache:
                    hsic_xx[i] += hsic_cache[layer1]

            for j, layer2 in enumerate(self.model2_layers):
                if layer2 in hsic_cache:
                    hsic_yy[j] += hsic_cache[layer2]

            for i, layer1 in enumerate(self.model1_layers):
                gram1 = gram_cache.get(layer1)
                if gram1 is None:
                    continue

                for j, layer2 in enumerate(self.model2_layers):
                    gram2 = gram_cache.get(layer2)
                    if gram2 is None:
                        continue

                    hsic_kl = hsic(gram1, gram2)
                    hsic_xy[i, j] += hsic_kl
        else:
            gram1_cache: Dict[str, torch.Tensor] = {}
            gram2_cache: Dict[str, torch.Tensor] = {}

            for i, layer1 in enumerate(self.model1_layers):
                feat1 = self._features1.get(layer1)
                if feat1 is None:
                    continue

                gram1, hsic_kk = self._prepare_gram_and_self_hsic(feat1)
                gram1_cache[layer1] = gram1
                hsic_xx[i] += hsic_kk

            for j, layer2 in enumerate(self.model2_layers):
                feat2 = self._features2.get(layer2)
                if feat2 is None:
                    continue

                gram2, hsic_ll = self._prepare_gram_and_self_hsic(feat2)
                gram2_cache[layer2] = gram2
                hsic_yy[j] += hsic_ll

            for i, layer1 in enumerate(self.model1_layers):
                gram1 = gram1_cache.get(layer1)
                if gram1 is None:
                    continue

                for j, layer2 in enumerate(self.model2_layers):
                    gram2 = gram2_cache.get(layer2)
                    if gram2 is None:
                        continue

                    hsic_kl = hsic(gram1, gram2)
                    hsic_xy[i, j] += hsic_kl


    def _accumulate_hsic_symmetric(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> None:
        """Accumulate HSIC values for symmetric case (same model + same layers).

        Optimizations:
        - hsic_xx == hsic_yy (compute once)
        - hsic_xy is symmetric (compute upper triangle only)

        Args:
            hsic_xy: Accumulator for HSIC(K, L), shape (n_layers, n_layers).
            hsic_xx: Accumulator for HSIC(K, K), shape (n_layers,).
            hsic_yy: Accumulator for HSIC(L, L), shape (n_layers,).
        """
        gram_cache: Dict[str, torch.Tensor] = {}

        for i, layer in enumerate(self.model1_layers):
            feat = self._features1.get(layer)
            if feat is None:
                continue

            gram, hsic_self = self._prepare_gram_and_self_hsic(feat)
            gram_cache[layer] = gram
            hsic_xx[i] += hsic_self
            hsic_yy[i] += hsic_self

        for i, layer1 in enumerate(self.model1_layers):
            gram1 = gram_cache.get(layer1)
            if gram1 is None:
                continue

            for j in range(i, len(self.model1_layers)):
                layer2 = self.model1_layers[j]
                gram2 = gram_cache.get(layer2)
                if gram2 is None:
                    continue

                hsic_kl = hsic(gram1, gram2)
                hsic_xy[i, j] += hsic_kl
                if i != j:
                    hsic_xy[j, i] += hsic_kl


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
        denominator = torch.sqrt(torch.clamp(hsic_xx.unsqueeze(1) * hsic_yy.unsqueeze(0), min=0.0)) + EPSILON
        return hsic_xy / denominator


    def export(self, cka_matrix: torch.Tensor) -> Dict[str, Any]:
        """Export CKA results as a dictionary.

        Args:
            cka_matrix: Computed CKA matrix from compare().

        Returns:
            Dictionary containing model names, layer names, and CKA matrix.
        """
        return {
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "model1_layers": self.model1_layers,
            "model2_layers": self.model2_layers,
            "cka_matrix": cka_matrix,
        }

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
