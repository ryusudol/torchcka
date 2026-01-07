"""Tests for pytorch_cka.utils module."""

import pytest
import torch
import torch.nn as nn

from pytorch_cka.utils import (
    FeatureCache,
    eval_mode,
    get_all_layer_names,
    get_device,
    unwrap_model,
    validate_batch_size,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


class NoParamModel(nn.Module):
    """Model with no parameters."""

    def forward(self, x):
        return x * 2


class TestValidateBatchSize:
    """Tests for validate_batch_size function."""

    def test_unbiased_too_small(self):
        """Should raise ValueError for batch size <= 3 in unbiased mode."""
        with pytest.raises(ValueError, match="Unbiased HSIC requires batch size > 3"):
            validate_batch_size(n=3, unbiased=True)

        with pytest.raises(ValueError, match="Unbiased HSIC requires batch size > 3"):
            validate_batch_size(n=1, unbiased=True)

    def test_biased_too_small(self):
        """Should raise ValueError for batch size <= 1 in biased mode."""
        with pytest.raises(ValueError, match="HSIC requires batch size > 1"):
            validate_batch_size(n=1, unbiased=False)

    def test_valid_sizes(self):
        """Should not raise for valid batch sizes."""
        validate_batch_size(n=4, unbiased=True)  # No exception
        validate_batch_size(n=2, unbiased=False)  # No exception


class TestGetAllLayerNames:
    """Tests for get_all_layer_names function."""

    def test_exclude_root(self):
        """Should exclude root module when include_root=False."""
        model = SimpleModel()
        names = get_all_layer_names(model, include_root=False)
        assert "" not in names
        assert "layer1" in names
        assert "layer2" in names

    def test_include_root(self):
        """Should include root module when include_root=True."""
        model = SimpleModel()
        names = get_all_layer_names(model, include_root=True)
        assert "" in names  # Root module has empty string name


class TestGetDevice:
    """Tests for get_device function."""

    def test_model_with_parameters(self):
        """Should return device from model parameters."""
        model = SimpleModel()
        device = get_device(model)
        assert device == torch.device("cpu")

    def test_no_parameters_with_fallback(self):
        """Should return fallback for model with no parameters."""
        model = NoParamModel()
        fallback = torch.device("cpu")
        device = get_device(model, fallback=fallback)
        assert device == fallback

    def test_no_parameters_no_fallback(self):
        """Should default to CPU when no parameters and no fallback."""
        model = NoParamModel()
        device = get_device(model)
        assert device == torch.device("cpu")


class TestUnwrapModel:
    """Tests for unwrap_model function."""

    def test_regular_model(self):
        """Should return model unchanged if not wrapped."""
        model = SimpleModel()
        unwrapped = unwrap_model(model)
        assert unwrapped is model

    def test_data_parallel(self):
        """Should unwrap DataParallel models."""
        model = SimpleModel()
        # DataParallel requires CUDA, so we mock the check
        wrapped = nn.DataParallel(model)
        unwrapped = unwrap_model(wrapped)
        assert unwrapped is model


class TestFeatureCache:
    """Tests for FeatureCache class."""

    def test_store_and_get(self):
        """Should store and retrieve tensors."""
        cache = FeatureCache()
        tensor = torch.randn(5, 10)
        cache.store("layer1", tensor)
        retrieved = cache.get("layer1")
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)

    def test_get_missing(self):
        """Should return None for missing keys."""
        cache = FeatureCache()
        assert cache.get("nonexistent") is None

    def test_clear(self):
        """Should clear all features."""
        cache = FeatureCache()
        cache.store("layer1", torch.randn(5, 10))
        cache.clear()
        assert cache.get("layer1") is None

    def test_items(self):
        """Should iterate over cached features."""
        cache = FeatureCache()
        cache.store("layer1", torch.randn(5, 10))
        cache.store("layer2", torch.randn(5, 20))

        items = list(cache.items())
        assert len(items) == 2
        names = [name for name, _ in items]
        assert "layer1" in names
        assert "layer2" in names

    def test_keys(self):
        """Should iterate over layer names."""
        cache = FeatureCache()
        cache.store("layer1", torch.randn(5, 10))
        cache.store("layer2", torch.randn(5, 20))

        keys = list(cache.keys())
        assert set(keys) == {"layer1", "layer2"}

    def test_len(self):
        """Should return correct count of stored features."""
        cache = FeatureCache()
        assert len(cache) == 0
        cache.store("layer1", torch.randn(5, 10))
        assert len(cache) == 1

    def test_contains(self):
        """Should check if layer name exists in cache."""
        cache = FeatureCache()
        cache.store("layer1", torch.randn(5, 10))

        assert "layer1" in cache
        assert "layer2" not in cache


class TestEvalMode:
    """Tests for eval_mode context manager."""

    def test_sets_eval_mode(self):
        """Should set model to eval mode inside context."""
        model = SimpleModel()
        model.train()

        with eval_mode(model) as m:
            assert not m.training

    def test_restores_training_mode(self):
        """Should restore training mode after context."""
        model = SimpleModel()
        model.train()

        with eval_mode(model):
            pass

        assert model.training

    def test_preserves_eval_mode(self):
        """Should keep eval mode if started in eval."""
        model = SimpleModel()
        model.eval()

        with eval_mode(model):
            pass

        assert not model.training  # Still in eval

    def test_exception_restores_state(self):
        """Should restore state even when exception occurs."""
        model = SimpleModel()
        model.train()

        with pytest.raises(RuntimeError):
            with eval_mode(model):
                assert not model.training
                raise RuntimeError("Test exception")

        # Should be restored despite exception
        assert model.training
