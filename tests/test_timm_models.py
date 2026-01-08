"""Tests for pytorch_cka compatibility with timm models.

Tests modern CNN architectures (ConvNeXt V2, MobileNetV4) from the timm library.
These represent state-of-the-art CNN designs released in 2023-2024.
"""

import pytest
import torch
import torch.nn as nn

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from pytorch_cka import CKA

from .helpers import get_sample_layers, get_layers_by_type


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def convnextv2_tiny():
    """ConvNeXt V2 Tiny with random weights."""
    if not TIMM_AVAILABLE:
        pytest.skip("timm not installed")
    return timm.create_model("convnextv2_tiny", pretrained=False)


@pytest.fixture
def mobilenetv4_small():
    """MobileNetV4 Small with random weights."""
    if not TIMM_AVAILABLE:
        pytest.skip("timm not installed")
    return timm.create_model("mobilenetv4_conv_small", pretrained=False)


@pytest.fixture
def convnext_tiny_torchvision():
    """ConvNeXt Tiny from torchvision for comparison."""
    from torchvision import models

    return models.convnext_tiny(weights=None)


# ============================================================================
# TEST CLASSES
# ============================================================================


@pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not installed")
class TestModernCNNs:
    """Tests for modern CNN architectures from timm (ConvNeXt V2, MobileNetV4)."""

    def test_convnextv2_tiny_basic_compatibility(
        self, convnextv2_tiny, image_dataloader
    ):
        """ConvNeXt V2 Tiny should work with CKA feature extraction."""
        layers = get_sample_layers(convnextv2_tiny, max_layers=5)

        with CKA(convnextv2_tiny, convnextv2_tiny, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_convnextv2_layernorm_layers(self, convnextv2_tiny, image_dataloader):
        """ConvNeXt V2 LayerNorm layers should be properly handled."""
        ln_layers = get_layers_by_type(convnextv2_tiny, nn.LayerNorm)

        if len(ln_layers) > 3:
            ln_layers = ln_layers[:3]

        if ln_layers:
            with CKA(convnextv2_tiny, convnextv2_tiny, model1_layers=ln_layers, model2_layers=ln_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_convnextv2_self_comparison_diagonal(
        self, convnextv2_tiny, image_dataloader
    ):
        """ConvNeXt V2 self-comparison diagonal should be approximately 1.0."""
        layers = get_sample_layers(convnextv2_tiny, max_layers=3)

        with CKA(convnextv2_tiny, convnextv2_tiny, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)

    def test_convnextv2_hook_cleanup(self, convnextv2_tiny, image_dataloader):
        """ConvNeXt V2 hooks should be cleaned up after context exit."""
        layers = get_sample_layers(convnextv2_tiny, max_layers=3)
        hooks_before = sum(len(m._forward_hooks) for m in convnextv2_tiny.modules())

        with CKA(convnextv2_tiny, convnextv2_tiny, model1_layers=layers, model2_layers=layers) as cka:
            _ = cka.compare(image_dataloader, progress=False)

        hooks_after = sum(len(m._forward_hooks) for m in convnextv2_tiny.modules())
        assert hooks_after == hooks_before

    def test_mobilenetv4_small_basic_compatibility(
        self, mobilenetv4_small, image_dataloader
    ):
        """MobileNetV4 Small should work with CKA feature extraction."""
        layers = get_sample_layers(mobilenetv4_small, max_layers=5)

        with CKA(mobilenetv4_small, mobilenetv4_small, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_mobilenetv4_conv_layers(self, mobilenetv4_small, image_dataloader):
        """MobileNetV4 convolutional layers should be properly handled."""
        conv_layers = get_layers_by_type(mobilenetv4_small, nn.Conv2d)

        if len(conv_layers) > 3:
            conv_layers = conv_layers[:3]

        if conv_layers:
            with CKA(mobilenetv4_small, mobilenetv4_small, model1_layers=conv_layers, model2_layers=conv_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_mobilenetv4_self_comparison_diagonal(
        self, mobilenetv4_small, image_dataloader
    ):
        """MobileNetV4 self-comparison diagonal should be approximately 1.0."""
        layers = get_sample_layers(mobilenetv4_small, max_layers=3)

        with CKA(mobilenetv4_small, mobilenetv4_small, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)


@pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not installed")
class TestCrossLibraryComparison:
    """Tests for comparing models across timm and torchvision."""

    def test_convnextv2_vs_convnext_v1(
        self, convnextv2_tiny, convnext_tiny_torchvision, image_dataloader
    ):
        """Should compare ConvNeXt V2 (timm) with ConvNeXt V1 (torchvision)."""
        v2_layers = get_sample_layers(convnextv2_tiny, max_layers=3)
        v1_layers = get_sample_layers(convnext_tiny_torchvision, max_layers=3)

        with CKA(
            convnextv2_tiny,
            convnext_tiny_torchvision,
            model1_layers=v2_layers,
            model2_layers=v1_layers,
        ) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(v2_layers), len(v1_layers))
        assert not torch.isnan(matrix).any()
        # With random init, CKA can be negative (no shared structure)
        assert (matrix >= -1 - 1e-6).all()
        assert (matrix <= 1 + 1e-6).all()

    def test_two_convnextv2_different_init(self, image_dataloader):
        """Two ConvNeXt V2 with different init should produce valid CKA."""
        if not TIMM_AVAILABLE:
            pytest.skip("timm not installed")

        model1 = timm.create_model("convnextv2_tiny", pretrained=False)
        model2 = timm.create_model("convnextv2_tiny", pretrained=False)

        layers = get_sample_layers(model1, max_layers=3)

        with CKA(model1, model2, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()


@pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not installed")
class TestNumericalStability:
    """Tests for numerical stability with modern CNNs."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "convnextv2_tiny",
            "mobilenetv4_conv_small",
        ],
    )
    def test_no_nan_inf(self, model_name, image_dataloader):
        """Modern CNNs should produce valid CKA values without NaN/Inf."""
        model = timm.create_model(model_name, pretrained=False)
        layers = get_sample_layers(model, max_layers=3)

        with CKA(model, model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert not torch.isnan(matrix).any(), f"{model_name} produced NaN values"
        assert not torch.isinf(matrix).any(), f"{model_name} produced Inf values"
