"""Tests for pytorch_cka compatibility with torchvision models.

Tests traditional CNNs (ResNet, VGG, AlexNet) and Vision Transformers (ViT, Swin).
"""

import pytest
import torch
import torch.nn as nn
from torchvision import models

from pytorch_cka import CKA

from .helpers import get_sample_layers, get_layers_by_type


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def resnet18():
    """ResNet-18 with random weights."""
    return models.resnet18(weights=None)


@pytest.fixture
def vgg11_bn():
    """VGG-11 with batch normalization and random weights."""
    return models.vgg11_bn(weights=None)


@pytest.fixture
def alexnet():
    """AlexNet with random weights."""
    return models.alexnet(weights=None)


@pytest.fixture
def vit_b_16():
    """ViT-B/16 with random weights."""
    return models.vit_b_16(weights=None)


@pytest.fixture
def swin_t():
    """Swin-T with random weights."""
    return models.swin_t(weights=None)


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestTraditionalCNNs:
    """Tests for traditional CNN architectures (ResNet, VGG, AlexNet)."""

    def test_resnet18_basic_compatibility(self, resnet18, image_dataloader):
        """ResNet-18 should work with CKA feature extraction."""
        layers = ["layer1", "layer2", "layer3", "layer4", "fc"]

        with CKA(resnet18, resnet18, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (5, 5)
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_resnet18_hook_cleanup(self, resnet18, image_dataloader):
        """ResNet-18 hooks should be cleaned up after context exit."""
        layers = ["layer1", "layer2"]
        hooks_before = sum(len(m._forward_hooks) for m in resnet18.modules())

        with CKA(resnet18, resnet18, model1_layers=layers, model2_layers=layers) as cka:
            _ = cka.compare(image_dataloader, progress=False)

        hooks_after = sum(len(m._forward_hooks) for m in resnet18.modules())
        assert hooks_after == hooks_before

    def test_resnet18_self_comparison_diagonal(self, resnet18, image_dataloader):
        """ResNet-18 self-comparison diagonal should be approximately 1.0."""
        layers = ["layer1", "layer2", "layer3"]

        with CKA(resnet18, resnet18, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)

    def test_vgg11_bn_basic_compatibility(self, vgg11_bn, image_dataloader):
        """VGG-11-BN should work with CKA feature extraction."""
        layers = ["features.3", "features.7", "features.14", "classifier.0"]

        with CKA(vgg11_bn, vgg11_bn, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (4, 4)
        assert not torch.isnan(matrix).any()

    def test_alexnet_basic_compatibility(self, alexnet, image_dataloader):
        """AlexNet should work with CKA feature extraction."""
        layers = ["features.0", "features.3", "features.6", "classifier.1"]

        with CKA(alexnet, alexnet, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (4, 4)
        assert not torch.isnan(matrix).any()


class TestTorchvisionTransformers:
    """Tests for Vision Transformer architectures from torchvision (ViT, Swin)."""

    def test_vit_b_16_basic_compatibility(self, vit_b_16, image_dataloader):
        """ViT-B/16 should work with CKA feature extraction."""
        layers = get_sample_layers(vit_b_16, max_layers=5)

        with CKA(vit_b_16, vit_b_16, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_vit_attention_layers(self, vit_b_16, image_dataloader):
        """ViT attention layers should be properly handled."""
        attention_layers = get_layers_by_type(vit_b_16, nn.MultiheadAttention)

        if len(attention_layers) > 2:
            attention_layers = attention_layers[:2]

        if attention_layers:
            with CKA(vit_b_16, vit_b_16, model1_layers=attention_layers, model2_layers=attention_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert matrix.shape == (len(attention_layers), len(attention_layers))
            assert not torch.isnan(matrix).any()

    def test_vit_encoder_blocks(self, vit_b_16, image_dataloader):
        """ViT encoder blocks should be accessible."""
        encoder_layers = [
            name
            for name, _ in vit_b_16.named_modules()
            if "encoder_layer" in name and name.count(".") == 2
        ][:3]

        if encoder_layers:
            with CKA(vit_b_16, vit_b_16, model1_layers=encoder_layers, model2_layers=encoder_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_swin_t_basic_compatibility(self, swin_t, image_dataloader):
        """Swin-T should work with CKA feature extraction."""
        layers = get_sample_layers(swin_t, max_layers=5)

        with CKA(swin_t, swin_t, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()

    def test_swin_window_attention(self, swin_t, image_dataloader):
        """Swin-T window attention layers should be accessible."""
        layers = [
            name
            for name, module in swin_t.named_modules()
            if "attn" in name.lower() or "norm" in name.lower()
        ][:3]

        if layers:
            with CKA(swin_t, swin_t, model1_layers=layers, model2_layers=layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert not torch.isnan(matrix).any()


class TestCrossModelComparison:
    """Tests for comparing different model architectures."""

    # TODO: Fix numerical instability causing anomalous CKA values
    # def test_resnet_vs_vgg_comparison(self, resnet18, vgg11_bn, image_dataloader):
    #     """Should compare ResNet-18 with VGG-11-BN."""
    #     resnet_layers = ["layer1", "layer2", "layer4"]
    #     vgg_layers = ["features.3", "features.7", "features.21"]
    #
    #     with CKA(
    #         resnet18, vgg11_bn, model1_layers=resnet_layers, model2_layers=vgg_layers
    #     ) as cka:
    #         matrix = cka.compare(image_dataloader, progress=False)
    #
    #     assert matrix.shape == (3, 3)
    #     assert not torch.isnan(matrix).any()
    #     # Note: CKA with random init can be negative (no shared structure)
    #     assert (matrix >= -1 - 1e-6).all()
    #     assert (matrix <= 1 + 1e-6).all()

    def test_cnn_vs_transformer_comparison(
        self, resnet18, vit_b_16, image_dataloader
    ):
        """Should compare CNN (ResNet-18) with Transformer (ViT-B/16)."""
        resnet_layers = ["layer1", "layer4"]
        vit_layers = get_sample_layers(vit_b_16, max_layers=2)

        with CKA(
            resnet18, vit_b_16, model1_layers=resnet_layers, model2_layers=vit_layers
        ) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert matrix.shape == (2, 2)
        assert not torch.isnan(matrix).any()

    def test_two_resnets_different_init(self, image_dataloader):
        """Two ResNets with different random init should produce valid CKA."""
        model1 = models.resnet18(weights=None)
        model2 = models.resnet18(weights=None)

        layers = ["layer1", "layer4"]

        with CKA(model1, model2, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        # With random init, CKA can be negative (no shared structure)
        assert -1 - 1e-6 <= matrix[0, 0] <= 1 + 1e-6
        assert -1 - 1e-6 <= matrix[1, 1] <= 1 + 1e-6
        assert not torch.isnan(matrix).any()


class TestLayerTypeVariety:
    """Tests for different layer types across architectures."""

    def test_conv2d_layers(self, resnet18, image_dataloader):
        """Conv2d layers should be properly handled."""
        conv_layers = get_layers_by_type(resnet18, nn.Conv2d)[:3]

        with CKA(resnet18, resnet18, model1_layers=conv_layers, model2_layers=conv_layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert not torch.isnan(matrix).any()

    def test_batchnorm_layers(self, resnet18, image_dataloader):
        """BatchNorm2d layers should be properly handled."""
        bn_layers = get_layers_by_type(resnet18, nn.BatchNorm2d)[:3]

        with CKA(resnet18, resnet18, model1_layers=bn_layers, model2_layers=bn_layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert not torch.isnan(matrix).any()

    def test_layernorm_layers(self, vit_b_16, image_dataloader):
        """LayerNorm layers (common in transformers) should be handled."""
        ln_layers = get_layers_by_type(vit_b_16, nn.LayerNorm)[:3]

        if ln_layers:
            with CKA(vit_b_16, vit_b_16, model1_layers=ln_layers, model2_layers=ln_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_multihead_attention_layers(self, vit_b_16, image_dataloader):
        """MultiheadAttention layers should handle tuple outputs correctly."""
        mha_layers = get_layers_by_type(vit_b_16, nn.MultiheadAttention)[:2]

        if mha_layers:
            with CKA(vit_b_16, vit_b_16, model1_layers=mha_layers, model2_layers=mha_layers) as cka:
                matrix = cka.compare(image_dataloader, progress=False)

            assert matrix.shape == (len(mha_layers), len(mha_layers))
            assert not torch.isnan(matrix).any()


class TestNumericalStability:
    """Tests for numerical stability across different model architectures."""

    @pytest.mark.parametrize(
        "model_name,model_fn",
        [
            ("resnet18", lambda: models.resnet18(weights=None)),
            ("vgg11_bn", lambda: models.vgg11_bn(weights=None)),
            ("vit_b_16", lambda: models.vit_b_16(weights=None)),
            ("swin_t", lambda: models.swin_t(weights=None)),
        ],
    )
    def test_no_nan_inf(self, model_name, model_fn, image_dataloader):
        """All models should produce valid CKA values without NaN/Inf."""
        model = model_fn()
        layers = get_sample_layers(model, max_layers=3)

        with CKA(model, model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(image_dataloader, progress=False)

        assert not torch.isnan(matrix).any(), f"{model_name} produced NaN values"
        assert not torch.isinf(matrix).any(), f"{model_name} produced Inf values"
