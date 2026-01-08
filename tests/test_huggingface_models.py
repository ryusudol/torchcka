"""Tests for pytorch_cka compatibility with HuggingFace transformers models.

Tests LLM architectures:
- BERT (encoder-only)
- GPT-2 (decoder-only)
- T5 (encoder-decoder)

All models use config-based initialization with small sizes to avoid downloading
pretrained weights and to keep tests fast.
"""

import pytest
import torch

try:
    from transformers import (
        BertConfig,
        BertModel,
        GPT2Config,
        GPT2Model,
        T5Config,
        T5EncoderModel,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from pytorch_cka import CKA

from .helpers import get_sample_layers


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def bert_model():
    """Small BERT model with random weights."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=128,
    )
    return BertModel(config)


@pytest.fixture
def gpt2_model():
    """Small GPT-2 model with random weights."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    config = GPT2Config(
        n_embd=256,
        n_layer=2,
        n_head=4,
        vocab_size=1000,
        n_positions=128,
    )
    return GPT2Model(config)


@pytest.fixture
def t5_encoder():
    """Small T5 encoder-only model with random weights."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    config = T5Config(
        d_model=256,
        d_ff=512,
        num_layers=2,
        num_heads=4,
        vocab_size=1000,
    )
    return T5EncoderModel(config)


# ============================================================================
# TEST CLASSES
# ============================================================================


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestBERT:
    """Tests for BERT (encoder-only architecture)."""

    def test_bert_basic_compatibility(self, bert_model, text_dataloader):
        """BERT should work with CKA feature extraction."""
        layers = get_sample_layers(bert_model, max_layers=5)

        with CKA(bert_model, bert_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_bert_encoder_layers(self, bert_model, text_dataloader):
        """BERT encoder layers should be accessible."""
        encoder_layers = [
            name
            for name, _ in bert_model.named_modules()
            if "encoder.layer" in name and name.count(".") == 2
        ]

        if encoder_layers:
            with CKA(bert_model, bert_model, model1_layers=encoder_layers, model2_layers=encoder_layers) as cka:
                matrix = cka.compare(text_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_bert_attention_layers(self, bert_model, text_dataloader):
        """BERT attention layers should be properly handled."""
        attention_layers = [
            name
            for name, _ in bert_model.named_modules()
            if "attention" in name.lower() and "self" in name.lower()
        ][:2]

        if attention_layers:
            with CKA(bert_model, bert_model, model1_layers=attention_layers, model2_layers=attention_layers) as cka:
                matrix = cka.compare(text_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_bert_dict_batch_extraction(self, bert_model, text_dataloader):
        """BERT should correctly extract input from dict batches."""
        layers = get_sample_layers(bert_model, max_layers=3)

        with CKA(bert_model, bert_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)

    def test_bert_hook_cleanup(self, bert_model, text_dataloader):
        """BERT hooks should be cleaned up after context exit."""
        layers = get_sample_layers(bert_model, max_layers=3)
        hooks_before = sum(len(m._forward_hooks) for m in bert_model.modules())

        with CKA(bert_model, bert_model, model1_layers=layers, model2_layers=layers) as cka:
            _ = cka.compare(text_dataloader, progress=False)

        hooks_after = sum(len(m._forward_hooks) for m in bert_model.modules())
        assert hooks_after == hooks_before


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestGPT2:
    """Tests for GPT-2 (decoder-only architecture)."""

    def test_gpt2_basic_compatibility(self, gpt2_model, text_dataloader):
        """GPT-2 should work with CKA feature extraction."""
        layers = get_sample_layers(gpt2_model, max_layers=5)

        with CKA(gpt2_model, gpt2_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_gpt2_transformer_blocks(self, gpt2_model, text_dataloader):
        """GPT-2 transformer blocks should be accessible."""
        block_layers = [
            name for name, _ in gpt2_model.named_modules() if name.startswith("h.")
        ][:3]

        if block_layers:
            with CKA(gpt2_model, gpt2_model, model1_layers=block_layers, model2_layers=block_layers) as cka:
                matrix = cka.compare(text_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_gpt2_attention_layers(self, gpt2_model, text_dataloader):
        """GPT-2 attention layers should be properly handled."""
        attention_layers = [
            name for name, _ in gpt2_model.named_modules() if "attn" in name.lower()
        ][:2]

        if attention_layers:
            with CKA(gpt2_model, gpt2_model, model1_layers=attention_layers, model2_layers=attention_layers) as cka:
                matrix = cka.compare(text_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_gpt2_self_comparison_diagonal(self, gpt2_model, text_dataloader):
        """GPT-2 self-comparison diagonal should be approximately 1.0."""
        layers = get_sample_layers(gpt2_model, max_layers=3)

        with CKA(gpt2_model, gpt2_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestT5Encoder:
    """Tests for T5 encoder (using T5EncoderModel to avoid decoder requirements)."""

    def test_t5_encoder_basic(self, t5_encoder, text_dataloader):
        """T5 encoder should work with CKA feature extraction."""
        layers = get_sample_layers(t5_encoder, max_layers=5)

        with CKA(t5_encoder, t5_encoder, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()

    def test_t5_encoder_block_layers(self, t5_encoder, text_dataloader):
        """T5 encoder block layers should be accessible."""
        block_layers = [
            name for name, _ in t5_encoder.named_modules() if "block" in name
        ][:3]

        if block_layers:
            with CKA(t5_encoder, t5_encoder, model1_layers=block_layers, model2_layers=block_layers) as cka:
                matrix = cka.compare(text_dataloader, progress=False)

            assert not torch.isnan(matrix).any()

    def test_t5_encoder_self_comparison_diagonal(self, t5_encoder, text_dataloader):
        """T5 encoder self-comparison diagonal should be approximately 1.0."""
        layers = get_sample_layers(t5_encoder, max_layers=3)

        with CKA(t5_encoder, t5_encoder, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        diagonal = torch.diag(matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.05)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestCrossArchitectureComparison:
    """Tests for comparing different LLM architectures."""

    def test_bert_vs_gpt2_comparison(self, bert_model, gpt2_model, text_dataloader):
        """Should compare BERT (encoder) with GPT-2 (decoder)."""
        bert_layers = get_sample_layers(bert_model, max_layers=3)
        gpt2_layers = get_sample_layers(gpt2_model, max_layers=3)

        with CKA(
            bert_model, gpt2_model, model1_layers=bert_layers, model2_layers=gpt2_layers
        ) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(bert_layers), len(gpt2_layers))
        assert not torch.isnan(matrix).any()
        # With random init, CKA can be negative (no shared structure)
        assert (matrix >= -1 - 1e-6).all()
        assert (matrix <= 1 + 1e-6).all()

    def test_two_berts_different_init(self, text_dataloader):
        """Two BERTs with different init should produce valid CKA."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")

        config = BertConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
        )
        model1 = BertModel(config)
        model2 = BertModel(config)

        layers = get_sample_layers(model1, max_layers=3)

        with CKA(model1, model2, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert matrix.shape == (len(layers), len(layers))
        assert not torch.isnan(matrix).any()


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestNumericalStability:
    """Tests for numerical stability with LLM architectures."""

    def test_no_nan_inf_bert(self, bert_model, text_dataloader):
        """BERT should produce valid CKA values without NaN/Inf."""
        layers = get_sample_layers(bert_model, max_layers=3)

        with CKA(bert_model, bert_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert not torch.isnan(matrix).any(), "BERT produced NaN values"
        assert not torch.isinf(matrix).any(), "BERT produced Inf values"

    def test_no_nan_inf_gpt2(self, gpt2_model, text_dataloader):
        """GPT-2 should produce valid CKA values without NaN/Inf."""
        layers = get_sample_layers(gpt2_model, max_layers=3)

        with CKA(gpt2_model, gpt2_model, model1_layers=layers, model2_layers=layers) as cka:
            matrix = cka.compare(text_dataloader, progress=False)

        assert not torch.isnan(matrix).any(), "GPT-2 produced NaN values"
        assert not torch.isinf(matrix).any(), "GPT-2 produced Inf values"

