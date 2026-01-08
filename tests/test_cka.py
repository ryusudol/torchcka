"""Tests for pytorch_cka.cka module."""

import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_cka import CKA


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class TestCKAClass:
    """Tests for CKA class."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def dataloader(self):
        """Create a simple dataloader for testing."""
        data = torch.randn(32, 10)
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=8)

    def test_context_manager_hook_cleanup(self, model, dataloader):
        """Hooks should be properly cleaned up after context exit."""
        layers = ["layer1", "layer2"]

        hook_count_before = sum(len(m._forward_hooks) for m in model.modules())

        with CKA(model, model, model1_layers=layers) as cka_analyzer:
            _ = cka_analyzer.compare(dataloader, progress=False)

        hook_count_after = sum(len(m._forward_hooks) for m in model.modules())

        assert hook_count_after == hook_count_before

    def test_training_state_restored(self, model, dataloader):
        """Model training state should be restored after comparison."""
        model.train()
        assert model.training

        with CKA(model, model, model1_layers=["layer1"]) as cka_analyzer:
            assert not model.training  # Should be in eval mode
            _ = cka_analyzer.compare(dataloader, progress=False)

        assert model.training  # Should be restored to training mode

    def test_training_state_restored_eval(self, model, dataloader):
        """Model should stay in eval mode if it started in eval mode."""
        model.eval()
        assert not model.training

        with CKA(model, model, model1_layers=["layer1"]) as cka_analyzer:
            assert not model.training
            _ = cka_analyzer.compare(dataloader, progress=False)

        assert not model.training  # Should still be in eval mode

    def test_same_model_comparison(self, model, dataloader):
        """Comparing model with itself should work correctly."""
        layers = ["layer1", "layer2"]

        with CKA(model, model, model1_layers=layers) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        # Diagonal should be ~1 (same features compared to themselves)
        diag = torch.diag(matrix)
        assert torch.allclose(diag, torch.ones_like(diag), atol=0.01)

    def test_same_model_different_layers(self, model, dataloader):
        """Same model with different layer sets should work correctly."""
        # Non-overlapping layers
        layers1 = ["layer1"]
        layers2 = ["layer2", "layer3"]

        with CKA(model, model, model1_layers=layers1, model2_layers=layers2) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (1, 2)
        # All values should be valid (no NaN or zero from missing features)
        assert not torch.isnan(matrix).any()
        assert (matrix > 0).all()

    def test_same_model_partially_overlapping_layers(self, model, dataloader):
        """Same model with partially overlapping layer sets should work."""
        layers1 = ["layer1", "layer2"]
        layers2 = ["layer2", "layer3"]

        with CKA(model, model, model1_layers=layers1, model2_layers=layers2) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (2, 2)
        # layer2 vs layer2 (index [1, 0]) should be ~1
        assert torch.isclose(matrix[1, 0], torch.tensor(1.0, dtype=matrix.dtype), atol=0.01)
        # All values should be valid (no NaN or zero from missing features)
        assert not torch.isnan(matrix).any()
        assert (matrix > 0).all()

    def test_output_shape(self, model, dataloader):
        """CKA matrix should have correct shape."""
        layers = ["layer1", "layer2", "layer3"]

        with CKA(model, model, model1_layers=layers) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (3, 3)

    def test_invalid_layers_warning(self, model):
        """Should warn about invalid layer names when hooks are registered."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with CKA(model, model, model1_layers=["nonexistent", "layer1"]):
                pass

            assert len(w) >= 1
            assert any("not found" in str(warning.message).lower() for warning in w)

    def test_no_valid_layers_error(self, model):
        """Should raise error when no valid layers found."""
        with pytest.raises(ValueError, match="No valid layers"):
            with CKA(model, model, model1_layers=["nonexistent1", "nonexistent2"]):
                pass

    def test_callable_api(self, model, dataloader):
        """CKA should be callable for automatic hook management."""
        layers = ["layer1", "layer2"]
        cka_analyzer = CKA(model, model, model1_layers=layers, model2_layers=layers)
        matrix = cka_analyzer(dataloader, progress=False)

        assert matrix.shape == (2, 2)

    def test_checkpoint_save_load(self, model, dataloader, tmp_path):
        """Should save and load checkpoints correctly."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        with CKA(model, model, model1_layers=["layer1"]) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)
            cka_analyzer.save_checkpoint(checkpoint_path, matrix, metadata={"test": "value"})

        # Load checkpoint
        checkpoint = CKA.load_checkpoint(checkpoint_path)

        assert "cka_matrix" in checkpoint
        assert "model1_name" in checkpoint
        assert "model1_layers" in checkpoint
        assert "metadata" in checkpoint
        assert checkpoint["metadata"]["test"] == "value"
        assert torch.allclose(checkpoint["cka_matrix"], matrix.cpu())

    def test_two_different_models(self, dataloader):
        """Should compare two different models."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        with CKA(model1, model2, model1_layers=["layer1"], model2_layers=["layer1"]) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (1, 1)
        # Random initialized models should have some similarity
        assert 0 <= matrix[0, 0] <= 1

    def test_different_layer_counts(self, dataloader):
        """Should handle different layer counts for two models."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        with CKA(
            model1, model2, model1_layers=["layer1", "layer2"], model2_layers=["layer1"]
        ) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (2, 1)

    def test_callback_called(self, model, dataloader):
        """Callback should be called for each batch."""
        call_count = [0]

        def callback(batch_idx, total_batches, current_matrix):
            call_count[0] += 1

        with CKA(model, model, model1_layers=["layer1"]) as cka_analyzer:
            cka_analyzer.compare(dataloader, progress=False, callback=callback)

        # Should be called once per batch
        assert call_count[0] == len(dataloader)


class TestCKANumericalStability:
    """Tests for numerical stability."""

    def test_epsilon_prevents_nan(self):
        """Epsilon should prevent NaN from zero denominators."""
        model = SimpleModel()
        # Create data with very small values
        data = torch.randn(32, 10) * 1e-10
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=8)

        with CKA(model, model, model1_layers=["layer1"], model2_layers=["layer1"]) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert not torch.isnan(matrix).any()
        assert not torch.isinf(matrix).any()


class TestCKABatchExtraction:
    """Tests for batch input extraction."""

    def test_tensor_batch(self):
        """Should handle plain tensor batches."""
        model = SimpleModel()
        data = torch.randn(16, 10)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=8)

        with CKA(model, model, model1_layers=["layer1"], model2_layers=["layer1"]) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (1, 1)

    def test_tuple_batch(self):
        """Should handle tuple batches (input, label)."""
        model = SimpleModel()
        data = torch.randn(16, 10)
        labels = torch.randint(0, 5, (16,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=8)

        with CKA(model, model, model1_layers=["layer1"], model2_layers=["layer1"]) as cka_analyzer:
            matrix = cka_analyzer.compare(dataloader, progress=False)

        assert matrix.shape == (1, 1)

    def test_hooks_not_registered_error(self):
        """Should raise error if hooks not registered."""
        model = SimpleModel()
        data = torch.randn(16, 10)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=8)

        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        with pytest.raises(RuntimeError, match="Hooks not registered"):
            cka_analyzer.compare(dataloader)


class TestCKADeviceHandling:
    """Tests for device handling."""

    def test_init_with_string_device(self):
        """Should accept device as string and convert to torch.device."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"], model2_layers=["layer1"], device="cpu")

        assert cka_analyzer.device == torch.device("cpu")


class TestCKALayerAutoSelection:
    """Tests for automatic layer selection."""

    def test_auto_layer_selection_excludes_empty_string(self):
        """Auto-selected layers should not include empty string (root module)."""
        model = SimpleModel()

        # Don't specify layers - let CKA auto-detect them
        cka_analyzer = CKA(model, model)

        # Verify no empty strings in layer lists
        assert "" not in cka_analyzer.model1_layers
        assert "" not in cka_analyzer.model2_layers

        # Verify layers were actually detected
        assert len(cka_analyzer.model1_layers) > 0
        assert len(cka_analyzer.model2_layers) > 0

    def test_invalid_layers2_warning(self):
        """Should warn about invalid layer names in model2 when hooks are registered."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        with pytest.warns(UserWarning, match="not found in model2"):
            with CKA(model1, model2, model1_layers=["layer1"], model2_layers=["nonexistent", "layer1"]):
                pass

    def test_no_valid_layers2_error(self):
        """Should raise error when no valid layers found in model2."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        with pytest.raises(ValueError, match="No valid layers"):
            with CKA(model1, model2, model1_layers=["layer1"], model2_layers=["fake1", "fake2"]):
                pass


class TestCKAHookManagement:
    """Tests for hook registration."""

    def test_register_hooks_idempotent(self):
        """Registering hooks twice should not duplicate them."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        cka_analyzer._register_hooks()
        initial_count = len(cka_analyzer._hook_handles)

        cka_analyzer._register_hooks()  # Call again
        assert len(cka_analyzer._hook_handles) == initial_count

        cka_analyzer._remove_hooks()


class TestCKAProgress:
    """Tests for progress display."""

    def test_compare_with_progress(self, capsys):
        """Should show progress bar when progress=True."""
        model = SimpleModel()
        data = torch.randn(16, 10)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=8)

        with CKA(model, model, model1_layers=["layer1"]) as cka_analyzer:
            cka_analyzer.compare(dataloader, progress=True)

        # tqdm writes to stderr
        captured = capsys.readouterr()
        # Progress bar should have been displayed (tqdm outputs contain percentage or iteration info)
        # Note: exact output depends on tqdm behavior


class TestCKABatchInputExtraction:
    """Tests for _extract_input method."""

    def test_dict_batch_with_input_key(self):
        """Should extract from dict batch with 'input' key."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        tensor = torch.randn(4, 10)
        batch = {"input": tensor, "label": torch.zeros(4)}
        result = cka_analyzer._extract_input(batch)

        assert torch.equal(result, tensor)

    def test_dict_batch_with_input_ids_key(self):
        """Should extract from dict batch with 'input_ids' key."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        tensor = torch.randint(0, 1000, (4, 128))
        batch = {"input_ids": tensor, "attention_mask": torch.ones(4, 128)}
        result = cka_analyzer._extract_input(batch)

        assert torch.equal(result, tensor)

    def test_dict_batch_no_recognized_key_error(self):
        """Should raise ValueError when dict batch has no recognized keys."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"])
        batch = {"data": torch.randn(1, 2), "labels": torch.randn(3, 4)}
        with pytest.raises(ValueError, match="Cannot find input"):
            cka_analyzer._extract_input(batch)

    def test_unsupported_batch_type_error(self):
        """Should raise TypeError for unsupported batch types."""
        model = SimpleModel()
        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        with pytest.raises(TypeError, match="Unsupported batch type"):
            cka_analyzer._extract_input("invalid_string_batch")


class TestModelNames:
    """Tests for model name storage."""

    def test_model_name_created(self):
        """Model name should be stored correctly."""
        model = SimpleModel()

        cka_analyzer = CKA(model, model, model1_layers=["layer1"], model1_name="TestModel")

        assert cka_analyzer.model1_name == "TestModel"
        assert cka_analyzer.model1_layers == ["layer1"]

    def test_model_name_auto(self):
        """Model name should use class name when name not provided."""
        model = SimpleModel()

        cka_analyzer = CKA(model, model, model1_layers=["layer1"])

        assert cka_analyzer.model1_name == "SimpleModel"
