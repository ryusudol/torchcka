"""Tests for pytorch_cka.config module."""

import pytest
import torch

from pytorch_cka.config import CKAConfig


class TestCKAConfigValidation:
    """Tests for CKAConfig validation in __post_init__."""

    def test_invalid_kernel(self):
        """CKAConfig should raise ValueError for invalid kernel type."""
        with pytest.raises(ValueError, match="kernel must be"):
            CKAConfig(kernel="not-linear-nor-rbf")

    def test_non_positive_epsilon(self):
        """CKAConfig should raise ValueError for non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            CKAConfig(epsilon=0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            CKAConfig(epsilon=-0.001)

    def test_non_positive_sigma(self):
        """CKAConfig should raise ValueError for non-positive sigma."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            CKAConfig(sigma=0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            CKAConfig(sigma=-1.0)

    def test_device_string_conversion(self):
        """CKAConfig should convert device string to torch.device."""
        config = CKAConfig(device="cpu")
        assert isinstance(config.device, torch.device)
        assert config.device == torch.device("cpu")

    def test_valid_config_defaults(self):
        """CKAConfig with defaults should not raise."""
        config = CKAConfig()
        assert config.kernel == "linear"
        assert config.epsilon == 1e-6
        assert config.unbiased is True
