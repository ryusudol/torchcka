"""Tests for pytorch_cka.core module."""

import pytest
import torch

from pytorch_cka.config import CKAConfig
from pytorch_cka.core import (
    center_gram_matrix,
    cka,
    cka_from_gram,
    compute_gram_matrix,
    hsic,
    hsic_biased,
    hsic_unbiased,
    linear_kernel,
    rbf_kernel,
)


class TestLinearKernel:
    """Tests for linear_kernel function."""

    def test_invalid_dims(self):
        """linear_kernel should raise ValueError for non-2D tensors."""
        x_1d = torch.randn(10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            linear_kernel(x_1d)

        x_3d = torch.randn(5, 10, 5)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            linear_kernel(x_3d)

    def test_shape(self):
        """Linear kernel should produce (n, n) gram matrix."""
        x = torch.randn(10, 5)
        K = linear_kernel(x)
        assert K.shape == (10, 10)

    def test_symmetry(self):
        """Linear kernel gram matrix should be symmetric."""
        x = torch.randn(10, 5)
        K = linear_kernel(x)
        assert torch.allclose(K, K.T)

    def test_positive_semidefinite(self):
        """Linear kernel gram matrix should be positive semi-definite."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = linear_kernel(x)
        eigenvalues = torch.linalg.eigvalsh(K)
        # Allow small numerical errors
        assert (eigenvalues >= -1e-6).all()

    def test_computation(self):
        """Linear kernel should compute K = X @ X^T."""
        x = torch.randn(5, 3)
        K = linear_kernel(x)
        expected = x @ x.T
        assert torch.allclose(K, expected)


class TestRBFKernel:
    """Tests for rbf_kernel function."""

    def test_invalid_dims(self):
        """rbf_kernel should raise ValueError for non-2D tensors."""
        x_1d = torch.randn(10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            rbf_kernel(x_1d)

        x_4d = torch.randn(5, 3, 4, 4)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            rbf_kernel(x_4d)

    def test_single_sample(self):
        """rbf_kernel with n=1 should use default sigma."""
        x = torch.randn(1, 5)
        K = rbf_kernel(x, sigma=None)
        assert K.shape == (1, 1)
        assert torch.isclose(K[0, 0], torch.tensor(1.0), atol=1e-6)

    def test_shape(self):
        """RBF kernel should produce (n, n) gram matrix."""
        x = torch.randn(10, 5)
        K = rbf_kernel(x)
        assert K.shape == (10, 10)

    def test_diagonal_ones(self):
        """RBF kernel diagonal should be 1 (k(x,x) = 1)."""
        x = torch.randn(10, 5)
        K = rbf_kernel(x)
        assert torch.allclose(torch.diag(K), torch.ones(10), atol=1e-6)

    def test_bounds(self):
        """RBF kernel values should be in (0, 1]."""
        x = torch.randn(10, 5)
        K = rbf_kernel(x)
        assert (K > 0).all()
        assert (K <= 1 + 1e-6).all()

    def test_symmetry(self):
        """RBF kernel gram matrix should be symmetric."""
        x = torch.randn(10, 5)
        K = rbf_kernel(x)
        assert torch.allclose(K, K.T)

    def test_numerical_stability_constant_features(self):
        """RBF kernel should handle constant features without NaN."""
        x = torch.ones(10, 5)  # All same -> zero variance
        K = rbf_kernel(x, epsilon=1e-10)
        assert not torch.isnan(K).any()
        assert not torch.isinf(K).any()

    def test_custom_sigma(self):
        """RBF kernel should use custom sigma when provided."""
        x = torch.randn(10, 5)
        K1 = rbf_kernel(x, sigma=1.0)
        K2 = rbf_kernel(x, sigma=2.0)
        # Different sigma should give different results
        assert not torch.allclose(K1, K2)


class TestComputeGramMatrix:
    """Tests for compute_gram_matrix function."""

    def test_linear_kernel(self):
        """compute_gram_matrix with linear kernel should match linear_kernel."""
        x = torch.randn(10, 5)
        K1 = compute_gram_matrix(x, kernel="linear")
        K2 = linear_kernel(x)
        assert torch.allclose(K1, K2)

    def test_rbf_kernel(self):
        """compute_gram_matrix with rbf kernel should match rbf_kernel."""
        x = torch.randn(10, 5)
        K1 = compute_gram_matrix(x, kernel="rbf")
        K2 = rbf_kernel(x)
        assert torch.allclose(K1, K2)

    def test_invalid_kernel(self):
        """compute_gram_matrix should raise error for invalid kernel."""
        x = torch.randn(10, 5)
        with pytest.raises(ValueError, match="kernel must be"):
            compute_gram_matrix(x, kernel="invalid")


class TestCenterGramMatrix:
    """Tests for center_gram_matrix function."""

    def test_invalid_dims(self):
        """center_gram_matrix should raise for non-2D tensors."""
        gram_1d = torch.randn(10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            center_gram_matrix(gram_1d)

        gram_3d = torch.randn(5, 10, 10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            center_gram_matrix(gram_3d)

    def test_non_square(self):
        """center_gram_matrix should raise for non-square matrices."""
        gram_rect = torch.randn(10, 15)
        with pytest.raises(ValueError, match="requires square matrix"):
            center_gram_matrix(gram_rect)

    def test_empty_matrix(self):
        """center_gram_matrix should raise for empty matrix."""
        gram_empty = torch.randn(0, 0)
        with pytest.raises(ValueError, match="requires non-empty matrix"):
            center_gram_matrix(gram_empty)

    def test_centered_sum_zero(self):
        """Centered gram matrix rows/cols should sum to ~zero."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = linear_kernel(x)
        Kc = center_gram_matrix(K)
        assert torch.allclose(Kc.sum(dim=0), torch.zeros(10, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(Kc.sum(dim=1), torch.zeros(10, dtype=torch.float64), atol=1e-10)

    def test_symmetry_preserved(self):
        """Centering should preserve symmetry."""
        x = torch.randn(10, 5)
        K = linear_kernel(x)
        Kc = center_gram_matrix(K)
        assert torch.allclose(Kc, Kc.T)


class TestHSIC:
    """Tests for HSIC functions."""

    def test_biased_invalid_dims(self):
        """hsic_biased should raise for non-2D gram matrices."""
        gram_valid = torch.randn(10, 10)
        gram_1d = torch.randn(10)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic_biased(gram_1d, gram_valid)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic_biased(gram_valid, gram_1d)

    def test_biased_shape_mismatch(self):
        """hsic_biased should raise when gram matrix shapes don't match."""
        gram_10x10 = torch.randn(10, 10)
        gram_8x8 = torch.randn(8, 8)

        with pytest.raises(ValueError, match="requires matching shapes"):
            hsic_biased(gram_10x10, gram_8x8)

    def test_biased_non_square(self):
        """hsic_biased should raise for non-square gram matrices."""
        gram_rect = torch.randn(10, 15)

        with pytest.raises(ValueError, match="requires square matrices"):
            hsic_biased(gram_rect, gram_rect)

    def test_unbiased_invalid_dims(self):
        """hsic_unbiased should raise for non-2D tensors."""
        gram_valid = torch.randn(10, 10)
        gram_3d = torch.randn(5, 10, 10)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic_unbiased(gram_3d, gram_valid)

    def test_unbiased_shape_mismatch(self):
        """hsic_unbiased should raise for shape mismatch."""
        gram_10x10 = torch.randn(10, 10)
        gram_5x5 = torch.randn(5, 5)

        with pytest.raises(ValueError, match="requires matching shapes"):
            hsic_unbiased(gram_10x10, gram_5x5)

    def test_unbiased_non_square(self):
        """hsic_unbiased should raise for non-square matrices."""
        gram_rect = torch.randn(10, 15)

        with pytest.raises(ValueError, match="requires square matrices"):
            hsic_unbiased(gram_rect, gram_rect)

    def test_biased_requires_n_gt_1(self):
        """Biased HSIC should raise error for n <= 1."""
        K = torch.randn(1, 1)
        with pytest.raises(ValueError, match="n > 1"):
            hsic_biased(K, K)

    def test_unbiased_requires_n_gt_3(self):
        """Unbiased HSIC should raise error for n <= 3."""
        K = torch.randn(3, 3)
        K = K @ K.T  # Make symmetric
        with pytest.raises(ValueError, match="n > 3"):
            hsic_unbiased(K, K)

    def test_hsic_identical_positive(self):
        """HSIC(K, K) should be positive for non-trivial K."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = linear_kernel(x)
        hsic_val = hsic_unbiased(K, K)
        assert hsic_val > 0

    def test_hsic_different_estimators(self):
        """Biased and unbiased HSIC should give similar values for correlated data."""
        # Use correlated data (same x for both) to ensure positive HSIC
        x = torch.randn(100, 5, dtype=torch.float64)
        Kx = linear_kernel(x)

        hsic_b = hsic_biased(Kx, Kx)
        hsic_u = hsic_unbiased(Kx, Kx)

        # Both should be positive for self-comparison
        assert hsic_b > 0
        assert hsic_u > 0
        # Should be in same order of magnitude
        assert abs(hsic_b - hsic_u) / (abs(hsic_b) + 1e-10) < 1.0

    def test_hsic_wrapper(self):
        """hsic function should dispatch correctly."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = linear_kernel(x)

        h1 = hsic(K, K, unbiased=True)
        h2 = hsic_unbiased(K, K)
        assert torch.allclose(h1, h2)

        h3 = hsic(K, K, unbiased=False)
        h4 = hsic_biased(K, K)
        assert torch.allclose(h3, h4)


class TestCKA:
    """Tests for CKA computation functions."""

    def test_identical_features_gives_one(self):
        """CKA(X, X) should equal 1."""
        x = torch.randn(10, 5, dtype=torch.float64)
        config = CKAConfig(epsilon=1e-10)
        cka_val = cka(x, x, config)
        assert torch.isclose(cka_val, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    def test_cka_bounds(self):
        """CKA should be in [0, 1] for correlated features."""
        # Use correlated features to ensure positive CKA
        x = torch.randn(50, 5, dtype=torch.float64)
        # y is a noisy version of x
        y = x + 0.1 * torch.randn(50, 5, dtype=torch.float64)
        cka_val = cka(x, y)
        assert 0 <= cka_val <= 1 + 1e-6

    def test_cka_symmetry(self):
        """CKA(X, Y) should equal CKA(Y, X)."""
        x = torch.randn(50, 5, dtype=torch.float64)
        y = torch.randn(50, 8, dtype=torch.float64)
        assert torch.isclose(cka(x, y), cka(y, x), atol=1e-10)

    def test_cka_rbf_kernel(self):
        """CKA should work with RBF kernel."""
        # Use larger sample size for stability with unbiased estimator
        x = torch.randn(50, 5, dtype=torch.float64)
        y = torch.randn(50, 5, dtype=torch.float64)
        config = CKAConfig(kernel="rbf")
        cka_val = cka(x, y, config)
        # CKA should be finite
        assert not torch.isnan(cka_val)
        assert not torch.isinf(cka_val)

    def test_cka_flattens_high_dim(self):
        """CKA should automatically flatten >2D tensors."""
        # Use larger batch size for numerical stability with unbiased estimator
        x = torch.randn(50, 3, 4, 4, dtype=torch.float64)  # (B, C, H, W)
        y = torch.randn(50, 5, 2, 2, dtype=torch.float64)
        cka_val = cka(x, y)
        # CKA should be finite (not NaN or inf)
        assert not torch.isnan(cka_val)
        assert not torch.isinf(cka_val)

    def test_numerical_stability_small_values(self):
        """CKA should handle very small feature values."""
        x = torch.randn(10, 5) * 1e-10
        y = torch.randn(10, 5) * 1e-10
        cka_val = cka(x, y)
        assert not torch.isnan(cka_val)
        assert not torch.isinf(cka_val)

    def test_cka_from_gram(self):
        """cka_from_gram should match cka for same inputs."""
        x = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(10, 8, dtype=torch.float64)

        Kx = linear_kernel(x)
        Ky = linear_kernel(y)

        cka1 = cka(x, y, CKAConfig(kernel="linear"))
        cka2 = cka_from_gram(Kx, Ky)

        assert torch.isclose(cka1, cka2, atol=1e-10)

    def test_self_cka_optimization(self):
        """CKA should detect same tensor and optimize."""
        x = torch.randn(10, 5, dtype=torch.float64)
        # Pass same tensor twice - should trigger optimization
        cka_val = cka(x, x)
        assert torch.isclose(cka_val, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    def test_cka_from_gram_same_matrix_optimization(self):
        """cka_from_gram should optimize when same gram matrix passed twice."""
        x = torch.randn(10, 5, dtype=torch.float64)
        gram_x = linear_kernel(x)
        cka_val = cka_from_gram(gram_x, gram_x)
        assert torch.isclose(cka_val, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)