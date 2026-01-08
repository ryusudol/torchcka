"""Tests for pytorch_cka.core module."""

import pytest
import torch

from pytorch_cka.core import (
    center_gram_matrix,
    cka,
    cka_from_gram,
    compute_gram_matrix,
    hsic,
)


class TestComputeGramMatrix:
    """Tests for compute_gram_matrix function."""

    def test_invalid_dims(self):
        """compute_gram_matrix should raise ValueError for non-2D tensors."""
        x_1d = torch.randn(10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            compute_gram_matrix(x_1d)

        x_3d = torch.randn(5, 10, 5)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            compute_gram_matrix(x_3d)

    def test_shape(self):
        """compute_gram_matrix should produce (n, n) gram matrix."""
        x = torch.randn(10, 5)
        K = compute_gram_matrix(x)
        assert K.shape == (10, 10)

    def test_symmetry(self):
        """Gram matrix should be symmetric."""
        x = torch.randn(10, 5)
        K = compute_gram_matrix(x)
        assert torch.allclose(K, K.T)

    def test_positive_semidefinite(self):
        """Gram matrix should be positive semi-definite."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = compute_gram_matrix(x)
        eigenvalues = torch.linalg.eigvalsh(K)
        # Allow small numerical errors
        assert (eigenvalues >= -1e-6).all()

    def test_computation(self):
        """compute_gram_matrix should compute K = X @ X^T."""
        x = torch.randn(5, 3)
        K = compute_gram_matrix(x)
        expected = x @ x.T
        assert torch.allclose(K, expected)


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
        K = compute_gram_matrix(x)
        Kc = center_gram_matrix(K)
        assert torch.allclose(Kc.sum(dim=0), torch.zeros(10, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(Kc.sum(dim=1), torch.zeros(10, dtype=torch.float64), atol=1e-10)

    def test_symmetry_preserved(self):
        """Centering should preserve symmetry."""
        x = torch.randn(10, 5)
        K = compute_gram_matrix(x)
        Kc = center_gram_matrix(K)
        assert torch.allclose(Kc, Kc.T)


class TestHSIC:
    """Tests for HSIC function."""

    def test_invalid_dims(self):
        """hsic should raise for non-2D gram matrices."""
        gram_valid = torch.randn(10, 10)
        gram_1d = torch.randn(10)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic(gram_1d, gram_valid)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic(gram_valid, gram_1d)

    def test_shape_mismatch(self):
        """hsic should raise when gram matrix shapes don't match."""
        gram_10x10 = torch.randn(10, 10)
        gram_8x8 = torch.randn(8, 8)

        with pytest.raises(ValueError, match="requires matching shapes"):
            hsic(gram_10x10, gram_8x8)

    def test_non_square(self):
        """hsic should raise for non-square gram matrices."""
        gram_rect = torch.randn(10, 15)

        with pytest.raises(ValueError, match="requires square matrices"):
            hsic(gram_rect, gram_rect)

    def test_requires_n_gt_3(self):
        """HSIC should raise error for n <= 3."""
        K = torch.randn(3, 3)
        K = K @ K.T  # Make symmetric
        with pytest.raises(ValueError, match="n > 3"):
            hsic(K, K)

    def test_hsic_identical_positive(self):
        """HSIC(K, K) should be positive for non-trivial K."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = compute_gram_matrix(x)
        hsic_val = hsic(K, K)
        assert hsic_val > 0


class TestCKA:
    """Tests for CKA computation functions."""

    def test_identical_features_gives_one(self):
        """CKA(X, X) should equal 1."""
        x = torch.randn(10, 5, dtype=torch.float64)
        cka_val = cka(x, x, epsilon=1e-10)
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

        Kx = compute_gram_matrix(x)
        Ky = compute_gram_matrix(y)

        cka1 = cka(x, y)
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
        gram_x = compute_gram_matrix(x)
        cka_val = cka_from_gram(gram_x, gram_x)
        assert torch.isclose(cka_val, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
