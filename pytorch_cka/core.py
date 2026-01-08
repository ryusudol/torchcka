"""Core mathematical functions for CKA computation.

This module implements:
- Gram matrix computation (using linear kernel)
- HSIC (Hilbert-Schmidt Independence Criterion) computation
- CKA (Centered Kernel Alignment) computation

References:
    - Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
    - Nguyen et al., 2020: "Do Wide and Deep Networks Learn the Same Things?"
"""

import torch

# Numerical stability constant for CKA computations
EPSILON = 1e-6


# =============================================================================
# GRAM MATRIX COMPUTATION
# =============================================================================


def compute_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix using linear kernel: K = X @ X^T.

    Args:
        x: Feature matrix of shape (n, d) where n is samples, d is features.

    Returns:
        Gram matrix of shape (n, n).

    Raises:
        ValueError: If x is not a 2D tensor.
    """
    if x.dim() != 2:
        raise ValueError(f"compute_gram_matrix requires 2D tensor, got shape {x.shape}")
    return torch.mm(x, x.T)


# =============================================================================
# CENTERING
# =============================================================================


def center_gram_matrix(gram: torch.Tensor) -> torch.Tensor:
    """Center a gram matrix using the centering matrix H = I - (1/n) * 11^T.

    The centered gram matrix is: K_c = H @ K @ H

    Args:
        gram: Gram matrix of shape (n, n).

    Returns:
        Centered gram matrix of shape (n, n).

    Raises:
        ValueError: If gram is not a 2D square tensor with n >= 1.
    """
    # Validate dimensions
    if gram.dim() != 2:
        raise ValueError(
            f"center_gram_matrix requires 2D tensor, got shape {gram.shape}"
        )

    # Validate square matrices
    n, m = gram.shape
    if n != m:
        raise ValueError(
            f"center_gram_matrix requires square matrix, got shape {gram.shape}"
        )

    # Validate sample size
    if n < 1:
        raise ValueError(f"center_gram_matrix requires non-empty matrix, got n={n}")

    # Efficient centering without explicit matrix construction
    # H @ K @ H = K - (1/n) * K @ 1 @ 1^T - (1/n) * 1 @ 1^T @ K + (1/n^2) * 1 @ 1^T @ K @ 1 @ 1^T
    # Simplified: K_c[i,j] = K[i,j] - mean(K[i,:]) - mean(K[:,j]) + mean(K)
    row_mean = gram.mean(dim=1, keepdim=True)
    col_mean = gram.mean(dim=0, keepdim=True)
    total_mean = row_mean.mean()
    return gram - row_mean - col_mean + total_mean


# =============================================================================
# HSIC COMPUTATION
# =============================================================================


def hsic(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Compute unbiased HSIC estimator (Song et al. 2012, Nguyen et al. 2020).

    Formula:
        HSIC_1 = (1 / (n * (n-3))) * [
            tr(K'L') + (1^T K' 1)(1^T L' 1) / ((n-1)(n-2)) - 2(1^T K' L' 1) / (n-2)
        ]
    where K' and L' have zeros on diagonal.

    Args:
        gram_x: Gram matrix K of shape (n, n).
        gram_y: Gram matrix L of shape (n, n).
        epsilon: Numerical stability constant.

    Returns:
        HSIC value as scalar tensor.

    Raises:
        ValueError: If inputs are not 2D square tensors with matching shapes and n > 3.
    """
    # Validate dimensions
    if gram_x.dim() != 2 or gram_y.dim() != 2:
        raise ValueError(
            f"hsic requires 2D tensors, got shapes {gram_x.shape} and {gram_y.shape}"
        )

    # Validate shapes match
    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"hsic requires matching shapes, got {gram_x.shape} and {gram_y.shape}"
        )

    # Validate square matrices
    n, m = gram_x.shape
    if n != m:
        raise ValueError(
            f"hsic requires square matrices, got shape {gram_x.shape}"
        )

    # Validate sample size
    if n <= 3:
        raise ValueError(f"hsic requires n > 3, got n={n}")

    # Extract diagonal elements (no cloning needed)
    diag_x = gram_x.diagonal()
    diag_y = gram_y.diagonal()

    # Term 1: tr(K @ L) where K, L have zero diagonals
    # = sum(gram_x * gram_y) - sum(diag_x * diag_y)
    trace_KL = (gram_x * gram_y).sum() - (diag_x * diag_y).sum()

    # Term 2: (1^T K 1)(1^T L 1) / ((n-1)(n-2))
    # K.sum() = gram_x.sum() - diag_x.sum()
    sum_K = gram_x.sum() - diag_x.sum()
    sum_L = gram_y.sum() - diag_y.sum()
    term2 = (sum_K * sum_L) / ((n - 1) * (n - 2))

    # Term 3: 2 * sum(K @ L) / (n-2)
    # Optimization: sum(A @ B) = A.sum(dim=0) @ B.sum(dim=0)
    # For K with zero diagonal: col_sum_K = gram_x.sum(dim=0) - diag_x
    col_sum_K = gram_x.sum(dim=0) - diag_x
    col_sum_L = gram_y.sum(dim=0) - diag_y
    term3 = 2 * (col_sum_K @ col_sum_L) / (n - 2)

    # Combine terms
    main_term = trace_KL + term2 - term3

    # Normalize
    denominator = n * (n - 3) + epsilon
    return main_term / denominator


# =============================================================================
# CKA COMPUTATION
# =============================================================================


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Compute CKA between two feature matrices.

    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    Args:
        x: Features from model 1, shape (n, d1).
        y: Features from model 2, shape (n, d2).
        epsilon: Small constant for numerical stability.

    Returns:
        CKA similarity score in [0, 1].

    Note:
        When x and y are the same tensor (same memory), HSIC(K, K) is computed
        once and reused.
    """
    # Flatten if needed (B, C, H, W) -> (B, C*H*W)
    if x.dim() > 2:
        x = x.flatten(1)
    if y.dim() > 2:
        y = y.flatten(1)

    # Compute gram matrices
    gram_x = compute_gram_matrix(x)
    gram_y = compute_gram_matrix(y)

    # Compute HSIC values
    hsic_xy = hsic(gram_x, gram_y, epsilon)

    # Optimization: check if x and y point to same memory
    if x.data_ptr() == y.data_ptr():
        hsic_xx = hsic_xy
        hsic_yy = hsic_xy
    else:
        hsic_xx = hsic(gram_x, gram_x, epsilon)
        hsic_yy = hsic(gram_y, gram_y, epsilon)

    # Compute CKA with epsilon guard in denominator
    # Clamp to non-negative to handle potential negative unbiased HSIC values
    denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=0.0)) + epsilon
    return hsic_xy / denominator


def cka_from_gram(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Compute CKA from pre-computed gram matrices.

    Useful when gram matrices are cached or computed elsewhere.

    Args:
        gram_x: Gram matrix K of shape (n, n).
        gram_y: Gram matrix L of shape (n, n).
        epsilon: Numerical stability constant.

    Returns:
        CKA similarity score in [0, 1].
    """
    hsic_xy = hsic(gram_x, gram_y, epsilon)

    # Check if same gram matrix
    if gram_x.data_ptr() == gram_y.data_ptr():
        hsic_xx = hsic_xy
        hsic_yy = hsic_xy
    else:
        hsic_xx = hsic(gram_x, gram_x, epsilon)
        hsic_yy = hsic(gram_y, gram_y, epsilon)

    # Clamp to non-negative to handle potential negative unbiased HSIC values
    denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=0.0)) + epsilon
    return hsic_xy / denominator
