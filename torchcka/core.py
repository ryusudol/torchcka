"""Core mathematical functions for CKA computation.

This module implements:
- Kernel functions (linear, RBF)
- HSIC (Hilbert-Schmidt Independence Criterion) computation
- CKA (Centered Kernel Alignment) computation

References:
    - Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
    - Nguyen et al., 2020: "Do Wide and Deep Networks Learn the Same Things?"
"""

from typing import Optional

import torch

from .config import CKAConfig


# =============================================================================
# KERNEL FUNCTIONS
# =============================================================================


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """Compute linear (dot product) kernel: K = X @ X^T.

    Args:
        x: Feature matrix of shape (n, d) where n is samples, d is features.

    Returns:
        Gram matrix of shape (n, n).

    Raises:
        ValueError: If x is not a 2D tensor.
    """
    if x.dim() != 2:
        raise ValueError(f"linear_kernel requires 2D tensor, got shape {x.shape}")
    return torch.mm(x, x.T)


def rbf_kernel(
    x: torch.Tensor,
    sigma: Optional[float] = None,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel: K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2)).

    Args:
        x: Feature matrix of shape (n, d).
        sigma: Kernel bandwidth. If None, uses median heuristic.
        epsilon: Small constant for numerical stability.

    Returns:
        Gram matrix of shape (n, n).

    Raises:
        ValueError: If x is not a 2D tensor.

    Note:
        Median heuristic: sigma^2 = median(pairwise_squared_distances) / 2
        This is a standard choice from Gretton et al.
    """
    if x.dim() != 2:
        raise ValueError(f"rbf_kernel requires 2D tensor, got shape {x.shape}")

    # Compute squared pairwise distances efficiently
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T * x_j
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dot_products

    # Clamp to avoid negative values due to numerical errors
    sq_distances = torch.clamp(sq_distances, min=0.0)

    if sigma is None:
        # Median heuristic with epsilon guard
        # Use upper triangle to avoid diagonal zeros
        n = sq_distances.shape[0]
        if n > 1:
            triu_indices = torch.triu_indices(n, n, offset=1, device=x.device)
            pairwise_sq_dists = sq_distances[triu_indices[0], triu_indices[1]]
            if pairwise_sq_dists.numel() > 0:
                median_sq_dist = torch.median(pairwise_sq_dists)
                sigma_sq = torch.clamp(median_sq_dist / 2.0, min=epsilon)
            else:
                sigma_sq = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        else:
            sigma_sq = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    else:
        sigma_sq = sigma**2

    # Clamp exponent to prevent overflow/underflow
    exponent = -sq_distances / (2 * (sigma_sq + epsilon))
    exponent = torch.clamp(exponent, min=-50.0, max=0.0)

    return torch.exp(exponent)


def compute_gram_matrix(
    x: torch.Tensor,
    kernel: str = "linear",
    sigma: Optional[float] = None,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Compute gram matrix using specified kernel.

    Args:
        x: Feature matrix of shape (n, d).
        kernel: Kernel type ("linear" or "rbf").
        sigma: RBF bandwidth (only used for rbf kernel).
        epsilon: Numerical stability constant.

    Returns:
        Gram matrix of shape (n, n).

    Raises:
        ValueError: If kernel is not "linear" or "rbf".
    """
    if kernel == "linear":
        return linear_kernel(x)
    elif kernel == "rbf":
        return rbf_kernel(x, sigma, epsilon)
    else:
        raise ValueError(f"kernel must be 'linear' or 'rbf', got '{kernel}'")


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
    total_mean = gram.mean()
    return gram - row_mean - col_mean + total_mean


# =============================================================================
# HSIC COMPUTATION
# =============================================================================


def hsic_biased(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
) -> torch.Tensor:
    """Compute biased HSIC: (1/(n-1)^2) * tr(K @ H @ L @ H).

    This is the standard HSIC estimator, suitable for full-batch computation.

    Args:
        gram_x: Gram matrix K of shape (n, n).
        gram_y: Gram matrix L of shape (n, n).

    Returns:
        HSIC value as scalar tensor.

    Raises:
        ValueError: If inputs are not 2D square tensors with matching shapes and n > 1.
    """
    # Validate dimensions
    if gram_x.dim() != 2 or gram_y.dim() != 2:
        raise ValueError(
            f"hsic_biased requires 2D tensors, got shapes {gram_x.shape} and {gram_y.shape}"
        )

    # Validate shapes match
    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"hsic_biased requires matching shapes, got {gram_x.shape} and {gram_y.shape}"
        )

    # Validate square matrices
    n, m = gram_x.shape
    if n != m:
        raise ValueError(
            f"hsic_biased requires square matrices, got shape {gram_x.shape}"
        )

    # Validate sample size
    if n <= 1:
        raise ValueError(f"hsic_biased requires n > 1, got n={n}")

    # Center the gram matrices
    centered_x = center_gram_matrix(gram_x)
    centered_y = center_gram_matrix(gram_y)

    # HSIC = (1/(n-1)^2) * tr(K_c @ L_c)
    # tr(A @ B) = sum(A * B^T) for symmetric matrices = sum(A * B)
    trace = (centered_x * centered_y).sum()
    return trace / ((n - 1) ** 2)


def hsic_unbiased(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    epsilon: float = 1e-10,
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
        Unbiased HSIC value as scalar tensor.

    Raises:
        ValueError: If inputs are not 2D square tensors with matching shapes and n > 3.
    """
    # Validate dimensions
    if gram_x.dim() != 2 or gram_y.dim() != 2:
        raise ValueError(
            f"hsic_unbiased requires 2D tensors, got shapes {gram_x.shape} and {gram_y.shape}"
        )

    # Validate shapes match
    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"hsic_unbiased requires matching shapes, got {gram_x.shape} and {gram_y.shape}"
        )

    # Validate square matrices
    n, m = gram_x.shape
    if n != m:
        raise ValueError(
            f"hsic_unbiased requires square matrices, got shape {gram_x.shape}"
        )

    # Validate sample size
    if n <= 3:
        raise ValueError(f"hsic_unbiased requires n > 3, got n={n}")

    # Clone to avoid modifying input, then zero diagonals
    K = gram_x.clone()
    L = gram_y.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    # Compute terms efficiently
    # Term 1: tr(K @ L) = sum of element-wise product for symmetric matrices
    trace_KL = (K * L).sum()

    # Term 2: (1^T K 1)(1^T L 1) / ((n-1)(n-2))
    sum_K = K.sum()
    sum_L = L.sum()
    term2 = (sum_K * sum_L) / ((n - 1) * (n - 2))

    # Term 3: 2 * (1^T K L 1) / (n-2) = 2 * sum(K @ L) / (n-2)
    KL = torch.mm(K, L)
    term3 = 2 * KL.sum() / (n - 2)

    # Combine terms
    main_term = trace_KL + term2 - term3

    # Normalize
    denominator = n * (n - 3) + epsilon
    return main_term / denominator


def hsic(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    unbiased: bool = True,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Compute HSIC using specified estimator.

    Args:
        gram_x: Gram matrix K of shape (n, n).
        gram_y: Gram matrix L of shape (n, n).
        unbiased: Use unbiased estimator (requires n > 3).
        epsilon: Numerical stability constant.

    Returns:
        HSIC value as scalar tensor.
    """
    if unbiased:
        return hsic_unbiased(gram_x, gram_y, epsilon)
    else:
        return hsic_biased(gram_x, gram_y)


# =============================================================================
# CKA COMPUTATION
# =============================================================================


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Optional[CKAConfig] = None,
) -> torch.Tensor:
    """Compute CKA between two feature matrices.

    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    Args:
        x: Features from model 1, shape (n, d1).
        y: Features from model 2, shape (n, d2).
        config: CKA configuration. If None, uses defaults.

    Returns:
        CKA similarity score in [0, 1].

    Note:
        When x and y are the same tensor (same memory), HSIC(K, K) is computed
        once and reused.
    """
    if config is None:
        config = CKAConfig()

    # Convert to target dtype for precision
    x = x.to(dtype=config.dtype)
    y = y.to(dtype=config.dtype)

    # Flatten if needed (B, C, H, W) -> (B, C*H*W)
    if x.dim() > 2:
        x = x.flatten(1)
    if y.dim() > 2:
        y = y.flatten(1)

    # Compute gram matrices
    gram_x = compute_gram_matrix(x, config.kernel, config.sigma, config.epsilon)
    gram_y = compute_gram_matrix(y, config.kernel, config.sigma, config.epsilon)

    # Compute HSIC values
    hsic_xy = hsic(gram_x, gram_y, config.unbiased, config.epsilon)

    # Optimization: check if x and y point to same memory
    if x.data_ptr() == y.data_ptr():
        hsic_xx = hsic_xy
        hsic_yy = hsic_xy
    else:
        hsic_xx = hsic(gram_x, gram_x, config.unbiased, config.epsilon)
        hsic_yy = hsic(gram_y, gram_y, config.unbiased, config.epsilon)

    # Compute CKA with epsilon guard in denominator
    # Clamp to non-negative to handle potential negative unbiased HSIC values
    denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=0.0)) + config.epsilon
    return hsic_xy / denominator


def cka_from_gram(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    unbiased: bool = True,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Compute CKA from pre-computed gram matrices.

    Useful when gram matrices are cached or computed elsewhere.

    Args:
        gram_x: Gram matrix K of shape (n, n).
        gram_y: Gram matrix L of shape (n, n).
        unbiased: Use unbiased HSIC estimator.
        epsilon: Numerical stability constant.

    Returns:
        CKA similarity score in [0, 1].
    """
    hsic_xy = hsic(gram_x, gram_y, unbiased, epsilon)

    # Check if same gram matrix
    if gram_x.data_ptr() == gram_y.data_ptr():
        hsic_xx = hsic_xy
        hsic_yy = hsic_xy
    else:
        hsic_xx = hsic(gram_x, gram_x, unbiased, epsilon)
        hsic_yy = hsic(gram_y, gram_y, unbiased, epsilon)

    # Clamp to non-negative to handle potential negative unbiased HSIC values
    denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=0.0)) + epsilon
    return hsic_xy / denominator
