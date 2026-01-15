import torch

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


def hsic(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
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

    Returns:
        HSIC value as scalar tensor.

    Raises:
        ValueError: If inputs are not 2D square tensors with matching shapes and n > 3.
    """
    if gram_x.dim() != 2 or gram_y.dim() != 2:
        raise ValueError(
            f"hsic requires 2D tensors, got shapes {gram_x.shape} and {gram_y.shape}"
        )

    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"hsic requires matching shapes, got {gram_x.shape} and {gram_y.shape}"
        )

    n, m = gram_x.shape
    if n != m:
        raise ValueError(
            f"hsic requires square matrices, got shape {gram_x.shape}"
        )

    if n <= 3:
        raise ValueError(f"hsic requires n > 3, got n={n}")

    diag_x = gram_x.diagonal()
    diag_y = gram_y.diagonal()

    # Term 1: tr(K @ L) where K, L have zero diagonals
    trace_KL = (gram_x * gram_y).sum() - (diag_x * diag_y).sum()

    # Term 2: (1^T K 1)(1^T L 1) / ((n-1)(n-2))
    sum_K = gram_x.sum() - diag_x.sum()
    sum_L = gram_y.sum() - diag_y.sum()
    term2 = (sum_K * sum_L) / ((n - 1) * (n - 2))

    # Term 3: 2 * sum(K @ L) / (n-2)
    col_sum_K = gram_x.sum(dim=0) - diag_x
    col_sum_L = gram_y.sum(dim=0) - diag_y
    term3 = 2 * (col_sum_K @ col_sum_L) / (n - 2)

    main_term = trace_KL + term2 - term3

    denominator = n * (n - 3)
    return main_term / denominator
