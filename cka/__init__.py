"""pytorch-cka: Centered Kernel Alignment for PyTorch models.

A numerically stable, memory-safe library for comparing neural network
representations using Centered Kernel Alignment (CKA).

Example:
    >>> from pytorch_cka import CKA
    >>>
    >>> with CKA(model1, model2, model1_layers=["layer1", "layer2"]) as cka:
    ...     matrix = cka.compare(dataloader)
    ...     fig, ax = plot_cka_heatmap(matrix, model1_layers=["layer1", "layer2"])

References:
    - Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
    - Nguyen et al., 2020: "Do Wide and Deep Networks Learn the Same Things?"
"""

__version__ = "0.1.1"

from .cka import CKA
from .hsic import hsic, hsic_outer
from .viz import (
    plot_cka_comparison,
    plot_cka_heatmap,
    plot_cka_trend,
    save_figure,
)

__all__ = [
    "__version__",
    "CKA",
    "hsic",
    "hsic_outer",
    "plot_cka_heatmap",
    "plot_cka_trend",
    "plot_cka_comparison",
    "save_figure",
]
