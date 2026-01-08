<div align="center">

# CKA for PyTorch models

**Centered Kernel Alignment (CKA) for PyTorch**

Numerically stable, memory-safe comparison of neural network representations.

[![PyPI](https://img.shields.io/pypi/v/pytorch_cka?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/pytorch_cka/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## About

**Centered Kernel Alignment (CKA)** is a similarity metric for comparing representations learned by neural networks, based on the Hilbert-Schmidt Independence Criterion (HSIC). It answers the question: _"How similar are the features learned by two layers (or models)?"_

Given two matrices $X \in \mathbb{R}^{n \times p_1}$ and $Y \in \mathbb{R}^{n \times p_2}$ representing layer activations for $n$ samples, CKA computes:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where $K = XX^T$ and $L = YY^T$ are Gram matrices, and HSIC measures statistical dependence between them.

### Why pytorch-cka?

- **Memory-efficient**: Uses minibatch CKA—accumulates HSIC values instead of storing all activations
- **Safe hooks**: Context manager ensures forward hooks are always cleaned up
- **Same-model optimization**: Single forward pass when comparing a model with itself
- **DataParallel/DDP support**: Automatic model unwrapping
- **Auto layer selection**: Automatically finds suitable layers to compare
- **Publication-ready plots**: Heatmaps, trend plots, and comparison grids

---

## Installation

Requires Python >= 3.10.

```bash
# Using uv (recommended)
uv add pytorch-cka

# Using pip
pip install pytorch-cka

# From source
git clone https://github.com/ryusudol/pytorch-cka
cd pytorch-cka
uv sync  # or: pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from pytorch_cka import CKA, CKAConfig, plot_cka_heatmap

# Define layers to compare
layers = ["layer1", "layer2", "layer3", "fc"]

# Configure CKA (optional)
config = CKAConfig(kernel="linear", unbiased=True)

# Compute CKA using context manager (ensures hook cleanup)
with CKA(model, layers1=layers, config=config) as cka:
    matrix = cka.compare(dataloader)

# Visualize
fig, ax = plot_cka_heatmap(matrix, layers1=layers, layers2=layers)
fig.savefig("cka_heatmap.png")
```

### Comparing Two Models

```python
with CKA(
    model1, model2,
    layers1=["conv1", "conv2", "fc"],
    layers2=["conv1", "conv2", "fc"],
    model1_name="ResNet18",
    model2_name="ResNet34",
) as cka:
    matrix = cka.compare(dataloader)
```

### Callable API (Simpler)

```python
# No context manager needed—hooks managed automatically
cka = CKA(model, layers1=layers)
matrix = cka(dataloader)
```

### Auto Layer Selection

```python
# Automatically selects up to 50 layers if none specified
with CKA(model) as cka:
    matrix = cka.compare(dataloader)
    print(f"Compared layers: {cka.layers1}")
```

---

## Configuration

```python
from pytorch_cka import CKAConfig

config = CKAConfig(
    kernel="linear",    # "linear" or "rbf"
    sigma=None,         # RBF bandwidth (None = median heuristic)
    unbiased=True,      # Unbiased HSIC estimator (requires batch_size > 3)
    epsilon=1e-6,       # Numerical stability constant
    dtype=torch.float64,# Computation precision
    device=None,        # Auto-detected from model
)
```

> **Note**: Unbiased HSIC (default) requires batch size > 3. For smaller batches, set `unbiased=False`.

---

## Visualization

### Heatmap

```python
from pytorch_cka import plot_cka_heatmap

fig, ax = plot_cka_heatmap(
    matrix,
    layers1=layers,
    layers2=layers,
    model1_name="Model A",
    model2_name="Model B",
    annot=True,           # Show values in cells
    cmap="magma",         # Colormap
    mask_upper=True,      # Mask upper triangle (symmetric matrices)
)
```

### Trend Plot

```python
from pytorch_cka import plot_cka_trend

# Plot diagonal (self-similarity across layers)
diagonal = torch.diag(matrix)
fig, ax = plot_cka_trend(
    diagonal,
    labels=["Self-similarity"],
    xlabel="Layer",
    ylabel="CKA Score",
)
```

### Side-by-Side Comparison

```python
from pytorch_cka import plot_cka_comparison

fig, axes = plot_cka_comparison(
    matrices=[matrix1, matrix2, matrix3],
    titles=["Epoch 1", "Epoch 10", "Epoch 100"],
    layers=layers,
    share_colorbar=True,
)
```

---

## Checkpoints

Save and load CKA results for later analysis:

```python
# Save
with CKA(model, layers1=layers) as cka:
    matrix = cka.compare(dataloader)
    cka.save_checkpoint(
        "results.pt",
        matrix,
        metadata={"experiment": "ablation", "dataset": "CIFAR-10"},
    )

# Load
checkpoint = CKA.load_checkpoint("results.pt")
matrix = checkpoint["cka_matrix"]
layers = checkpoint["model1_info"]["layers"]
config = checkpoint["config"]
```

---

## API Reference

### Core Functions

| Function                 | Description                                 |
| ------------------------ | ------------------------------------------- |
| `cka(X, Y, ...)`         | Compute CKA between two activation matrices |
| `hsic(K, L, ...)`        | Compute HSIC between two Gram matrices      |
| `compute_gram_matrix(X)` | Compute Gram matrix using linear kernel     |

### Utilities

| Function                         | Description                    |
| -------------------------------- | ------------------------------ |
| `validate_layers(model, layers)` | Check which layers exist       |
| `unwrap_model(model)`            | Unwrap DataParallel/DDP models |
| `get_device(model)`              | Detect model device            |

### Visualization

| Function                   | Description                    |
| -------------------------- | ------------------------------ |
| `plot_cka_heatmap(...)`    | CKA similarity heatmap         |
| `plot_cka_trend(...)`      | Line plot for CKA trends       |
| `plot_cka_comparison(...)` | Side-by-side matrix comparison |
| `save_figure(fig, path)`   | Save with sensible defaults    |

---

## References

1. Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._

2. Nguyen, Thao, Maithra Raghu, and Simon Kornblith. ["Do Wide and Deep Networks Learn the Same Things?"](https://arxiv.org/abs/2010.15327) _arXiv 2020._ (Minibatch CKA)

3. Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert-Schmidt Independence Criterion: A Review."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) _Knowledge-Based Systems 2021._

### Related Projects

- [google-research/representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) — Original implementation
- [AntixK/PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare) — PyTorch CKA with hooks
- [numpee/CKA.pytorch](https://github.com/numpee/CKA.pytorch) — Minibatch CKA implementation

---

## License

[MIT](LICENSE)
