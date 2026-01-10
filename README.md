<div align="center">
    
# 🚀 PyTorch-CKA

**Centered Kernel Alignment (CKA) for PyTorch**

Fast, memory-efficient, and numerically stable Centered Kernel Alignment (CKA) for layer-wise similarity analysis.

[![PyPI](https://img.shields.io/pypi/v/pytorch_cka?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/pytorch_cka/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>


## ✨ Key Features

- **Fast** — optimized HSIC core + same-model single-pass optimization
- **Memory-efficient** — minibatch CKA without full-dataset loading
- **Safe & automatic** — context manager handles hooks and cleanup
- **Publication-ready plots** — heatmaps, trends, and comparison grids
- **Production-ready** — HuggingFace, DataParallel/DDP, auto layer selection


## ✍🏼 About CKA

**CKA** is a similarity metric for comparing representations learned by neural networks, based on the Hilbert-Schmidt Independence Criterion (HSIC). It answers the question: _"How similar are the features learned by two layers (or models)?"_

Given two matrices $X \in \mathbb{R}^{n \times p_1}$ and $Y \in \mathbb{R}^{n \times p_2}$ representing layer activations for $n$ samples, CKA computes:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where $K = XX^T$ and $L = YY^T$ are Gram matrices, and HSIC measures statistical dependence between them.


## 📦 Installation

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


## 👟 Quick Start

### Basic Usage

```python
from torch.utils.data import DataLoader
from pytorch_cka import CKA

pretrained_model = ...  # e.g. pretrained ResNet-18
fine_tuned_model = ...  # e.g. fine-tuned ResNet-18

layers = ["layer1", "layer2", "layer3", "fc"]

dataloader = DataLoader(..., batch_size=128)

cka = CKA(
    model1=pretrained_model,
    model2=fine_tuned_model,
    model1_name="ResNet-18 (pretrained)",
    model2_name="ResNet-18 (fine-tuned)",
    model1_layers=layers,
    model2_layers=layers,
    device="cuda"
)

# Most convenient usage (auto context manager)
cka_matrix = cka(dataloader)
cka_result = cka.export(cka_matrix)

# Or explicit control
with cka:
    cka_matrix = cka.compare(dataloader)
    cka_result = cka.export(cka_matrix)
```

### Real-time Monitoring

```python
def progress_callback(batch_idx: int, total: int, current_matrix: torch.Tensor):
    print(f"Batch {batch_idx}/{total} | Mean CKA: {current_matrix.mean():.4f}")

cka.compare(dataloader, callback=progress_callback)
```

### Visualization

**Heatmap**

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

<table>
    <tr>
      <td><img src="plots/heatmap_self.png" alt="Self-comparison heatmap" width="100%"/></td>
      <td><img src="plots/heatmap_cross.png" alt="Cross-model comparison heatmap" width="100%"/></td>
      <td><img src="plots/heatmap_masked.png" alt="Masked upper triangle heatmap" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">Self-comparison</td>
      <td align="center">Cross-model</td>
      <td align="center">Masked Upper</td>
    </tr>
</table>

**Trend Plot**

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

<table>
    <tr>
      <td><img src="plots/trend_single.png" alt="Single trend plot" width="100%"/></td>
      <td><img src="plots/trend_multi.png" alt="Multiple trends comparison" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">Single Trend</td>
      <td align="center">Multiple Trends</td>
    </tr>
</table>

**Side-by-Side Comparison**

```python
from pytorch_cka import plot_cka_comparison

fig, axes = plot_cka_comparison(
    matrices=[matrix1, matrix2, matrix3],
    titles=["Epoch 1", "Epoch 10", "Epoch 100"],
    layers=layers,
    share_colorbar=True,
)
```

<table>
    <tr>
      <td><img src="plots/comparison_grid.png" alt="CKA comparison grid" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">CKA comparison grid</td>
    </tr>
</table>


## 📚 References

1. Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._

2. Nguyen, Thao, Maithra Raghu, and Simon Kornblith. ["Do Wide and Deep Networks Learn the Same Things?"](https://arxiv.org/abs/2010.15327) _arXiv 2020._ (Minibatch CKA)

3. Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert-Schmidt Independence Criterion: A Review."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) _Knowledge-Based Systems 2021._


### Related Projects

- [AntixK/PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare)
- [RistoAle97/centered-kernel-alignment](https://github.com/RistoAle97/centered-kernel-alignment)

## 📝 License

[MIT License](LICENSE)
