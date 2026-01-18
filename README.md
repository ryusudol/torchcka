<div align="center">
# Centered Kernel Alignment (CKA)

**Lightning-fast, Memory-efficient, and Numerically Stable CKA for PyTorch**
</div>

## ✨ Key Features

- **Memory-efficient** — minibatch CKA without full-dataset loading
- **Safe & automatic** — context manager handles hooks and cleanup
- **Publication-ready plots** — heatmaps, trends, and comparison grids
- **Production-ready** — HuggingFace, DataParallel/DDP, auto layer selection

## 📦 Installation

Requires Python >= 3.10.

```bash
# Using pip
pip install pytorch-cka

# Using uv
uv add pytorch-cka

# From source
git clone https://github.com/ryusudol/Centered-Kernel-Alignment
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

### Visualization

**Heatmap**

```python
from pytorch_cka import plot_cka_heatmap

fig, ax = plot_cka_heatmap(
    cka_matrix,
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
      <td><img src="examples/plots/heatmap_self.png" alt="Self-comparison heatmap" width="100%"/></td>
      <td><img src="examples/plots/heatmap_cross.png" alt="Cross-model comparison heatmap" width="100%"/></td>
      <!-- <td><img src="plots/heatmap_masked.png" alt="Masked upper triangle heatmap" width="100%"/></td> -->
    </tr>
    <tr>
      <td align="center">Self-comparison</td>
      <td align="center">Cross-model</td>
      <!-- <td align="center">Masked Upper</td> -->
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
      <td><img src="examples/plots/trend_single.png" alt="Single trend plot" width="100%"/></td>
      <td><img src="examples/plots/trend_multi.png" alt="Multiple trends comparison" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">Single Trend</td>
      <td align="center">Multiple Trends</td>
    </tr>
</table>

<!-- **Side-by-Side Comparison**

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
      <td><img src="examples/plots/comparison_grid.png" alt="CKA comparison grid" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">CKA comparison grid</td>
    </tr>
</table> -->

## 📚 References

1. Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._

2. Nguyen, Thao, Maithra Raghu, and Simon Kornblith. ["Do Wide and Deep Networks Learn the Same Things?"](https://arxiv.org/abs/2010.15327) _arXiv 2020._ (Minibatch CKA)

3. Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert-Schmidt Independence Criterion: A Review."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) _Knowledge-Based Systems 2021._

### Related Projects

- [AntixK/PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare)
- [RistoAle97/centered-kernel-alignment](https://github.com/RistoAle97/centered-kernel-alignment)

## 📝 License

[MIT License](LICENSE)
