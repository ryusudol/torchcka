#!/usr/bin/env python3
"""Basic example of comparing two neural networks using CKA.

This example demonstrates:
1. Comparing a model with itself (self-similarity)
2. Comparing two different models
3. Visualizing results with heatmaps and trend plots
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_cka import CKA, plot_cka_heatmap, plot_cka_trend


# =============================================================================
# Define example models
# =============================================================================


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self, name: str = "SimpleCNN"):
        super().__init__()
        self.name = name
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class WiderCNN(nn.Module):
    """Wider CNN for comparison."""

    def __init__(self, name: str = "WiderCNN"):
        super().__init__()
        self.name = name
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def create_dummy_dataloader(
    batch_size: int = 16, num_samples: int = 64, image_size: int = 32
) -> DataLoader:
    """Create a dummy dataloader for demonstration."""
    data = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def example_self_comparison():
    """Example: Compare a model with itself."""
    print("\n" + "=" * 60)
    print("Example 1: Self-comparison (same model)")
    print("=" * 60)

    model = SimpleCNN()
    dataloader = create_dummy_dataloader()

    # Define layers to compare
    layers = ["conv1", "conv2", "conv3", "fc"]

    # Compute CKA using context manager
    with CKA(model, model, model1_layers=layers) as cka:
        matrix = cka.compare(dataloader, progress=False)

    print(f"\nCKA Matrix (self-comparison):\n{matrix}")
    print(f"\nDiagonal values (should be ~1.0): {torch.diag(matrix)}")

    # Visualize
    fig, ax = plot_cka_heatmap(
        matrix,
        layers1=layers,
        layers2=layers,
        model1_name="SimpleCNN",
        model2_name="SimpleCNN",
        annot=True,
        title="Self-Comparison: SimpleCNN",
    )
    fig.savefig("cka_self_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved: cka_self_comparison.png")


def example_two_model_comparison():
    """Example: Compare two different models."""
    print("\n" + "=" * 60)
    print("Example 2: Comparing two different models")
    print("=" * 60)

    model1 = SimpleCNN()
    model2 = WiderCNN()
    dataloader = create_dummy_dataloader()

    # Define layers (both models have similar structure)
    layers1 = ["conv1", "conv2", "conv3", "fc"]
    layers2 = ["conv1", "conv2", "conv3", "fc"]

    with CKA(
        model1,
        model2,
        model1_layers=layers1,
        model2_layers=layers2,
        model1_name="SimpleCNN",
        model2_name="WiderCNN",
    ) as cka:
        matrix = cka.compare(dataloader, progress=False)

    print(f"\nCKA Matrix (SimpleCNN vs WiderCNN):\n{matrix}")

    # Visualize
    fig, ax = plot_cka_heatmap(
        matrix,
        layers1=layers1,
        layers2=layers2,
        model1_name="SimpleCNN",
        model2_name="WiderCNN",
        annot=True,
        title="SimpleCNN vs WiderCNN",
    )
    fig.savefig("cka_two_models.png", dpi=150, bbox_inches="tight")
    print("\nSaved: cka_two_models.png")


def example_trend_plot():
    """Example: Plot CKA trend (diagonal values)."""
    print("\n" + "=" * 60)
    print("Example 3: CKA trend plot")
    print("=" * 60)

    model = SimpleCNN()
    dataloader = create_dummy_dataloader()

    layers = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "fc"]

    with CKA(model, model, model1_layers=layers) as cka:
        matrix = cka.compare(dataloader, progress=False)

    # Extract diagonal (self-similarity scores)
    diagonal = torch.diag(matrix)

    # Plot trend
    fig, ax = plot_cka_trend(
        diagonal,
        labels=["Self-similarity"],
        xlabel="Layer Index",
        ylabel="CKA Score",
        title="Layer Self-Similarity Across Network",
    )
    fig.savefig("cka_trend.png", dpi=150, bbox_inches="tight")
    print("\nSaved: cka_trend.png")


def example_callable_api():
    """Example: Using the callable API (simpler syntax)."""
    print("\n" + "=" * 60)
    print("Example 4: Callable API")
    print("=" * 60)

    model = SimpleCNN()
    dataloader = create_dummy_dataloader()

    layers = ["conv1", "conv2", "conv3"]

    # Using callable API (no context manager needed)
    cka = CKA(model, model, model1_layers=layers)
    matrix = cka(dataloader, progress=False)  # Automatically handles hooks

    print(f"\nCKA Matrix (callable API):\n{matrix}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("pytorch-cka Examples")
    print("=" * 60)

    example_self_comparison()
    example_two_model_comparison()
    example_trend_plot()
    example_callable_api()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
