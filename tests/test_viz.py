"""Tests for pytorch_cka.viz module."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from pytorch_cka.viz import plot_cka_comparison, plot_cka_heatmap, plot_cka_trend, save_figure


class TestPlotCKAHeatmap:
    """Tests for plot_cka_heatmap function."""

    def test_returns_figure_and_axes(self):
        """Should always return (Figure, Axes) tuple."""
        matrix = torch.rand(5, 5)
        fig, ax = plot_cka_heatmap(matrix)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_numpy_input(self):
        """Should accept numpy array input."""
        matrix = np.random.rand(5, 5)
        fig, ax = plot_cka_heatmap(matrix)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_tick_labels(self):
        """Tick labels should match layer lists correctly."""
        matrix = torch.rand(3, 3)
        layers1 = ["a.layer1", "a.layer2", "a.layer3"]
        layers2 = ["b.layer1", "b.layer2", "b.layer3"]

        fig, ax = plot_cka_heatmap(matrix, layers1=layers1, layers2=layers2)

        # Y-axis should have layers1
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == layers1

        # X-axis should have layers2
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert x_labels == layers2

        plt.close(fig)

    def test_layer_name_depth(self):
        """layer_name_depth should truncate layer names correctly."""
        matrix = torch.rand(2, 2)
        layers = ["encoder.block.layer1", "encoder.block.layer2"]

        fig, ax = plot_cka_heatmap(matrix, layers1=layers, layers2=layers, layer_name_depth=1)

        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == ["layer1", "layer2"]

        plt.close(fig)

    def test_custom_figsize(self):
        """Should respect custom figsize."""
        matrix = torch.rand(5, 5)
        fig, ax = plot_cka_heatmap(matrix, figsize=(10, 8))

        width, height = fig.get_size_inches()
        assert abs(width - 10) < 0.1
        assert abs(height - 8) < 0.1

        plt.close(fig)

    def test_annotations(self):
        """Should show annotations when annot=True."""
        matrix = torch.rand(3, 3)
        fig, ax = plot_cka_heatmap(matrix, annot=True)

        # Check that text annotations exist
        texts = ax.texts
        assert len(texts) == 9  # 3x3 matrix

        plt.close(fig)

    def test_no_colorbar(self):
        """Should not show colorbar when colorbar=False."""
        matrix = torch.rand(5, 5)
        fig, ax = plot_cka_heatmap(matrix, colorbar=False)

        # Check that only one axes exists (no colorbar axes)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_existing_axes(self):
        """Should plot on existing axes when provided."""
        matrix = torch.rand(5, 5)
        fig_orig, ax_orig = plt.subplots()

        fig, ax = plot_cka_heatmap(matrix, ax=ax_orig, colorbar=False)

        assert fig is fig_orig
        assert ax is ax_orig

        plt.close(fig)

    def test_mask_upper(self):
        """Should mask upper triangle when mask_upper=True."""
        matrix = torch.rand(5, 5)
        fig, ax = plot_cka_heatmap(matrix, mask_upper=True, annot=True)

        # Upper triangle annotations should not be present
        texts = ax.texts
        # For 5x5 matrix with upper masked, should have 5+4+3+2+1 = 15 cells
        # but upper triangle (10 cells) is masked, leaving 15-10=5? No wait
        # Actually the lower triangle including diagonal has 5+4+3+2+1 = 15 cells
        # Upper triangle excluding diagonal has 4+3+2+1 = 10 cells
        # So we should have 25 - 10 = 15 annotations
        assert len(texts) == 15

        plt.close(fig)

    def test_vmin_vmax(self):
        """Should respect vmin and vmax settings."""
        matrix = torch.rand(5, 5) * 0.5 + 0.25  # Values in [0.25, 0.75]
        fig, ax = plot_cka_heatmap(matrix, vmin=0, vmax=1)

        # Get the image data
        im = ax.images[0]
        assert im.get_clim() == (0, 1)

        plt.close(fig)

    def test_show_calls_plt_show(self, monkeypatch):
        """Should call plt.show() when show=True."""
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

        matrix = torch.rand(3, 3)
        fig, ax = plot_cka_heatmap(matrix, show=True)

        assert len(show_called) == 1
        plt.close(fig)


class TestPlotCKATrend:
    """Tests for plot_cka_trend function."""

    def test_returns_figure_and_axes(self):
        """Should always return (Figure, Axes) tuple."""
        values = torch.rand(10)
        fig, ax = plot_cka_trend(values)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_single_line(self):
        """Should plot single line correctly."""
        values = torch.rand(10)
        fig, ax = plot_cka_trend(values)

        assert len(ax.lines) == 1
        plt.close(fig)

    def test_multiple_lines(self):
        """Should plot multiple lines correctly."""
        values = [torch.rand(10) for _ in range(3)]
        labels = ["Line A", "Line B", "Line C"]

        fig, ax = plot_cka_trend(values, labels=labels)

        assert len(ax.lines) == 3
        plt.close(fig)

    def test_numpy_input(self):
        """Should accept numpy array input."""
        values = np.random.rand(10)
        fig, ax = plot_cka_trend(values)

        assert len(ax.lines) == 1
        plt.close(fig)

    def test_2d_array_input(self):
        """Should handle 2D array as multiple lines."""
        values = torch.rand(3, 10)  # 3 lines, 10 points each
        fig, ax = plot_cka_trend(values)

        assert len(ax.lines) == 3
        plt.close(fig)

    def test_custom_x_values(self):
        """Should use custom x values when provided."""
        values = torch.rand(5)
        x_values = [0, 2, 4, 6, 8]

        fig, ax = plot_cka_trend(values, x_values=x_values)

        line_data = ax.lines[0].get_xdata()
        assert list(line_data) == x_values

        plt.close(fig)

    def test_ylim(self):
        """Y-axis should be limited to [0, 1.05] for CKA."""
        values = torch.rand(10)
        fig, ax = plot_cka_trend(values)

        ylim = ax.get_ylim()
        assert ylim[0] == 0
        assert abs(ylim[1] - 1.05) < 0.01

        plt.close(fig)

    def test_grid(self):
        """Should show grid when grid=True."""
        values = torch.rand(10)
        fig, ax = plot_cka_trend(values, grid=True)

        # Grid should be enabled
        assert ax.xaxis.get_gridlines()[0].get_visible()

        plt.close(fig)

    def test_existing_axes(self):
        """Should plot on existing axes when provided."""
        fig_orig, ax_orig = plt.subplots()
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig)

    def test_with_title(self):
        """Should set title when provided."""
        values = torch.rand(10)
        title = "My CKA Trend"

        fig, ax = plot_cka_trend(values, title=title)

        assert ax.get_title() == title
        plt.close(fig)

    def test_show_calls_plt_show(self, monkeypatch):
        """Should call plt.show() when show=True."""
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

        values = torch.rand(10)  # You forgot this line!
        fig, ax = plot_cka_trend(values, show=True)

        assert len(show_called) == 1
        plt.close(fig)


class TestPlotCKAComparison:
    """Tests for plot_cka_comparison function."""

    def test_returns_figure_and_axes(self):
        """Should return (Figure, array of Axes)."""
        matrices = [torch.rand(5, 5), torch.rand(5, 5)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        plt.close(fig)

    def test_correct_number_of_subplots(self):
        """Should create correct number of subplots."""
        matrices = [torch.rand(5, 5) for _ in range(4)]
        titles = [f"Matrix {i}" for i in range(4)]

        fig, axes = plot_cka_comparison(matrices, titles, ncols=2)

        # 4 plots in 2 columns = 2 rows
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_shared_colorbar(self):
        """Should use shared colorbar when share_colorbar=True."""
        matrices = [torch.rand(5, 5), torch.rand(5, 5)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, share_colorbar=True)

        # With shared colorbar, individual plots shouldn't have colorbars
        # but a shared one is added
        plt.close(fig)

    def test_hides_empty_subplots(self):
        """Should hide empty subplots when grid is not full."""
        matrices = [torch.rand(5, 5) for _ in range(3)]
        titles = [f"Matrix {i}" for i in range(3)]

        fig, axes = plot_cka_comparison(matrices, titles, ncols=2)

        # 3 plots in 2 columns = 2 rows, 1 empty
        assert not axes[1, 1].get_visible()
        plt.close(fig)

    def test_numpy_input_matrices(self):
        """Should accept numpy arrays in matrices list."""
        matrices = [np.random.rand(5, 5), np.random.rand(5, 5)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, share_colorbar=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_calls_plt_show(self, monkeypatch):
        """Should call plt.show() when show=True."""
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

        matrices = [torch.rand(5, 5), torch.rand(5, 5)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, show=True)

        assert len(show_called) == 1
        plt.close(fig)


class TestSaveFigure:
    """Tests for save_figure function."""

    def test_saves_file(self, tmp_path):
        """Should save figure to file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test_figure.png"
        save_figure(fig, str(output_path))

        assert output_path.exists()

    def test_closes_figure(self, tmp_path):
        """Should close figure after saving."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        fig_num = fig.number

        output_path = tmp_path / "test_figure.png"
        save_figure(fig, str(output_path))

        # Figure should be closed
        assert fig_num not in plt.get_fignums()

    def test_custom_dpi(self, tmp_path):
        """Should respect custom dpi."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test_figure.png"
        save_figure(fig, str(output_path), dpi=300)

        assert output_path.exists()
