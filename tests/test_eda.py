"""Tests for bookdb.eda utility functions."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from bookdb.eda.plots import (
    plot_correlation_heatmap,
    plot_log_histogram,
    plot_rating_distribution,
)
from bookdb.eda.stats import iqr_outlier_bounds


# bookdb.eda.stats

class TestIqrOutlierBounds:
    def test_symmetric_data(self):
        # Values 1–9, Q1=3, Q3=7, IQR=4 → bounds = (3 - 3*4, 7 + 3*4) = (-9, 19)
        s = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        lower, upper = iqr_outlier_bounds(s)
        assert lower < 1
        assert upper > 9

    def test_outlier_falls_outside_bounds(self):
        s = pl.Series(list(range(1, 10)) + [100])
        _, upper = iqr_outlier_bounds(s, multiplier=1.5)
        assert upper < 100

    def test_inlier_falls_inside_bounds(self):
        s = pl.Series(list(range(1, 20)))
        lower, upper = iqr_outlier_bounds(s, multiplier=1.5)
        assert lower < 10 < upper

    def test_custom_multiplier(self):
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        lower_loose, upper_loose = iqr_outlier_bounds(s, multiplier=3.0)
        lower_tight, upper_tight = iqr_outlier_bounds(s, multiplier=1.5)
        assert upper_loose > upper_tight
        assert lower_loose < lower_tight

    def test_returns_tuple_of_floats(self):
        s = pl.Series([1.0, 2.0, 3.0])
        result = iqr_outlier_bounds(s)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_constant_series(self):
        # IQR=0 for constant series → bounds both equal the constant value
        s = pl.Series([5.0, 5.0, 5.0, 5.0])
        lower, upper = iqr_outlier_bounds(s)
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)


# bookdb.eda.plots

class TestPlotRatingDistribution:
    def _make_df(self):
        return pl.DataFrame({"rating": [0, 0, 1, 2, 3, 3, 4, 5, 5, 5]})

    def test_returns_figure(self):
        fig = plot_rating_distribution(self._make_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_axes(self):
        fig = plot_rating_distribution(self._make_df())
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_custom_rating_col(self):
        df = pl.DataFrame({"stars": [1, 2, 3, 4, 5]})
        fig = plot_rating_distribution(df, rating_col="stars")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        fig = plot_rating_distribution(self._make_df(), figsize=(8, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8.0)
        assert h == pytest.approx(4.0)
        plt.close(fig)

    def test_label_in_millions_false(self):
        # Should not raise even with small counts that produce "0.0M"
        fig = plot_rating_distribution(self._make_df(), label_in_millions=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLogHistogram:
    def _values(self):
        rng = np.random.default_rng(42)
        return rng.exponential(scale=100, size=500)

    def test_returns_figure(self):
        fig = plot_log_histogram(self._values(), cap=500, xlabel="X", title="T")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_axes(self):
        fig = plot_log_histogram(self._values(), cap=500, xlabel="X", title="T")
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_accepts_list(self):
        fig = plot_log_histogram([1, 2, 3, 100, 200], cap=150, xlabel="X", title="T")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_polars_series(self):
        s = pl.Series([10, 20, 30, 400])
        fig = plot_log_histogram(s, cap=300, xlabel="X", title="T")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_color(self):
        fig = plot_log_histogram(self._values(), cap=500, xlabel="X", title="T", color="coral")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCorrelationHeatmap:
    def _make_df(self):
        rng = np.random.default_rng(0)
        n = 50
        a = rng.normal(size=n)
        return pl.DataFrame({
            "x": a.tolist(),
            "y": (a * 2 + rng.normal(size=n) * 0.5).tolist(),
            "z": rng.normal(size=n).tolist(),
        })

    def test_returns_figure(self):
        df = self._make_df()
        fig = plot_correlation_heatmap(df, df.columns)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_subset_of_columns(self):
        df = self._make_df()
        fig = plot_correlation_heatmap(df, ["x", "y"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        df = self._make_df()
        fig = plot_correlation_heatmap(df, df.columns, title="My Heatmap")
        assert fig.axes[0].get_title() == "My Heatmap"
        plt.close(fig)

    def test_custom_figsize(self):
        df = self._make_df()
        fig = plot_correlation_heatmap(df, df.columns, figsize=(6, 5))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(5.0)
        plt.close(fig)

    def test_single_column(self):
        # 1x1 correlation matrix (always 1.0) — should not raise
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        fig = plot_correlation_heatmap(df, ["x"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
