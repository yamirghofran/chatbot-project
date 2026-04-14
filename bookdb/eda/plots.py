
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def plot_rating_distribution(
    df: pl.DataFrame,
    rating_col: str = "rating",
    figsize: tuple[int, int] = (14, 5),
    label_in_millions: bool = True,
) -> plt.Figure:
    """Plot rating distribution as two side-by-side bar charts.

    Left panel shows all ratings (0–5); right panel excludes unrated (rating=0).
    Bars are annotated with counts in millions (or raw counts if label_in_millions=False).

    Args:
        df: DataFrame containing the rating column.
        rating_col: Name of the ratings column (default ``"rating"``).
        figsize: Figure size (width, height) in inches.
        label_in_millions: If True, annotate bars as "X.XM"; otherwise use raw counts.

    Returns:
        Matplotlib Figure.
    """
    rating_all = df.group_by(rating_col).len().sort(rating_col)
    rating_nonzero = (
        df.filter(pl.col(rating_col) > 0).group_by(rating_col).len().sort(rating_col)
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    panels = [
        (axes[0], rating_all, "Rating Distribution (0 = unrated)", [0, 1, 2, 3, 4, 5]),
        (axes[1], rating_nonzero, "Rating Distribution (Excluding Unrated)", [1, 2, 3, 4, 5]),
    ]

    for ax, data, title, ticks in panels:
        ax.bar(
            data[rating_col].to_list(),
            data["len"].to_list(),
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xticks(ticks)
        for r, c in zip(data[rating_col].to_list(), data["len"].to_list()):
            label = f"{c / 1e6:.1f}M" if label_in_millions else str(c)
            ax.text(r, c, label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def plot_log_histogram(
    values,
    cap: float,
    xlabel: str,
    title: str,
    figsize: tuple[int, int] = (14, 5),
    color: str = "steelblue",
) -> plt.Figure:
    """Plot raw (capped) and log-transformed histograms side-by-side.

    Args:
        values: Array-like of numeric values (numpy array, list, or Polars Series).
        cap: Upper cap for the left (raw) histogram.
        xlabel: X-axis label for the raw histogram (e.g. ``"Word Count"``).
        title: Shared title prefix; raw panel uses it as-is, log panel appends " (log scale)".
        figsize: Figure size (width, height) in inches.
        color: Bar color for both panels.

    Returns:
        Matplotlib Figure.
    """
    values_np = np.asarray(values)
    fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=figsize)

    ax_raw.hist(np.clip(values_np, 0, cap), bins=100, edgecolor="black", alpha=0.7, color=color)
    ax_raw.set_xlabel(f"{xlabel} (capped at {cap:,.0f})")
    ax_raw.set_ylabel("Frequency")
    ax_raw.set_title(title)

    ax_log.hist(np.log1p(values_np), bins=100, edgecolor="black", alpha=0.7, color=color)
    ax_log.set_xlabel(f"log({xlabel} + 1)")
    ax_log.set_ylabel("Frequency")
    ax_log.set_title(f"{title} (log scale)")

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pl.DataFrame,
    cols: list[str],
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot a Pearson correlation heatmap for the specified columns.

    Args:
        df: Polars DataFrame containing the columns.
        cols: Column names to include in the correlation.
        title: Plot title.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib Figure.
    """
    corr = df.select(cols).to_pandas().corr(method="pearson")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", square=True, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
