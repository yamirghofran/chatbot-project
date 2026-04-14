
from __future__ import annotations

import polars as pl


def iqr_outlier_bounds(
    series: pl.Series,
    multiplier: float = 3.0,
) -> tuple[float, float]:
    """Compute IQR-based lower and upper outlier bounds for a Polars Series.

    Formula:
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

    Args:
        series: Polars Series of numeric values.
        multiplier: IQR multiplier (default 3.0). Use 1.5 for standard Tukey fences.

    Returns:
        ``(lower_bound, upper_bound)`` tuple of floats.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr
