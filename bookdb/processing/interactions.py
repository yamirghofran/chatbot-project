
from __future__ import annotations

import polars as pl

GOODREADS_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_goodreads_timestamps(
    lf: pl.LazyFrame,
    cols: list[str],
    strict: bool = False,
) -> pl.LazyFrame:
    """Parse one or more Goodreads date string columns into Unix timestamp columns.

    Each input column ``col`` is parsed and stored as ``ts_<col>`` (Int64, seconds).
    The original string columns are dropped.

    Args:
        lf: Source lazy frame.
        cols: List of column names containing Goodreads date strings.
        strict: If True, raise on unparseable values; otherwise fill with null.

    Returns:
        Lazy frame with original columns replaced by ``ts_<col>`` Unix timestamps.
    """
    return lf.with_columns(
        pl.col(c)
        .str.strptime(pl.Datetime, GOODREADS_DATE_FORMAT, strict=strict)
        .dt.epoch("s")
        .alias(f"ts_{c}")
        for c in cols
    ).drop(cols)


def calculate_weight_bpr(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate BPR confidence weight for each user-book interaction.

    Higher weight = stronger signal the user liked this book.

    Formula:
        weight = 1.0  (base interaction)
               + 2.0 * is_read
               + (rating - 1) if rating > 0, else 0
               + 3.0 * is_reviewed

    Args:
        df: DataFrame with columns: rating (Int), is_read (Bool), is_reviewed (Bool).

    Returns:
        Input DataFrame with an added Float32 column ``weight``.
    """
    return (
        df.with_columns(
            pl.when(pl.col("rating") > 0)
            .then(pl.col("rating") - 1)
            .otherwise(0.0)
            .alias("rating_contrib")
        )
        .with_columns(
            (
                1.0
                + 2.0 * pl.col("is_read").cast(pl.Float32)
                + pl.col("rating_contrib")
                + 3.0 * pl.col("is_reviewed").cast(pl.Float32)
            )
            .cast(pl.Float32)
            .alias("weight")
        )
        .drop("rating_contrib")
    )


def calculate_weight_sar(
    df: pl.DataFrame,
    base_weight: float = 0.1,
    rating_weight: float = 0.4,
    is_read_weight: float = 0.2,
    review_weight: float = 0.3,
) -> pl.DataFrame:
    """Calculate SAR interaction weight for each user-book interaction.

    Weights are normalised to [0, 1] and configurable. Review presence is
    detected from the ``review_text_incomplete`` column.

    Args:
        df: DataFrame with columns: rating (Int), is_read (Bool),
            review_text_incomplete (Str).
        base_weight: Constant added to every interaction.
        rating_weight: Multiplier for normalised rating (rating / 5).
        is_read_weight: Multiplier for the is_read flag.
        review_weight: Multiplier for having a non-empty review.

    Returns:
        Input DataFrame with an added Float32 column ``weight``.
    """
    return (
        df.with_columns(
            (pl.col("rating").fill_null(0) / 5).alias("rating_norm"),
            pl.col("is_read").cast(pl.Float32).alias("read_val"),
            (pl.col("review_text_incomplete").str.len_chars().fill_null(0) > 0)
            .cast(pl.Float32)
            .alias("has_review_val"),
        )
        .with_columns(
            (
                base_weight
                + rating_weight * pl.col("rating_norm")
                + is_read_weight * pl.col("read_val")
                + review_weight * pl.col("has_review_val")
            )
            .cast(pl.Float32)
            .alias("weight")
        )
        .drop(["rating_norm", "read_val", "has_review_val"])
    )


def convert_to_unix_timestamp(
    df: pl.DataFrame,
    timestamp_col: str = "date_updated",
) -> pl.DataFrame:
    """Parse a date string column into a Unix timestamp (seconds) column.

    The input format is the Twitter-style date used in the Goodreads dataset,
    e.g. ``"Thu Mar 22 15:43:00 -0700 2012"``.

    Args:
        df: DataFrame containing the date string column.
        timestamp_col: Name of the column to parse.

    Returns:
        Input DataFrame with an added Int64 column ``timestamp``.
    """
    return df.with_columns(
        pl.col(timestamp_col)
        .str.strptime(pl.Datetime, format="%a %b %d %H:%M:%S %z %Y")
        .dt.epoch("s")
        .alias("timestamp")
    )
