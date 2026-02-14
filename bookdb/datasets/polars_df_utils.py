# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging
from functools import lru_cache, wraps
from typing import Any

import numpy as np
import polars as pl

from bookdb.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
)


logger = logging.getLogger(__name__)


def user_item_pairs(
    user_df: pl.DataFrame,
    item_df: pl.DataFrame,
    user_col: str = DEFAULT_USER_COL,
    item_col: str = DEFAULT_ITEM_COL,
    user_item_filter_df: pl.DataFrame | None = None,
    shuffle: bool = True,
    seed: int | None = None,
) -> pl.DataFrame:
    """Get all pairs of users and items data.

    Args:
        user_df: User data containing unique user ids and maybe their features.
        item_df: Item data containing unique item ids and maybe their features.
        user_col: User id column name.
        item_col: Item id column name.
        user_item_filter_df: User-item pairs to be used as a filter.
        shuffle: If True, shuffles the result.
        seed: Random seed for shuffle.

    Returns:
        All pairs of user-item from user_df and item_df, excepting the pairs in user_item_filter_df.
    """
    # Get all user-item pairs using cross join (Polars native way)
    users_items = user_df.join(item_df, how="cross")

    # Filter out existing pairs if filter dataframe is provided
    if user_item_filter_df is not None:
        users_items = filter_by(users_items, user_item_filter_df, [user_col, item_col])

    if shuffle:
        users_items = users_items.sample(fraction=1.0, shuffle=True, seed=seed)

    return users_items


def filter_by(
    df: pl.DataFrame,
    filter_by_df: pl.DataFrame,
    filter_by_cols: list[str],
) -> pl.DataFrame:
    """From the input DataFrame `df`, remove the records whose target column
    `filter_by_cols` values are exist in the filter-by DataFrame `filter_by_df`.

    Args:
        df: Source dataframe.
        filter_by_df: Filter dataframe.
        filter_by_cols: Filter columns.

    Returns:
        Dataframe filtered by `filter_by_df` on `filter_by_cols`.
    """
    # Use anti-join to exclude rows that exist in filter_by_df
    return df.join(
        filter_by_df.select(filter_by_cols).unique(subset=filter_by_cols),
        on=filter_by_cols,
        how="anti",
    )


class LibffmConverter:
    """Converts an input dataframe to another dataframe in libffm format.

    A text file of the converted Dataframe is optionally generated.

    Note:
        The input dataframe is expected to represent the feature data in the following schema:

        .. code-block:: python

            |field-1|field-2|...|field-n|rating|
            |feature-1-1|feature-2-1|...|feature-n-1|1|
            |feature-1-2|feature-2-2|...|feature-n-2|0|
            ...
            |feature-1-i|feature-2-j|...|feature-n-k|0|

        Where
        1. each `field-*` is the column name of the dataframe (column of label/rating is excluded), and
        2. `feature-*-*` can be either a string or a numerical value, representing the categorical variable or
        actual numerical variable of the feature value in the field, respectively.
        3. If there are ordinal variables represented in int types, users should make sure these columns
        are properly converted to string type.

        The above data will be converted to the libffm format by following the convention as explained in
        `this paper <https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`_.

        i.e. `<field_index>:<field_feature_index>:1` or `<field_index>:<field_feature_index>:<field_feature_value>`,
        depending on the data type of the features in the original dataframe.

    Args:
        filepath: Path to save the converted data.

    Attributes:
        field_count: Count of field in the libffm format data.
        feature_count: Count of feature in the libffm format data.
        filepath: File path where the output is stored - it can be None or a string.

    Examples:
        >>> import polars as pl
        >>> df_feature = pl.DataFrame({
                'rating': [1, 0, 0, 1, 1],
                'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
                'field2': [3, 4, 5, 6, 7],
                'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
                'field4': ['1', '2', '3', '4', '5']
            })
        >>> converter = LibffmConverter().fit(df_feature, col_rating='rating')
        >>> df_out = converter.transform(df_feature)
        >>> df_out
        shape: (5, 5)
        ┌────────┬────────┬────────┬──────────┬────────┐
        │ rating ┆ field1 ┆ field2 ┆ field3   ┆ field4 │
        │ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---    │
        │ i64    ┆ str    ┆ str    ┆ str      ┆ str    │
        ╞════════╪════════╪════════╪══════════╪════════╡
        │ 1      ┆ 1:1:1  ┆ 2:4:3  ┆ 3:5:1.0  ┆ 4:6:1  │
        │ 0      ┆ 1:2:1  ┆ 2:4:4  ┆ 3:5:2.0  ┆ 4:7:1  │
        │ 0      ┆ 1:3:1  ┆ 2:4:5  ┆ 3:5:3.0  ┆ 4:8:1  │
        │ 1      ┆ 1:3:1  ┆ 2:4:6  ┆ 3:5:4.0  ┆ 4:9:1  │
        │ 1      ┆ 1:3:1  ┆ 2:4:7  ┆ 3:5:5.0  ┆ 4:10:1 │
        └────────┴────────┴────────┴──────────┴────────┘
    """

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath
        self.col_rating: str | None = None
        self.field_names: list[str] | None = None
        self.field_count: int | None = None
        self.feature_count: int | None = None
        self.field_feature_dict: dict[tuple[str, Any], int] = {}

    def fit(self, df: pl.DataFrame, col_rating: str = DEFAULT_RATING_COL) -> "LibffmConverter":
        """Fit the dataframe for libffm format.
        This method does nothing but check the validity of the input columns.

        Args:
            df: Input Polars dataframe.
            col_rating: Rating column name.

        Returns:
            The instance of the converter.
        """
        # Check column types - Polars uses different type system
        valid_types = (pl.String, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                       pl.Float32, pl.Float64)
        
        for col in df.columns:
            if df[col].dtype not in valid_types:
                raise TypeError("Input columns should be only string and/or numeric types.")

        if col_rating not in df.columns:
            raise TypeError(f"Column of {col_rating} is not in input dataframe columns")

        self.col_rating = col_rating
        self.field_names = [c for c in df.columns if c != col_rating]

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform an input dataset with the same schema (column names and dtypes) to libffm format
        by using the fitted converter.

        Args:
            df: Input Polars dataframe.

        Returns:
            Output libffm format dataframe.
        """
        if self.col_rating is None or self.field_names is None:
            raise ValueError("Converter has not been fitted. Call fit() first.")

        if self.col_rating not in df.columns:
            raise ValueError(
                f"Input dataset does not contain the label column {self.col_rating} "
                "in the fitting dataset"
            )

        if not all(x in df.columns for x in self.field_names):
            raise ValueError(
                "Not all columns in the input dataset appear in the fitting dataset"
            )

        # Encode field-feature.
        idx = 1
        self.field_feature_dict = {}
        
        for field in self.field_names:
            col_dtype = df[field].dtype
            unique_features = df[field].unique().to_list()
            
            for feature in unique_features:
                if (field, feature) not in self.field_feature_dict:
                    self.field_feature_dict[(field, feature)] = idx
                    if col_dtype == pl.String:
                        idx += 1
            if col_dtype != pl.String:
                idx += 1

        self.field_count = len(self.field_names)
        self.feature_count = idx - 1

        def _convert(field: str, feature: Any, field_index: int) -> str:
            field_feature_index = self.field_feature_dict[(field, feature)]
            if isinstance(feature, str):
                feature_val = 1
            else:
                feature_val = feature
            return f"{field_index}:{field_feature_index}:{feature_val}"

        # Transform each field column
        result_df = df.clone()
        for col_index, col in enumerate(self.field_names):
            result_df = result_df.with_columns(
                pl.col(col).map_elements(
                    lambda x, c=col, ci=col_index + 1: _convert(c, x, ci),
                    return_dtype=pl.String
                ).alias(col)
            )

        # Reorder columns to put rating first
        column_order = [self.col_rating] + self.field_names
        result_df = result_df.select(column_order)

        if self.filepath is not None:
            # Write to file in libffm format
            with open(self.filepath, "w") as f:
                for row in result_df.iter_rows():
                    f.write(" ".join(str(v) for v in row) + "\n")

        return result_df

    def fit_transform(
        self, df: pl.DataFrame, col_rating: str = DEFAULT_RATING_COL
    ) -> pl.DataFrame:
        """Do fit and transform in a row.

        Args:
            df: Input Polars dataframe.
            col_rating: Rating column name.

        Returns:
            Output libffm format dataframe.
        """
        return self.fit(df, col_rating=col_rating).transform(df)

    def get_params(self) -> dict[str, Any]:
        """Get parameters (attributes) of the libffm converter.

        Returns:
            A dictionary that contains parameters field count, feature count, and file path.
        """
        return {
            "field_count": self.field_count,
            "feature_count": self.feature_count,
            "filepath": self.filepath,
        }


def negative_feedback_sampler(
    df: pl.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_label: str = DEFAULT_LABEL_COL,
    col_feedback: str = "feedback",
    ratio_neg_per_user: int = 1,
    pos_value: int = 1,
    neg_value: int = 0,
    seed: int = 42,
) -> pl.DataFrame:
    """Utility function to sample negative feedback from user-item interaction dataset.

    This negative sampling function will take the user-item interaction data to create
    binarized feedback, i.e., 1 and 0 indicate positive and negative feedback,
    respectively.

    Negative sampling is used in the literature frequently to generate negative samples
    from a user-item interaction data.

    See for example the `neural collaborative filtering paper <https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf>`_.

    Args:
        df: Input data that contains user-item tuples.
        col_user: User id column name.
        col_item: Item id column name.
        col_label: Label column name in df.
        col_feedback: Feedback column name in the returned data frame; it is used for
            the generated column of positive and negative feedback.
        ratio_neg_per_user: Ratio of negative feedback w.r.t to the number of positive
            feedback for each user. If the samples exceed the number of total possible
            negative feedback samples, it will be reduced to the number of all the
            possible samples.
        pos_value: Value of positive feedback.
        neg_value: Value of negative feedback.
        seed: Seed for the random state of the sampling function.

    Returns:
        Data with negative feedback.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
                'userID': [1, 2, 3],
                'itemID': [1, 2, 3],
                'rating': [5, 5, 5]
            })
        >>> df_neg_sampled = negative_feedback_sampler(
                df, col_user='userID', col_item='itemID', ratio_neg_per_user=1
            )
        >>> df_neg_sampled
        shape: (6, 3)
        ┌────────┬────────┬──────────┐
        │ userID ┆ itemID ┆ feedback │
        │ ---    ┆ ---    ┆ ---      │
        │ i64    ┆ i64    ┆ i64      │
        ╞════════╪════════╪══════════╡
        │ 1      ┆ 1      ┆ 1        │
        │ 1      ┆ 2      ┆ 0        │
        │ 2      ┆ 2      ┆ 1        │
        │ 2      ┆ 1      ┆ 0        │
        │ 3      ┆ 3      ┆ 1        │
        │ 3      ┆ 1      ┆ 0        │
        └────────┴────────┴──────────┘
    """
    # Get all items
    items = df[col_item].unique().to_list()
    rng = np.random.default_rng(seed=seed)

    def sample_items(user_df: pl.DataFrame) -> pl.DataFrame:
        """Sample negative items for a specific user."""
        n_u = len(user_df)
        neg_sample_size = max(round(n_u * ratio_neg_per_user), 1)
        
        # Get items already interacted by user
        user_items = set(user_df[col_item].to_list())
        
        # Draw items and keep those not already in user's items
        sample_size = min(n_u + neg_sample_size, len(items))
        items_sample = rng.choice(items, sample_size, replace=False)
        new_items = np.setdiff1d(items_sample, list(user_items))[:neg_sample_size]
        
        # Create negative samples dataframe
        if len(new_items) > 0:
            user_id = user_df[col_user][0]
            neg_df = pl.DataFrame({
                col_user: [user_id] * len(new_items),
                col_item: new_items,
                col_feedback: [neg_value] * len(new_items),
            })
            # Add positive label to original data
            pos_df = user_df.select(col_user, col_item).with_columns(
                pl.lit(pos_value).alias(col_feedback)
            )
            return pl.concat([pos_df, neg_df])
        else:
            return user_df.select(col_user, col_item).with_columns(
                pl.lit(pos_value).alias(col_feedback)
            )

    # Process each user group and concatenate results
    result_dfs = []
    for user_id in df[col_user].unique().to_list():
        user_df = df.filter(pl.col(col_user) == user_id)
        result_dfs.append(sample_items(user_df))

    return pl.concat(result_dfs)


def has_columns(df: pl.DataFrame, columns: list[str] | set[str]) -> bool:
    """Check if DataFrame has necessary columns.

    Args:
        df: Polars DataFrame.
        columns: Columns to check for.

    Returns:
        True if DataFrame has specified columns.
    """
    if not isinstance(columns, set):
        columns = set(columns)
    return columns.issubset(set(df.columns))


def has_same_base_dtype(
    df_1: pl.DataFrame,
    df_2: pl.DataFrame,
    columns: list[str] | None = None,
) -> bool:
    """Check if specified columns have the same base dtypes across both DataFrames.

    Args:
        df_1: First DataFrame.
        df_2: Second DataFrame.
        columns: Columns to check, None checks all columns.

    Returns:
        True if DataFrames columns have the same base dtypes.
    """
    if columns is None:
        if set(df_1.columns) != set(df_2.columns):
            logger.error(
                "Cannot test all columns because they are not all shared across DataFrames"
            )
            return False
        columns = df_1.columns

    if not (has_columns(df=df_1, columns=columns) and has_columns(df=df_2, columns=columns)):
        return False

    # Map Polars dtypes to their base types
    def get_base_type(dtype: pl.DataType) -> type:
        """Get the base Python type for a Polars dtype."""
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            return int
        elif dtype in (pl.Float32, pl.Float64):
            return float
        elif dtype == pl.String:
            return str
        elif dtype == pl.Boolean:
            return bool
        else:
            return type(dtype)

    result = True
    for column in columns:
        base_type_1 = get_base_type(df_1[column].dtype)
        base_type_2 = get_base_type(df_2[column].dtype)
        if base_type_1 != base_type_2:
            logger.error(f"Column '{column}' does not have the same base datatype")
            result = False

    return result


class PolarsHash:
    """Wrapper class to allow Polars objects (DataFrames or Series) to be hashable."""

    __slots__ = "polars_object"

    def __init__(self, polars_object: pl.DataFrame | pl.Series):
        """Initialize class.

        Args:
            polars_object: Polars DataFrame or Series.

        Raises:
            TypeError: If input is not a Polars DataFrame or Series.
        """
        if not isinstance(polars_object, (pl.DataFrame, pl.Series)):
            raise TypeError("Can only wrap Polars DataFrame or Series objects")
        self.polars_object = polars_object

    def __eq__(self, other: object) -> bool:
        """Overwrite equality comparison.

        Args:
            other: Object to compare.

        Returns:
            Whether other object is the same as this one.
        """
        if not isinstance(other, PolarsHash):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Overwrite hash operator for use with Polars objects.

        Returns:
            Hashed value of object.
        """
        hashable = tuple(self.polars_object.to_numpy().tobytes())
        if isinstance(self.polars_object, pl.DataFrame):
            hashable += tuple(self.polars_object.columns)
        else:
            hashable += (self.polars_object.name,)
        return hash(hashable)


def lru_cache_df(maxsize: int | None, typed: bool = False):
    """Least-recently-used cache decorator for Polars DataFrames.

    Decorator to wrap a function with a memoizing callable that saves up to the
    maxsize most recent calls. It can save time when an expensive or I/O bound
    function is periodically called with the same arguments.

    Inspired by the `lru_cache function <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.

    Args:
        maxsize: Max size of cache, if set to None cache is boundless.
        typed: Arguments of different types are cached separately.
    """

    def to_polars_hash(val: Any) -> Any:
        """Return PolarsHash object if input is a DataFrame otherwise return input unchanged."""
        return PolarsHash(val) if isinstance(val, (pl.DataFrame, pl.Series)) else val

    def from_polars_hash(val: Any) -> Any:
        """Extract DataFrame/Series if input is PolarsHash object otherwise return input unchanged."""
        return val.polars_object if isinstance(val, PolarsHash) else val

    def decorating_function(user_function):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            # Convert DataFrames/Series in args and kwargs to PolarsHash objects
            args = tuple([to_polars_hash(a) for a in args])
            kwargs = {k: to_polars_hash(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            # Get DataFrames/Series from PolarsHash objects in args and kwargs
            args = tuple([from_polars_hash(a) for a in args])
            kwargs = {k: from_polars_hash(v) for k, v in kwargs.items()}
            return user_function(*args, **kwargs)

        # Retain lru_cache attributes (added dynamically by lru_cache)
        wrapper.cache_info = cached_wrapper.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached_wrapper.cache_clear  # type: ignore[attr-defined]

        return wrapper

    return decorating_function
