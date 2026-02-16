# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""BPR (Bayesian Personalized Ranking) model using the implicit library.

This implementation uses the implicit library for memory-efficient training
on large-scale implicit feedback datasets. It handles string-to-integer ID
mapping internally and processes recommendations in batches to avoid memory
issues with large user/item spaces.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

try:
    from implicit.cpu.bpr import BayesianPersonalizedRanking
except ImportError:
    from implicit.bpr import BayesianPersonalizedRanking

from bookdb.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)

logger = logging.getLogger(__name__)


class BPR:
    """Bayesian Personalized Ranking model using the implicit library.

    This class wraps the implicit library's BPR implementation, providing:
    - Automatic string-to-integer ID mapping for memory efficiency
    - Batched recommendation generation to handle large datasets
    - Polars DataFrame integration

    Args:
        factors: Number of latent factors. Default: 64
        iterations: Number of training iterations. Default: 100
        learning_rate: Learning rate for SGD. Default: 0.01
        regularization: L2 regularization factor. Default: 0.01
        verify_negative_samples: Whether to verify negative samples are truly negative.
        random_state: Random seed for reproducibility.
        num_threads: Number of threads for parallel training. 0 = auto.

    Example:
        >>> model = BPR(factors=64, iterations=100)
        >>> model.fit(train_df, col_user="user_id", col_item="item_id", col_rating="rating")
        >>> recommendations = model.recommend_k_items(test_df, top_k=100, remove_seen=True)
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        verify_negative_samples: bool = True,
        random_state: Optional[int] = None,
        num_threads: int = 0,
    ):
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state
        self.num_threads = num_threads

        # Initialize the underlying implicit model
        self._model = BayesianPersonalizedRanking(
            factors=factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            verify_negative_samples=verify_negative_samples,
            random_state=random_state,
            num_threads=num_threads,
        )

        # ID mappings (set after fit)
        self._user_id_map: dict[str, int] = {}
        self._item_id_map: dict[str, int] = {}
        self._reverse_user_map: dict[int, str] = {}
        self._reverse_item_map: dict[int, str] = {}
        self._user_item_matrix: Optional[csr_matrix] = None
        self._col_user: str = DEFAULT_USER_COL
        self._col_item: str = DEFAULT_ITEM_COL
        # Store original dtypes to preserve types in output
        self._user_dtype: Optional[pl.DataType] = None
        self._item_dtype: Optional[pl.DataType] = None

    @property
    def n_users(self) -> int:
        """Number of unique users in the training data."""
        return len(self._user_id_map)

    @property
    def n_items(self) -> int:
        """Number of unique items in the training data."""
        return len(self._item_id_map)

    @property
    def user_factors(self) -> np.ndarray:
        """User latent factor matrix of shape (n_users, factors)."""
        return self._model.user_factors

    @property
    def item_factors(self) -> np.ndarray:
        """Item latent factor matrix of shape (n_items, factors)."""
        return self._model.item_factors

    def _create_mappings(
        self,
        df: pl.DataFrame,
        col_user: str,
        col_item: str,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Create string-to-integer ID mappings.

        Args:
            df: Input DataFrame with user/item columns.
            col_user: Name of the user column.
            col_item: Name of the item column.

        Returns:
            Tuple of (user_id_map, item_id_map)
        """
        # Get unique users and items
        unique_users = df.select(col_user).unique().to_series()
        unique_items = df.select(col_item).unique().to_series()

        # Create mappings (string/any -> int)
        user_id_map = {str(uid): idx for idx, uid in enumerate(unique_users)}
        item_id_map = {str(iid): idx for idx, iid in enumerate(unique_items)}

        logger.info(f"Created mappings: {len(user_id_map)} users, {len(item_id_map)} items")

        return user_id_map, item_id_map

    def _build_sparse_matrix(
        self,
        df: pl.DataFrame,
        user_id_map: dict[str, int],
        item_id_map: dict[str, int],
        col_user: str,
        col_item: str,
        col_rating: Optional[str] = None,
    ) -> csr_matrix:
        """Build a CSR sparse matrix from the DataFrame.

        Args:
            df: DataFrame with user, item, and optionally rating columns.
            user_id_map: Mapping from user IDs to integer indices.
            item_id_map: Mapping from item IDs to integer indices.
            col_user: Name of the user column.
            col_item: Name of the item column.
            col_rating: Name of the rating/confidence column. If None, uses 1.0.

        Returns:
            CSR sparse matrix of shape (n_users, n_items)
        """
        n_users = len(user_id_map)
        n_items = len(item_id_map)

        # Map user/item IDs to integers
        df_mapped = df.select(
            pl.col(col_user).cast(pl.String).replace_strict(user_id_map, return_dtype=pl.Int32).alias("user_idx"),
            pl.col(col_item).cast(pl.String).replace_strict(item_id_map, return_dtype=pl.Int32).alias("item_idx"),
        )

        if col_rating and col_rating in df.columns:
            ratings = df[col_rating].to_numpy()
        else:
            ratings = np.ones(len(df))

        user_indices = df_mapped["user_idx"].to_numpy()
        item_indices = df_mapped["item_idx"].to_numpy()

        # Build CSR matrix
        matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"Built sparse matrix: {matrix.shape}, {matrix.nnz} non-zero entries")

        return matrix

    def fit(
        self,
        df: pl.DataFrame,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_rating: Optional[str] = None,
    ) -> "BPR":
        """Fit the BPR model on the training data.

        Args:
            df: Training data as a Polars DataFrame.
            col_user: Name of the user column.
            col_item: Name of the item column.
            col_rating: Name of the rating/confidence column. If None, treats as implicit (1.0).

        Returns:
            self (fitted model)
        """
        logger.info("Fitting BPR model...")
        logger.info(f"  factors={self.factors}, iterations={self.iterations}, lr={self.learning_rate}")

        # Store column names and original dtypes for later use
        self._col_user = col_user
        self._col_item = col_item
        self._user_dtype = df.schema[col_user]
        self._item_dtype = df.schema[col_item]

        # Create ID mappings
        self._user_id_map, self._item_id_map = self._create_mappings(df, col_user, col_item)

        # Create reverse mappings for output
        self._reverse_user_map = {v: k for k, v in self._user_id_map.items()}
        self._reverse_item_map = {v: k for k, v in self._item_id_map.items()}

        # Build sparse user-item matrix
        self._user_item_matrix = self._build_sparse_matrix(
            df, self._user_id_map, self._item_id_map, col_user, col_item, col_rating
        )

        # Fit the implicit model
        self._model.fit(self._user_item_matrix, show_progress=True)

        logger.info(f"BPR model fitted: {self.n_users} users, {self.n_items} items")

        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        filter_already_liked_items: bool = True,
    ) -> pl.DataFrame:
        """Recommend items for a single user.

        Args:
            user_id: User ID to generate recommendations for.
            top_k: Number of recommendations to return.
            filter_already_liked_items: Whether to exclude items the user has already interacted with.

        Returns:
            DataFrame with columns [user_col, item_col, prediction]
        """
        if user_id not in self._user_id_map:
            logger.warning(f"User {user_id} not in training data, returning empty recommendations")
            return pl.DataFrame({
                self._col_user: [],
                self._col_item: [],
                DEFAULT_PREDICTION_COL: [],
            })

        user_idx = self._user_id_map[user_id]

        # Get recommendations from implicit model
        assert self._user_item_matrix is not None  # for type checker
        ids, scores = self._model.recommend(
            user_idx,
            self._user_item_matrix[user_idx],
            N=top_k,
            filter_already_liked_items=filter_already_liked_items,
        )

        # Convert back to original IDs
        original_user_ids = [user_id] * len(ids)
        original_item_ids = [self._reverse_item_map[iid] for iid in ids]

        df = pl.DataFrame({
            self._col_user: original_user_ids,
            self._col_item: original_item_ids,
            DEFAULT_PREDICTION_COL: scores.astype(np.float32),
        })

        # Cast columns to original dtypes to match input data types
        if self._user_dtype is not None:
            df = df.with_columns(pl.col(self._col_user).cast(self._user_dtype))
        if self._item_dtype is not None:
            df = df.with_columns(pl.col(self._col_item).cast(self._item_dtype))

        return df

    def recommend_k_items(
        self,
        data: pl.DataFrame,
        top_k: Optional[int] = None,
        remove_seen: bool = False,
        col_user: Optional[str] = None,
        col_item: Optional[str] = None,
        col_prediction: str = DEFAULT_PREDICTION_COL,
        batch_size: int = 1000,
    ) -> pl.DataFrame:
        """Generate top-k recommendations for all users in the data.

        This method processes users in batches to avoid memory issues with
        large user/item spaces.

        Args:
            data: DataFrame containing users to generate recommendations for.
            top_k: Number of recommendations per user. If None, uses min(100, n_items).
            remove_seen: Whether to exclude items the user has already interacted with.
            col_user: Name of the user column. Uses default from fit() if None.
            col_item: Name of the item column. Uses default from fit() if None.
            col_prediction: Name of the prediction column.
            batch_size: Number of users to process at once. Smaller = less memory.

        Returns:
            DataFrame with columns [col_user, col_item, col_prediction]
        """
        col_user = col_user or self._col_user
        col_item = col_item or self._col_item

        # Get unique users from the data
        users = data.select(col_user).unique().to_series().to_list()

        # Determine top_k
        top_k = min(top_k or 100, self.n_items)

        logger.info(f"Generating top-{top_k} recommendations for {len(users)} users (batch_size={batch_size})")

        # Process users in batches
        all_recommendations = []

        for i in range(0, len(users), batch_size):
            batch_users = users[i:i + batch_size]
            batch_recs = []

            for user_id in batch_users:
                user_id_str = str(user_id)

                if user_id_str not in self._user_id_map:
                    continue

                user_idx = self._user_id_map[user_id_str]

                # Get recommendations
                assert self._user_item_matrix is not None  # for type checker
                ids, scores = self._model.recommend(
                    user_idx,
                    self._user_item_matrix[user_idx],
                    N=top_k,
                    filter_already_liked_items=remove_seen,
                )

                # Convert to original IDs
                for item_idx, score in zip(ids, scores):
                    batch_recs.append({
                        col_user: user_id,  # Preserve original user_id type
                        col_item: self._reverse_item_map[item_idx],
                        col_prediction: float(score),
                    })

            if batch_recs:
                all_recommendations.append(pl.DataFrame(batch_recs))

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"  Processed {min(i + batch_size, len(users))}/{len(users)} users")

        if not all_recommendations:
            logger.warning("No recommendations generated")
            return pl.DataFrame({
                col_user: [],
                col_item: [],
                col_prediction: [],
            })

        # Concatenate all batches
        recommendations = pl.concat(all_recommendations)

        # Cast columns to original dtypes to match input data types
        if self._user_dtype is not None:
            recommendations = recommendations.with_columns(
                pl.col(col_user).cast(self._user_dtype)
            )
        if self._item_dtype is not None:
            recommendations = recommendations.with_columns(
                pl.col(col_item).cast(self._item_dtype)
            )

        logger.info(f"Generated {len(recommendations)} recommendations for {recommendations.select(col_user).n_unique()} users")

        return recommendations

    def similar_items(
        self,
        item_id: str,
        top_k: int = 10,
    ) -> pl.DataFrame:
        """Find items most similar to the given item.

        Args:
            item_id: Item ID to find similar items for.
            top_k: Number of similar items to return.

        Returns:
            DataFrame with columns [item_col, "similar_item_id", "similarity"]
        """
        if item_id not in self._item_id_map:
            logger.warning(f"Item {item_id} not in training data")
            return pl.DataFrame()

        item_idx = self._item_id_map[item_id]

        ids, scores = self._model.similar_items(item_idx, N=top_k)

        return pl.DataFrame({
            self._col_item: [item_id] * len(ids),
            "similar_item_id": [self._reverse_item_map[iid] for iid in ids],
            "similarity": scores.astype(np.float32),
        })

    def save(self, path: str | Path):
        """Save the model to disk.

        Args:
            path: Directory path to save the model.
        """
        import json
        import pickle
        from pathlib import Path

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the implicit model
        with open(save_path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        # Save mappings
        mappings = {
            "user_id_map": self._user_id_map,
            "item_id_map": self._item_id_map,
            "col_user": self._col_user,
            "col_item": self._col_item,
        }
        with open(save_path / "mappings.json", "w") as f:
            json.dump(mappings, f)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPR":
        """Load a model from disk.

        Args:
            path: Directory path containing the saved model.

        Returns:
            Loaded BPR model
        """
        import json
        import pickle

        load_path = Path(path)

        # Load the implicit model
        with open(load_path / "model.pkl", "rb") as f:
            model = pickle.load(f)

        # Create BPR instance and restore state
        bpr = cls(
            factors=model.factors,
            iterations=model.iterations,
            learning_rate=model.learning_rate,
            regularization=model.regularization,
        )
        bpr._model = model

        # Load mappings
        with open(load_path / "mappings.json", "r") as f:
            mappings = json.load(f)

        bpr._user_id_map = mappings["user_id_map"]
        bpr._item_id_map = mappings["item_id_map"]
        bpr._reverse_user_map = {v: k for k, v in bpr._user_id_map.items()}
        bpr._reverse_item_map = {v: k for k, v in bpr._item_id_map.items()}
        bpr._col_user = mappings["col_user"]
        bpr._col_item = mappings["col_item"]

        logger.info(f"Model loaded from {load_path}")

        return bpr
