# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
Evaluation metrics for recommender systems.

This module provides standard evaluation metrics for recommendation systems including:
- Precision@k
- Recall@k  
- Mean Average Precision (MAP@k)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Hit Rate@k
- Coverage
- Diversity
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def precision_at_k(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Precision@k.

    Precision@k = (Recommended items that are relevant) / k

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Average Precision@k across all users
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    precisions = []

    for user in users.to_series().to_list():
        user_recs = top_k_recs.filter(pl.col(col_user) == user).select(col_item).to_series().to_list()
        user_truth = ground_truth.filter(pl.col(col_user) == user).select(col_item).to_series().to_list()

        if len(user_truth) > 0:
            hits = len(set(user_recs) & set(user_truth))
            precision = hits / min(k, len(user_recs)) if len(user_recs) > 0 else 0
            precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def recall_at_k(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Recall@k.

    Recall@k = (Recommended items that are relevant) / (All relevant items)

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Average Recall@k across all users
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    recalls = []

    for user in users.to_series().to_list():
        user_recs = top_k_recs.filter(pl.col(col_user) == user).select(col_item).to_series().to_list()
        user_truth = ground_truth.filter(pl.col(col_user) == user).select(col_item).to_series().to_list()

        if len(user_truth) > 0:
            hits = len(set(user_recs) & set(user_truth))
            recall = hits / len(user_truth)
            recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def map_at_k(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Mean Average Precision@k.

    MAP@k = mean(AP@k for all users)
    AP@k = sum(Precision@i * rel(i)) / min(m, k)
    where rel(i) is 1 if item at rank i is relevant, 0 otherwise

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Mean Average Precision@k
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    aps = []

    for user in users.to_series().to_list():
        user_recs = top_k_recs.filter(pl.col(col_user) == user).sort(col_prediction, descending=True)
        user_recs_items = user_recs.select(col_item).to_series().to_list()
        user_truth = set(ground_truth.filter(pl.col(col_user) == user).select(col_item).to_series().to_list())

        if len(user_truth) > 0:
            hits = 0
            precision_sum = 0.0

            for i, item in enumerate(user_recs_items):
                if item in user_truth:
                    hits += 1
                    precision_sum += hits / (i + 1)

            ap = precision_sum / min(len(user_truth), k)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def ndcg_at_k(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k.

    NDCG@k = DCG@k / IDCG@k
    DCG@k = sum(rel(i) / log2(i + 1)) for i in 1..k
    IDCG@k = ideal DCG (all relevant items in top positions)

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Average NDCG@k across all users
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    ndcgs = []

    for user in users.to_series().to_list():
        user_recs = top_k_recs.filter(pl.col(col_user) == user).sort(col_prediction, descending=True)
        user_recs_items = user_recs.select(col_item).to_series().to_list()
        user_truth = set(ground_truth.filter(pl.col(col_user) == user).select(col_item).to_series().to_list())

        if len(user_truth) > 0:
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(user_recs_items):
                if item in user_truth:
                    dcg += 1.0 / np.log2(i + 2)  # i + 2 because log2(1) = 0

            # Calculate IDCG (ideal case: all relevant items in top positions)
            idcg = 0.0
            for i in range(min(len(user_truth), k)):
                idcg += 1.0 / np.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

    return np.mean(ndcgs) if ndcgs else 0.0


def hit_rate_at_k(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Hit Rate@k.

    Hit Rate@k = (# users with at least one relevant item in top-k) / (# users)

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Hit Rate@k
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    hits = 0
    total_users = 0

    for user in users.to_series().to_list():
        user_recs = set(top_k_recs.filter(pl.col(col_user) == user).select(col_item).to_series().to_list())
        user_truth = set(ground_truth.filter(pl.col(col_user) == user).select(col_item).to_series().to_list())

        if len(user_truth) > 0:
            total_users += 1
            if len(user_recs & user_truth) > 0:
                hits += 1

    return hits / total_users if total_users > 0 else 0.0


def coverage(
    recommendations: pl.DataFrame,
    all_items: List[Any],
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Catalog Coverage@k.

    Coverage@k = (# unique items recommended in top-k) / (# total items)

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        all_items: List of all items in the catalog
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Coverage@k
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by("userID" if "userID" in recommendations.columns else col_item)
        .head(k)
    )

    recommended_items = set(top_k_recs.select(col_item).unique().to_series().to_list())
    all_items_set = set(all_items)

    return len(recommended_items & all_items_set) / len(all_items_set) if all_items_set else 0.0


def diversity(
    recommendations: pl.DataFrame,
    item_features: Optional[Dict[Any, List[Any]]] = None,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k: int = 10,
) -> float:
    """
    Calculate Diversity@k (intra-list diversity).

    Diversity@k = average(1 - similarity between all pairs in top-k)
    
    Note: This is a simplified version that counts unique items per user.

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        item_features: Optional dict mapping item_id to feature list (for similarity)
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k: Number of top recommendations to consider

    Returns:
        float: Average diversity across users
    """
    # Get top-k recommendations per user
    top_k_recs = (
        recommendations.sort(col_prediction, descending=True)
        .group_by(col_user)
        .head(k)
    )

    # Get unique users
    users = top_k_recs.select(col_user).unique()

    diversities = []

    for user in users.to_series().to_list():
        user_recs = top_k_recs.filter(pl.col(col_user) == user).select(col_item).to_series().to_list()
        
        if len(user_recs) > 1:
            # Simple diversity: proportion of unique items
            unique_items = len(set(user_recs))
            diversity_score = unique_items / len(user_recs)
            diversities.append(diversity_score)

    return np.mean(diversities) if diversities else 0.0


def evaluate_recommendations(
    recommendations: pl.DataFrame,
    ground_truth: pl.DataFrame,
    all_items: Optional[List[Any]] = None,
    col_user: str = "userID",
    col_item: str = "itemID",
    col_prediction: str = "prediction",
    k_values: List[int] = [5, 10, 20],
    mlflow_compatible_names: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        recommendations: DataFrame with user, item, and prediction columns
        ground_truth: DataFrame with user and item columns for relevant items
        all_items: Optional list of all items in catalog (for coverage)
        col_user: User column name
        col_item: Item column name
        col_prediction: Prediction column name
        k_values: List of k values to evaluate
        mlflow_compatible_names: If True, use MLFlow-compatible metric names (e.g., "precision_at_5" instead of "precision@5")

    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    metrics = {}

    for k in k_values:
        if mlflow_compatible_names:
            metrics[f"precision_at_{k}"] = precision_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"recall_at_{k}"] = recall_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"map_at_{k}"] = map_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"hit_rate_at_{k}"] = hit_rate_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
        else:
            metrics[f"precision@{k}"] = precision_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"recall@{k}"] = recall_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"map@{k}"] = map_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"ndcg@{k}"] = ndcg_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(
                recommendations, ground_truth, col_user, col_item, col_prediction, k
            )

    if all_items is not None:
        for k in k_values:
            if mlflow_compatible_names:
                metrics[f"coverage_at_{k}"] = coverage(
                    recommendations, all_items, col_item, col_prediction, k
                )
            else:
                metrics[f"coverage@{k}"] = coverage(
                    recommendations, all_items, col_item, col_prediction, k
                )

    return metrics
