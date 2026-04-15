"""Neural Collaborative Filtering (NCF) model using PyTorch.

NCF combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP)
to learn user-item interactions. This implementation follows the architecture from:
"Neural Collaborative Filtering" (He et al., WWW 2017)

The model consists of:
- GMF component: Element-wise product of user and item embeddings
- MLP component: Concatenated embeddings passed through hidden layers
- Final prediction: Sigmoid over concatenated GMF + MLP outputs
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from bookdb.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_USER_COL,
)

logger = logging.getLogger(__name__)


class NCFNetwork(nn.Module):
    """Neural Collaborative Filtering network architecture.

    Combines GMF (Generalized Matrix Factorization) and MLP (Multi-Layer Perceptron)
    for learning user-item interactions.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int = 64,
        layers: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.factors = factors
        self.layers = layers or [64, 32, 16]
        self.dropout = dropout

        # GMF component embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, factors)
        self.gmf_item_embedding = nn.Embedding(n_items, factors)

        # MLP component embeddings (can have different dimension)
        mlp_factor = self.layers[0] // 2 if self.layers else factors
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_factor)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_factor)

        # MLP hidden layers
        mlp_input_dim = mlp_factor * 2
        self.mlp_layers = nn.Sequential()
        for i, layer_size in enumerate(self.layers):
            self.mlp_layers.add_module(f"fc_{i}", nn.Linear(mlp_input_dim, layer_size))
            self.mlp_layers.add_module(f"relu_{i}", nn.ReLU())
            self.mlp_layers.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
            mlp_input_dim = layer_size

        # Final prediction layer (concatenate GMF + MLP outputs)
        final_input_dim = factors + self.layers[-1] if self.layers else factors * 2
        self.output_layer = nn.Linear(final_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices

        Returns:
            Predicted interaction probabilities
        """
        # GMF component
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_vector = gmf_user * gmf_item  # Element-wise product

        # MLP component
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_vector = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_vector)

        # Concatenate GMF and MLP
        concat_vector = torch.cat([gmf_vector, mlp_output], dim=-1)

        # Final prediction
        prediction = self.output_layer(concat_vector)
        return self.sigmoid(prediction).squeeze()


class InteractionDataset(Dataset):
    """Dataset for user-item interactions with negative sampling."""

    def __init__(
        self,
        interactions: list[tuple[int, int]],
        n_users: int,
        n_items: int,
        n_negatives: int = 4,
    ):
        self.interactions = interactions
        self.n_users = n_users
        self.n_items = n_items
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]

        # Positive sample
        users = [user] * (1 + self.n_negatives)
        items = [pos_item]
        labels = [1.0]

        # Negative samples
        for _ in range(self.n_negatives):
            neg_item = np.random.randint(0, self.n_items)
            # Simple negative sampling (may sample positives, but probability is low)
            items.append(neg_item)
            labels.append(0.0)

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )


class NCF:
    """Neural Collaborative Filtering model.

    This class wraps a PyTorch NCF implementation, providing:
    - Automatic string-to-integer ID mapping
    - Negative sampling for implicit feedback
    - Batched recommendation generation
    - Polars DataFrame integration

    Args:
        factors: Number of latent factors for GMF component. Default: 64
        layers: List of MLP layer sizes. Default: [64, 32, 16]
        epochs: Number of training epochs. Default: 20
        batch_size: Training batch size. Default: 256
        learning_rate: Learning rate. Default: 0.001
        dropout: Dropout probability. Default: 0.2
        n_negatives: Number of negative samples per positive. Default: 4
        device: Device to use ('cpu', 'cuda', or 'mps'). Default: auto-detect
        random_state: Random seed for reproducibility.

    Example:
        >>> model = NCF(factors=64, layers=[64, 32, 16], epochs=20)
        >>> model.fit(train_df, col_user="user_id", col_item="book_id")
        >>> recommendations = model.recommend_k_items(test_df, top_k=100)
    """

    def __init__(
        self,
        factors: int = 64,
        layers: Optional[list[int]] = None,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        n_negatives: int = 4,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.factors = factors
        self.layers = layers or [64, 32, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.n_negatives = n_negatives
        self.random_state = random_state

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"NCF using device: {self.device}")

        # ID mappings (set after fit)
        self._user_id_map: dict[str, int] = {}
        self._item_id_map: dict[str, int] = {}
        self._reverse_user_map: dict[int, str] = {}
        self._reverse_item_map: dict[int, str] = {}
        self._col_user: str = DEFAULT_USER_COL
        self._col_item: str = DEFAULT_ITEM_COL
        self._user_dtype: Optional[pl.DataType] = None
        self._item_dtype: Optional[pl.DataType] = None

        # Model (initialized during fit)
        self._model: Optional[NCFNetwork] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def _create_mappings(
        self,
        df: pl.DataFrame,
        col_user: str,
        col_item: str,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Create string-to-integer ID mappings."""
        unique_users = df.select(col_user).unique().to_series()
        unique_items = df.select(col_item).unique().to_series()

        user_id_map = {str(uid): idx for idx, uid in enumerate(unique_users)}
        item_id_map = {str(iid): idx for idx, iid in enumerate(unique_items)}

        logger.info(
            f"Created mappings: {len(user_id_map)} users, {len(item_id_map)} items"
        )

        return user_id_map, item_id_map

    def fit(
        self,
        df: pl.DataFrame,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_rating: Optional[str] = None,
        verbose: bool = True,
    ) -> "NCF":
        """Fit the NCF model on the training data.

        Args:
            df: Training data as a Polars DataFrame.
            col_user: Name of the user column.
            col_item: Name of the item column.
            col_rating: Not used (NCF uses implicit feedback with negative sampling).
            verbose: Whether to print progress.

        Returns:
            self (fitted model)
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        logger.info("Fitting NCF model...")
        logger.info(
            f"  factors={self.factors}, layers={self.layers}, epochs={self.epochs}"
        )
        logger.info(
            f"  batch_size={self.batch_size}, lr={self.learning_rate}, dropout={self.dropout}"
        )

        # Store column names and dtypes
        self._col_user = col_user
        self._col_item = col_item
        self._user_dtype = df.schema[col_user]
        self._item_dtype = df.schema[col_item]

        # Create ID mappings
        self._user_id_map, self._item_id_map = self._create_mappings(
            df, col_user, col_item
        )
        self._reverse_user_map = {v: k for k, v in self._user_id_map.items()}
        self._reverse_item_map = {v: k for k, v in self._item_id_map.items()}

        self.n_users = len(self._user_id_map)
        self.n_items = len(self._item_id_map)

        # Create interaction list
        interactions = []
        for row in df.iter_rows(named=True):
            user_idx = self._user_id_map[str(row[col_user])]
            item_idx = self._item_id_map[str(row[col_item])]
            interactions.append((user_idx, item_idx))

        logger.info(f"Training on {len(interactions)} positive interactions")

        # Create dataset and dataloader
        dataset = InteractionDataset(
            interactions, self.n_users, self.n_items, self.n_negatives
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Initialize model
        self._model = NCFNetwork(
            self.n_users,
            self.n_items,
            self.factors,
            self.layers,
            self.dropout,
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_labels in dataloader:
                # Flatten batch (each sample has 1 positive + n_negatives)
                batch_users = batch_users.view(-1).to(self.device)
                batch_items = batch_items.view(-1).to(self.device)
                batch_labels = batch_labels.view(-1).to(self.device)

                # Forward pass
                predictions = self._model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        logger.info(f"NCF model fitted: {self.n_users} users, {self.n_items} items")

        return self

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

        Args:
            data: DataFrame containing users to generate recommendations for.
            top_k: Number of recommendations per user. Default: 10.
            remove_seen: Whether to exclude items the user has already interacted with.
            col_user: Name of the user column. Uses default from fit() if None.
            col_item: Name of the item column. Uses default from fit() if None.
            col_prediction: Name of the prediction column.
            batch_size: Number of items to score at once (memory control).

        Returns:
            DataFrame with columns [col_user, col_item, col_prediction]
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before generating recommendations")

        col_user = col_user or self._col_user
        col_item = col_item or self._col_item
        top_k = min(top_k or 10, self.n_items)

        users = data.select(col_user).unique().to_series().to_list()
        logger.info(f"Generating top-{top_k} recommendations for {len(users)} users")

        self._model.eval()
        all_recommendations = []

        with torch.no_grad():
            for user_id in users:
                user_id_str = str(user_id)
                if user_id_str not in self._user_id_map:
                    continue

                user_idx = self._user_id_map[user_id_str]

                # Score all items
                item_indices = torch.arange(self.n_items, device=self.device)
                user_tensor = torch.full(
                    (self.n_items,), user_idx, dtype=torch.long, device=self.device
                )

                # Score in batches to avoid OOM
                scores_list = []
                for i in range(0, self.n_items, batch_size):
                    batch_end = min(i + batch_size, self.n_items)
                    batch_scores = self._model(
                        user_tensor[i:batch_end], item_indices[i:batch_end]
                    )
                    scores_list.append(batch_scores.cpu().numpy())

                scores = np.concatenate(scores_list)

                # Get top-k
                top_indices = np.argsort(-scores)[:top_k]
                top_scores = scores[top_indices]

                # Convert to original IDs
                for item_idx, score in zip(top_indices, top_scores):
                    all_recommendations.append(
                        {
                            col_user: user_id,
                            col_item: self._reverse_item_map[int(item_idx)],
                            col_prediction: float(score),
                        }
                    )

        if not all_recommendations:
            logger.warning("No recommendations generated")
            return pl.DataFrame(
                {
                    col_user: [],
                    col_item: [],
                    col_prediction: [],
                }
            )

        recommendations = pl.DataFrame(all_recommendations)

        # Cast columns to original dtypes
        if self._user_dtype is not None:
            recommendations = recommendations.with_columns(
                pl.col(col_user).cast(self._user_dtype)
            )
        if self._item_dtype is not None:
            recommendations = recommendations.with_columns(
                pl.col(col_item).cast(self._item_dtype)
            )

        logger.info(f"Generated {len(recommendations)} recommendations")

        return recommendations

    def save(self, path: Union[str, Path]):
        """Save the model to disk.

        Args:
            path: Directory path to save the model.
        """
        import json

        if self._model is None:
            raise RuntimeError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self._model.state_dict(), save_path / "model.pt")

        # Save config
        config = {
            "factors": self.factors,
            "layers": self.layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "n_negatives": self.n_negatives,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "col_user": self._col_user,
            "col_item": self._col_item,
            "user_dtype": str(self._user_dtype) if self._user_dtype else None,
            "item_dtype": str(self._item_dtype) if self._item_dtype else None,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save mappings
        mappings = {
            "user_id_map": self._user_id_map,
            "item_id_map": self._item_id_map,
        }
        with open(save_path / "mappings.json", "w") as f:
            json.dump(mappings, f)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NCF":
        """Load a model from disk.

        Args:
            path: Directory path containing the saved model.

        Returns:
            Loaded NCF model
        """
        import json

        load_path = Path(path)

        # Load config
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            factors=config["factors"],
            layers=config["layers"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            dropout=config["dropout"],
            n_negatives=config["n_negatives"],
        )

        instance.n_users = config["n_users"]
        instance.n_items = config["n_items"]
        instance._col_user = config["col_user"]
        instance._col_item = config["col_item"]

        # Restore dtypes
        dtype_map = {
            "Int8": pl.Int8,
            "Int16": pl.Int16,
            "Int32": pl.Int32,
            "Int64": pl.Int64,
            "UInt8": pl.UInt8,
            "UInt16": pl.UInt16,
            "UInt32": pl.UInt32,
            "UInt64": pl.UInt64,
            "Float32": pl.Float32,
            "Float64": pl.Float64,
            "String": pl.String,
        }
        instance._user_dtype = (
            dtype_map.get(config.get("user_dtype"))
            if config.get("user_dtype")
            else None
        )
        instance._item_dtype = (
            dtype_map.get(config.get("item_dtype"))
            if config.get("item_dtype")
            else None
        )

        # Load mappings
        with open(load_path / "mappings.json", "r") as f:
            mappings = json.load(f)
        instance._user_id_map = mappings["user_id_map"]
        instance._item_id_map = mappings["item_id_map"]
        instance._reverse_user_map = {v: k for k, v in instance._user_id_map.items()}
        instance._reverse_item_map = {v: k for k, v in instance._item_id_map.items()}

        # Initialize and load model weights
        instance._model = NCFNetwork(
            instance.n_users,
            instance.n_items,
            instance.factors,
            instance.layers,
            instance.dropout,
        ).to(instance.device)

        instance._model.load_state_dict(
            torch.load(load_path / "model.pt", map_location=instance.device)
        )
        instance._model.eval()

        logger.info(f"Model loaded from {load_path}")

        return instance
