# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import polars as pl
import logging
from scipy import sparse

from bookdb.utils.python_utils import (
    cosine_similarity,
    inclusion_index,
    jaccard,
    lexicographers_mutual_information,
    lift,
    mutual_information,
    exponential_decay,
    get_top_k_scored_items,
    rescale,
)
from bookdb.utils import constants

# Try to import torch utilities for GPU/MPS acceleration
try:
    from bookdb.utils.torch_utils import (
        is_torch_available,
        is_mps_available,
        is_cuda_available,
        get_device,
        jaccard_torch,
        lift_torch,
        cosine_similarity_torch,
        mutual_information_torch,
        lexicographers_mutual_information_torch,
        inclusion_index_torch,
        compute_scores_torch,
        get_top_k_scored_items_torch,
        SIMILARITY_FUNCTIONS_TORCH,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    is_mps_available = lambda: False
    is_cuda_available = lambda: False


SIM_COOCCUR = "cooccurrence"
SIM_COSINE = "cosine"
SIM_INCLUSION_INDEX = "inclusion index"
SIM_JACCARD = "jaccard"
SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION = "lexicographers mutual information"
SIM_LIFT = "lift"
SIM_MUTUAL_INFORMATION = "mutual information"

logger = logging.getLogger()


class SARSingleNode:
    """Simple Algorithm for Recommendations (SAR) implementation

    SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history
    and items description. The core idea behind SAR is to recommend items like those that a user already has
    demonstrated an affinity to. It does this by 1) estimating the affinity of users for items, 2) estimating
    similarity across items, and then 3) combining the estimates to generate a set of recommendations for a given user.
    """

    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        similarity_type=SIM_JACCARD,
        time_decay_coefficient=30,
        time_now=None,
        timedecay_formula=False,
        threshold=1,
        normalize=False,
        use_torch=True,
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            similarity_type (str): ['cooccurrence', 'cosine', 'inclusion index', 'jaccard',
              'lexicographers mutual information', 'lift', 'mutual information'] option for
              computing item-item similarity
            time_decay_coefficient (float): number of days till ratings are decayed by 1/2
            time_now (int | None): current time for time decay calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this threshold will be removed
            normalize (bool): option for normalizing predictions to scale of original ratings
            use_torch (bool): whether to use PyTorch acceleration (MPS/CUDA) if available
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        
        # Configure torch acceleration
        self.use_torch = use_torch and TORCH_AVAILABLE
        if use_torch and not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch acceleration requested but not available. "
                "Falling back to NumPy. Install PyTorch for MPS/CUDA acceleration."
            )
        elif self.use_torch:
            if is_mps_available():
                logger.info("Using MPS (Apple Metal) acceleration")
            elif is_cuda_available():
                logger.info("Using CUDA acceleration")
            else:
                logger.info("Using PyTorch CPU acceleration")

        available_similarity_types = [
            SIM_COOCCUR,
            SIM_COSINE,
            SIM_INCLUSION_INDEX,
            SIM_JACCARD,
            SIM_LIFT,
            SIM_MUTUAL_INFORMATION,
            SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION,
        ]
        if similarity_type not in available_similarity_types:
            raise ValueError(
                'Similarity type must be one of ["'
                + '" | "'.join(available_similarity_types)
                + '"]'
            )
        self.similarity_type = similarity_type
        self.time_decay_half_life = (
            time_decay_coefficient * 24 * 60 * 60
        )  # convert to seconds
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold
        self.user_affinity = None
        self.item_similarity = None
        self.item_frequencies = None
        self.user_frequencies = None

        # threshold - items below this number get set to zero in co-occurrence counts
        if self.threshold <= 0:
            raise ValueError("Threshold cannot be < 1")

        # set flag to capture unity-rating user-affinity matrix for scaling scores
        self.normalize = normalize
        self.col_unity_rating = "_unity_rating"
        self.unity_user_affinity = None

        # column for mapping user / item ids to internal indices
        self.col_item_id = "_indexed_items"
        self.col_user_id = "_indexed_users"

        # obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # The min and max of the rating scale, obtained from the training data.
        self.rating_min = None
        self.rating_max = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above maps - map array index to actual string ID
        self.index2item = None
        self.index2user = None

    def compute_affinity_matrix(self, df, rating_col):
        """Affinity matrix.

        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we're treating
        the ratings as the event weights.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.

        Args:
            df (polars.DataFrame): Indexed df of users and items
            rating_col (str): Name of column to use for ratings

        Returns:
            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
        """

        return sparse.coo_matrix(
            (
                df.select(rating_col).to_series().to_numpy(),
                (
                    df.select(self.col_user_id).to_series().to_numpy(),
                    df.select(self.col_item_id).to_series().to_numpy(),
                ),
            ),
            shape=(self.n_users, self.n_items),
        ).tocsr()

    def compute_time_decay(self, df, decay_column):
        """Compute time decay on provided column.

        Args:
            df (polars.DataFrame): DataFrame of users and items
            decay_column (str): column to decay

        Returns:
            polars.DataFrame: with column decayed
        """

        # if time_now is None use the latest time
        if self.time_now is None:
            self.time_now = df.select(pl.col(self.col_timestamp).max()).item()

        # apply time decay to each rating
        timestamps = df.select(self.col_timestamp).to_series().to_numpy()
        decayed_ratings = df.select(decay_column).to_series().to_numpy() * exponential_decay(
            value=timestamps,
            max_val=self.time_now,
            half_life=self.time_decay_half_life,
        )

        df = df.with_columns(pl.Series(decayed_ratings).alias(decay_column))

        # group time decayed ratings by user-item and take the sum as the user-item affinity
        return df.group_by([self.col_user, self.col_item]).agg(pl.col(decay_column).sum())

    def compute_cooccurrence_matrix(self, df):
        """Co-occurrence matrix.

        The co-occurrence matrix is defined as :math:`C = U^T * U`

        where U is the user_affinity matrix with 1's as values (instead of ratings).

        Args:
            df (polars.DataFrame): DataFrame of users and items

        Returns:
            numpy.ndarray: Co-occurrence matrix
        """
        user_item_hits = sparse.coo_matrix(
            (
                np.repeat(1, df.height),
                (
                    df.select(self.col_user_id).to_series().to_numpy(),
                    df.select(self.col_item_id).to_series().to_numpy(),
                ),
            ),
            shape=(self.n_users, self.n_items),
        ).tocsr()

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(
            item_cooccurrence >= self.threshold
        )

        # Get the physical dtype for the cooccurrence matrix
        dtype = df.schema[self.col_rating]
        # Convert polars dtype to numpy dtype
        # Float32 and Float64 are already physical types in newer polars
        if str(dtype) == "Float32":
            target_dtype = np.float32
        elif str(dtype) == "Float64":
            target_dtype = np.float64
        elif str(dtype) == "Int32":
            target_dtype = np.int32
        elif str(dtype) == "Int64":
            target_dtype = np.int64
        else:
            # Fallback to float64 for unknown types
            target_dtype = np.float64
        return item_cooccurrence.astype(target_dtype)

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage.

        Args:
            df (polars.DataFrame): dataframe with user and item ids
        """

        # generate a map of continuous index values to items
        self.index2item = dict(
            enumerate(df.select(pl.col(self.col_item).unique()).to_series().to_list())
        )
        self.index2user = dict(
            enumerate(df.select(pl.col(self.col_user).unique()).to_series().to_list())
        )

        # invert the mappings from above
        self.item2index = {v: k for k, v in self.index2item.items()}
        self.user2index = {v: k for k, v in self.index2user.items()}

        # set values for the total count of users and items
        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

    def fit(self, df):
        """Main fit method for SAR.

        Note:
            Please make sure that `df` has no duplicates.

        Args:
            df (polars.DataFrame): User item rating dataframe (without duplicates).
        """
        select_columns = [self.col_user, self.col_item, self.col_rating]
        if self.time_decay_flag:
            select_columns += [self.col_timestamp]

        if df.select(select_columns).is_duplicated().any():
            raise ValueError("There should not be duplicates in the dataframe")

        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        logger.info("Collecting user affinity matrix")
        rating_dtype = df.schema[self.col_rating]
        if not rating_dtype.is_numeric():
            raise TypeError("Rating column data type must be numeric")

        # copy the DataFrame to avoid modification of the input
        temp_df = df.select(select_columns).clone()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities")
            temp_df = self.compute_time_decay(df=temp_df, decay_column=self.col_rating)

        logger.info("Creating index columns")
        # add mapping of user and item ids to indices
        temp_df = temp_df.with_columns(
            pl.col(self.col_item)
            .replace(self.item2index, default=None)
            .alias(self.col_item_id),
            pl.col(self.col_user)
            .replace(self.user2index, default=None)
            .alias(self.col_user_id),
        )

        if self.normalize:
            self.rating_min = temp_df.select(pl.col(self.col_rating).min()).item()
            self.rating_max = temp_df.select(pl.col(self.col_rating).max()).item()
            logger.info("Calculating normalization factors")
            temp_df = temp_df.with_columns(pl.lit(1.0).alias(self.col_unity_rating))
            if self.time_decay_flag:
                temp_df = self.compute_time_decay(
                    df=temp_df, decay_column=self.col_unity_rating
                )
            self.unity_user_affinity = self.compute_affinity_matrix(
                df=temp_df, rating_col=self.col_unity_rating
            )

        # affinity matrix
        logger.info("Building user affinity sparse matrix")
        self.user_affinity = self.compute_affinity_matrix(
            df=temp_df, rating_col=self.col_rating
        )

        # calculate item co-occurrence
        logger.info("Calculating item co-occurrence")
        item_cooccurrence = self.compute_cooccurrence_matrix(df=temp_df)

        # free up some space
        del temp_df

        # creates an array with the frequency of every unique item
        self.item_frequencies = item_cooccurrence.diagonal()

        logger.info("Calculating item similarity")
        if self.similarity_type == SIM_COOCCUR:
            logger.info("Using co-occurrence based similarity")
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == SIM_COSINE:
            logger.info("Using cosine similarity" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = cosine_similarity_torch(item_cooccurrence)
            else:
                self.item_similarity = cosine_similarity(item_cooccurrence)
        elif self.similarity_type == SIM_INCLUSION_INDEX:
            logger.info("Using inclusion index" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = inclusion_index_torch(item_cooccurrence)
            else:
                self.item_similarity = inclusion_index(item_cooccurrence)
        elif self.similarity_type == SIM_JACCARD:
            logger.info("Using jaccard based similarity" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = jaccard_torch(item_cooccurrence)
            else:
                self.item_similarity = jaccard(item_cooccurrence)
        elif self.similarity_type == SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION:
            logger.info("Using lexicographers mutual information similarity" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = lexicographers_mutual_information_torch(item_cooccurrence)
            else:
                self.item_similarity = lexicographers_mutual_information(item_cooccurrence)
        elif self.similarity_type == SIM_LIFT:
            logger.info("Using lift based similarity" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = lift_torch(item_cooccurrence)
            else:
                self.item_similarity = lift(item_cooccurrence)
        elif self.similarity_type == SIM_MUTUAL_INFORMATION:
            logger.info("Using mutual information similarity" + (" (torch accelerated)" if self.use_torch else ""))
            if self.use_torch:
                self.item_similarity = mutual_information_torch(item_cooccurrence)
            else:
                self.item_similarity = mutual_information(item_cooccurrence)
        else:
            raise ValueError("Unknown similarity type: {}".format(self.similarity_type))

        # free up some space
        del item_cooccurrence

        logger.info("Done training")

    def score(self, test, remove_seen=False):
        """Score all items for test users.

        Args:
            test (polars.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """

        # get user / item indices from test set
        unique_users = (
            test.select(pl.col(self.col_user).unique()).to_series().to_list()
        )
        user_ids = [
            self.user2index.get(user) for user in unique_users
        ]
        if any(uid is None for uid in user_ids):
            raise ValueError("SAR cannot score users that are not in the training set")

        # calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores")
        test_scores = self.user_affinity[user_ids, :].dot(self.item_similarity)

        # ensure we're working with a dense ndarray
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()

        if self.normalize:
            counts = self.unity_user_affinity[user_ids, :].dot(self.item_similarity)
            user_min_scores = (
                np.tile(counts.min(axis=1)[:, np.newaxis], test_scores.shape[1])
                * self.rating_min
            )
            user_max_scores = (
                np.tile(counts.max(axis=1)[:, np.newaxis], test_scores.shape[1])
                * self.rating_max
            )
            test_scores = rescale(
                test_scores,
                self.rating_min,
                self.rating_max,
                user_min_scores,
                user_max_scores,
            )

        # remove items in the train set so recommended items are always novel
        if remove_seen:
            logger.info("Removing seen items")
            test_scores += self.user_affinity[user_ids, :] * -np.inf

        return test_scores

    def get_popularity_based_topk(self, top_k=10, sort_top_k=True, items=True):
        """Get top K most frequently occurring items across all users.

        Args:
            top_k (int): number of top items to recommend.
            sort_top_k (bool): flag to sort top k results.
            items (bool): if false, return most frequent users instead

        Returns:
            polars.DataFrame: top k most popular items.
        """
        if items:
            frequencies = self.item_frequencies
            col = self.col_item
            idx = self.index2item
        else:
            if self.user_frequencies is None:
                self.user_frequencies = self.user_affinity.count_nonzero(axis=1).astype(
                    "int64"
                )
            frequencies = self.user_frequencies
            col = self.col_user
            idx = self.index2user

        test_scores = np.array([frequencies])

        logger.info("Getting top K")
        top_components, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        return pl.DataFrame(
            {
                col: [idx[item] for item in top_components.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

    def get_item_based_topk(self, items, top_k=10, sort_top_k=True):
        """Get top K similar items to provided seed items based on similarity metric defined.
        This method will take a set of items and use them to recommend the most similar items to that set
        based on the similarity matrix fit during training.
        This allows recommendations for cold-users (unseen during training), note - the model is not updated.

        The following options are possible based on information provided in the items input:
        1. Single user or seed of items: only item column (ratings are assumed to be 1)
        2. Single user or seed of items w/ ratings: item column and rating column
        3. Separate users or seeds of items: item and user column (user ids are only used to separate item sets)
        4. Separate users or seeds of items with ratings: item, user and rating columns provided

        Args:
            items (polars.DataFrame): DataFrame with item, user (optional), and rating (optional) columns
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            polars.DataFrame: sorted top k recommendation items
        """

        # convert item ids to indices
        item_ids = np.asarray(
            [
                self.item2index.get(item)
                for item in items.select(self.col_item).to_series().to_list()
            ]
        )

        # if no ratings were provided assume they are all 1
        if self.col_rating in items.columns:
            ratings = items.select(self.col_rating).to_series().to_numpy()
        else:
            ratings = np.ones(len(item_ids))

        # create local map of user ids
        if self.col_user in items.columns:
            test_users = items.select(self.col_user).to_series()
            unique_users = test_users.unique().to_list()
            user2index = {user: idx for idx, user in enumerate(unique_users)}
            user_ids = test_users.replace(user2index).to_numpy()
            test_users_unique = np.array(unique_users)
        else:
            # if no user column exists assume all entries are for a single user
            test_users = pl.Series([0] * len(item_ids))
            user_ids = np.zeros(len(item_ids), dtype=int)
            test_users_unique = np.array([0])
        n_users = len(test_users_unique)

        # generate pseudo user affinity using seed items
        pseudo_affinity = sparse.coo_matrix(
            (ratings, (user_ids, item_ids)), shape=(n_users, self.n_items)
        ).tocsr()

        # calculate raw scores with a matrix multiplication
        test_scores = pseudo_affinity.dot(self.item_similarity)

        # remove items in the seed set so recommended items are novel
        test_scores[user_ids, item_ids] = -np.inf

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pl.DataFrame(
            {
                self.col_user: np.repeat(
                    test_users_unique, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.filter(pl.col(self.col_prediction).is_not_nan() & (pl.col(self.col_prediction) != -np.inf))

    def get_topk_most_similar_users(self, user, top_k, sort_top_k=True):
        """Based on user affinity towards items, calculate the most similar users to the given user.

        Args:
            user (int): user to retrieve most similar users for
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            polars.DataFrame: top k most similar users and their scores
        """
        user_idx = self.user2index[user]
        similarities = self.user_affinity[user_idx].dot(self.user_affinity.T).toarray()
        similarities[0, user_idx] = -np.inf

        top_items, top_scores = get_top_k_scored_items(
            scores=similarities, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pl.DataFrame(
            {
                self.col_user: [self.index2user[user] for user in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.filter(pl.col(self.col_prediction).is_not_nan() & (pl.col(self.col_prediction) != -np.inf))

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (polars.DataFrame): users to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            polars.DataFrame: top k recommendation items for each user
        """

        test_scores = self.score(test, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pl.DataFrame(
            {
                self.col_user: np.repeat(
                    test.select(self.col_user).unique().to_series().to_numpy(),
                    top_items.shape[1],
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.filter(pl.col(self.col_prediction).is_not_nan() & (pl.col(self.col_prediction) != -np.inf))

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set

        Args:
            test (polars.DataFrame): DataFrame that contains users and items to test

        Returns:
            polars.DataFrame: DataFrame contains the prediction results
        """

        test_scores = self.score(test)
        user_ids = np.asarray(
            [
                self.user2index.get(user)
                for user in test.select(self.col_user).to_series().to_list()
            ]
        )

        # create mapping of new items to zeros
        item_ids = np.asarray(
            [
                self.item2index.get(item)
                for item in test.select(self.col_item).to_series().to_list()
            ]
        )
        nans = np.array([idx is None for idx in item_ids])
        if any(nans):
            logger.warning(
                "Items found in test not seen during training, new items will have score of 0"
            )
            test_scores = np.append(test_scores, np.zeros((self.n_users, 1)), axis=1)
            item_ids[nans] = self.n_items
            item_ids = item_ids.astype("int64")

        df = pl.DataFrame(
            {
                self.col_user: test.select(self.col_user).to_series().to_numpy(),
                self.col_item: test.select(self.col_item).to_series().to_numpy(),
                self.col_prediction: test_scores[user_ids, item_ids],
            }
        )
        return df
