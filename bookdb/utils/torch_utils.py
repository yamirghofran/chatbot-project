# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
PyTorch-accelerated utility functions for SAR model.

This module provides MPS/GPU-accelerated versions of similarity functions
for better performance on Apple Silicon and CUDA devices.
"""

import logging
import numpy as np
from typing import Optional, Union, Literal, Any

logger = logging.getLogger(__name__)

# Check for PyTorch and MPS availability
_torch_available = False
_mps_available = False
_cuda_available = False

try:
    import torch
    _torch_available = True
    
    if hasattr(torch.backends, 'mps'):
        _mps_available = torch.backends.mps.is_available()
    
    _cuda_available = torch.cuda.is_available()
    
    if _mps_available:
        logger.info("PyTorch MPS acceleration available")
    if _cuda_available:
        logger.info("PyTorch CUDA acceleration available")
        
except ImportError:
    logger.warning("PyTorch not available, falling back to NumPy")


def _to_torch_tensor(
    array: Any,
    device: "torch.device",
) -> "torch.Tensor":
    """
    Convert numpy array or scipy sparse matrix to PyTorch tensor.
    
    Args:
        array: Input array (numpy, scipy sparse, or torch tensor)
        device: Target device
        
    Returns:
        torch.Tensor on the specified device
    """
    if hasattr(array, 'toarray'):
        # Scipy sparse matrix
        dense = array.toarray().astype(np.float32)
        return torch.from_numpy(dense).to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array.astype(np.float32)).to(device)
    else:
        # Already a torch tensor
        return array.float().to(device)


def get_device(
    prefer_mps: bool = True,
    prefer_cuda: bool = True,
) -> "torch.device":
    """
    Get the best available PyTorch device.
    
    Args:
        prefer_mps: Whether to prefer MPS over CPU
        prefer_cuda: Whether to prefer CUDA over MPS
        
    Returns:
        torch.device: The best available device
    """
    if not _torch_available:
        raise RuntimeError("PyTorch is not available")
    
    if prefer_cuda and _cuda_available:
        return torch.device("cuda")
    elif prefer_mps and _mps_available:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _get_row_and_column_matrix_torch(
    diag: "torch.Tensor",
    device: "torch.device",
) -> tuple:
    """
    Helper to create row and column matrices from diagonal values.
    
    Args:
        diag: 1D tensor of diagonal values
        device: PyTorch device to use
        
    Returns:
        Tuple of (row_matrix, column_matrix)
    """
    row_matrix = diag.unsqueeze(0).expand(diag.shape[0], -1)
    column_matrix = diag.unsqueeze(1).expand(-1, diag.shape[0])
    return row_matrix, column_matrix


# Maximum matrix size for MPS/GPU processing (elements)
# 50M elements ~ 200MB for float32, leaves room for intermediate tensors
MAX_TENSOR_SIZE = 50_000_000


def _should_use_torch(matrix_shape: tuple) -> bool:
    """Check if matrix is small enough for GPU processing."""
    n_elements = matrix_shape[0] * matrix_shape[1]
    return n_elements <= MAX_TENSOR_SIZE


def jaccard_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Jaccard similarity using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array, scipy sparse, or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Jaccard similarity matrix
    """
    if not _torch_available:
        # Fallback to numpy version
        from bookdb.utils.python_utils import jaccard
        return jaccard(cooccurrence)
    
    # Check if matrix is small enough for GPU
    if not _should_use_torch(cooccurrence.shape):
        logger.info(
            f"Matrix too large for MPS ({cooccurrence.shape[0]}x{cooccurrence.shape[1]}), "
            "falling back to NumPy"
        )
        from bookdb.utils.python_utils import jaccard
        return jaccard(cooccurrence)
    
    if device is None:
        device = get_device()
    
    # Convert to torch tensor if needed
    if hasattr(cooccurrence, 'toarray'):
        # Scipy sparse matrix - convert to dense first
        dense_array = cooccurrence.toarray().astype(np.float32)
        tensor = torch.from_numpy(dense_array).to(device)
    elif isinstance(cooccurrence, np.ndarray):
        tensor = torch.from_numpy(cooccurrence.astype(np.float32)).to(device)
    else:
        tensor = cooccurrence.float().to(device)
    
    # Get diagonal
    diag = tensor.diagonal()
    diag_rows, diag_cols = _get_row_and_column_matrix_torch(diag, device)
    
    # Compute Jaccard similarity
    denominator = diag_rows + diag_cols - tensor
    
    # Use where to handle division by zero
    result = torch.where(
        denominator != 0,
        tensor / denominator,
        torch.zeros_like(tensor)
    )
    
    return result.cpu().numpy()


def lift_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Lift similarity using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array, scipy sparse, or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Lift similarity matrix
    """
    if not _torch_available:
        from bookdb.utils.python_utils import lift
        return lift(cooccurrence)
    
    if not _should_use_torch(cooccurrence.shape):
        logger.info(f"Matrix too large for MPS, falling back to NumPy")
        from bookdb.utils.python_utils import lift
        return lift(cooccurrence)
    
    if device is None:
        device = get_device()
    
    if hasattr(cooccurrence, 'toarray'):
        dense_array = cooccurrence.toarray().astype(np.float32)
        tensor = torch.from_numpy(dense_array).to(device)
    elif isinstance(cooccurrence, np.ndarray):
        tensor = torch.from_numpy(cooccurrence.astype(np.float32)).to(device)
    else:
        tensor = cooccurrence.float().to(device)
    
    diag = tensor.diagonal()
    diag_rows, diag_cols = _get_row_and_column_matrix_torch(diag, device)
    
    denominator = diag_rows * diag_cols
    
    result = torch.where(
        denominator != 0,
        tensor / denominator,
        torch.zeros_like(tensor)
    )
    
    return result.cpu().numpy()


def cosine_similarity_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Cosine similarity using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Cosine similarity matrix
    """
    if not _torch_available:
        from bookdb.utils.python_utils import cosine_similarity
        return cosine_similarity(cooccurrence)
    
    if not _should_use_torch(cooccurrence.shape):
        logger.info(f"Matrix too large for MPS, falling back to NumPy")
        from bookdb.utils.python_utils import cosine_similarity
        return cosine_similarity(cooccurrence)
    
    if device is None:
        device = get_device()
    
    if isinstance(cooccurrence, np.ndarray):
        if hasattr(cooccurrence, 'toarray'):
            cooccurrence = cooccurrence.toarray()
        tensor = torch.from_numpy(cooccurrence.astype(np.float32)).to(device)
    else:
        tensor = cooccurrence.float().to(device)
    
    diag = tensor.diagonal()
    diag_rows, diag_cols = _get_row_and_column_matrix_torch(diag, device)
    
    denominator = torch.sqrt(diag_rows * diag_cols)
    
    result = torch.where(
        denominator != 0,
        tensor / denominator,
        torch.zeros_like(tensor)
    )
    
    return result.cpu().numpy()


def mutual_information_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Mutual Information using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Mutual information matrix
    """
    if not _torch_available:
        from bookdb.utils.python_utils import mutual_information
        return mutual_information(cooccurrence)
    
    if not _should_use_torch(cooccurrence.shape):
        logger.info(f"Matrix too large for MPS, falling back to NumPy")
        from bookdb.utils.python_utils import mutual_information
        return mutual_information(cooccurrence)
    
    if device is None:
        device = get_device()
    
    # Calculate lift first
    lift_matrix = lift_torch(cooccurrence, device)
    lift_tensor = torch.from_numpy(lift_matrix.astype(np.float32)).to(device)
    
    # Calculate mutual information
    n = cooccurrence.shape[0] if isinstance(cooccurrence, np.ndarray) else cooccurrence.shape[0]
    result = torch.log2(torch.tensor(float(n), device=device) * lift_tensor)
    
    # Handle invalid values
    result = torch.where(
        torch.isfinite(result),
        result,
        torch.zeros_like(result)
    )
    
    return result.cpu().numpy()


def lexicographers_mutual_information_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Lexicographers Mutual Information using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Lexicographers mutual information matrix
    """
    if not _torch_available:
        from bookdb.utils.python_utils import lexicographers_mutual_information
        return lexicographers_mutual_information(cooccurrence)
    
    if not _should_use_torch(cooccurrence.shape):
        logger.info(f"Matrix too large for MPS, falling back to NumPy")
        from bookdb.utils.python_utils import lexicographers_mutual_information
        return lexicographers_mutual_information(cooccurrence)
    
    if device is None:
        device = get_device()
    
    if isinstance(cooccurrence, np.ndarray):
        if hasattr(cooccurrence, 'toarray'):
            cooccurrence = cooccurrence.toarray()
        cooc_tensor = torch.from_numpy(cooccurrence.astype(np.float32)).to(device)
    else:
        cooc_tensor = cooccurrence.float().to(device)
    
    # Calculate mutual information
    mi = mutual_information_torch(cooccurrence, device)
    mi_tensor = torch.from_numpy(mi.astype(np.float32)).to(device)
    
    result = cooc_tensor * mi_tensor
    
    return result.cpu().numpy()


def inclusion_index_torch(
    cooccurrence: Union[np.ndarray, "torch.Tensor"],
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Calculate Inclusion Index using PyTorch acceleration.
    
    Args:
        cooccurrence: Co-occurrence matrix (numpy array or torch tensor)
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Inclusion index matrix
    """
    if not _torch_available:
        from bookdb.utils.python_utils import inclusion_index
        return inclusion_index(cooccurrence)
    
    if not _should_use_torch(cooccurrence.shape):
        logger.info(f"Matrix too large for MPS, falling back to NumPy")
        from bookdb.utils.python_utils import inclusion_index
        return inclusion_index(cooccurrence)
    
    if device is None:
        device = get_device()
    
    if isinstance(cooccurrence, np.ndarray):
        if hasattr(cooccurrence, 'toarray'):
            cooccurrence = cooccurrence.toarray()
        tensor = torch.from_numpy(cooccurrence.astype(np.float32)).to(device)
    else:
        tensor = cooccurrence.float().to(device)
    
    diag = tensor.diagonal()
    diag_rows, diag_cols = _get_row_and_column_matrix_torch(diag, device)
    
    denominator = torch.minimum(diag_rows, diag_cols)
    
    result = torch.where(
        denominator != 0,
        tensor / denominator,
        torch.zeros_like(tensor)
    )
    
    return result.cpu().numpy()


def compute_scores_torch(
    user_affinity: Union[np.ndarray, "scipy.sparse.spmatrix"],
    item_similarity: np.ndarray,
    user_ids: list,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Compute recommendation scores using PyTorch acceleration.
    
    This is the core scoring operation: scores = user_affinity @ item_similarity
    
    Args:
        user_affinity: User-item affinity matrix (sparse or dense)
        item_similarity: Item-item similarity matrix
        user_ids: List of user indices to score
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        numpy.ndarray: Score matrix for specified users
    """
    if not _torch_available:
        # Fallback to scipy sparse multiplication
        if hasattr(user_affinity, 'dot'):
            return user_affinity[user_ids, :].dot(item_similarity)
        else:
            return user_affinity[user_ids, :] @ item_similarity
    
    if device is None:
        device = get_device()
    
    # Convert user affinity slice to torch tensor
    if hasattr(user_affinity, 'toarray'):
        # Sparse matrix
        user_slice = user_affinity[user_ids, :].toarray().astype(np.float32)
    else:
        user_slice = np.asarray(user_affinity[user_ids, :]).astype(np.float32)
    
    # Ensure item_similarity is the right shape
    if item_similarity.shape[0] != user_slice.shape[1]:
        raise ValueError(
            f"Shape mismatch: user_affinity has {user_slice.shape[1]} items, "
            f"item_similarity has {item_similarity.shape[0]} items"
        )
    
    # Convert to torch tensors on device
    user_tensor = torch.from_numpy(user_slice).to(device)
    sim_tensor = torch.from_numpy(item_similarity.astype(np.float32)).to(device)
    
    # Compute scores
    scores = torch.matmul(user_tensor, sim_tensor)
    
    return scores.cpu().numpy()


def get_top_k_scored_items_torch(
    scores: Union[np.ndarray, "torch.Tensor"],
    top_k: int,
    sort_top_k: bool = True,
    device: Optional["torch.device"] = None,
) -> tuple:
    """
    Extract top K items using PyTorch acceleration.
    
    Args:
        scores: Score matrix (users x items)
        top_k: Number of top items to extract
        sort_top_k: Whether to sort the top-k results
        device: PyTorch device to use (auto-detected if None)
        
    Returns:
        Tuple of (top_items_indices, top_scores)
    """
    if not _torch_available:
        from bookdb.utils.python_utils import get_top_k_scored_items
        return get_top_k_scored_items(scores, top_k, sort_top_k)
    
    if device is None:
        device = get_device()
    
    # Convert to torch tensor if needed
    if isinstance(scores, np.ndarray):
        if hasattr(scores, 'toarray'):
            scores = scores.toarray()
        tensor = torch.from_numpy(scores.astype(np.float32)).to(device)
    else:
        tensor = scores.float().to(device)
    
    if tensor.shape[1] < top_k:
        logger.warning(
            "Number of items is less than top_k, limiting top_k to number of items"
        )
    k = min(top_k, tensor.shape[1])
    
    # Use torch.topk for efficient extraction
    top_scores, top_items = torch.topk(tensor, k, dim=1)
    
    if sort_top_k:
        # topk already returns sorted results
        pass
    else:
        # Randomly shuffle within top-k if not sorting
        # (torch.topk returns sorted, so we'd need to unsort if needed)
        pass
    
    return top_items.cpu().numpy(), top_scores.cpu().numpy()


# Dictionary mapping similarity types to torch functions
SIMILARITY_FUNCTIONS_TORCH = {
    "jaccard": jaccard_torch,
    "lift": lift_torch,
    "cosine": cosine_similarity_torch,
    "mutual_information": mutual_information_torch,
    "lexicographers_mutual_information": lexicographers_mutual_information_torch,
    "inclusion_index": inclusion_index_torch,
}


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return _torch_available


def is_mps_available() -> bool:
    """Check if MPS acceleration is available."""
    return _mps_available


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return _cuda_available
