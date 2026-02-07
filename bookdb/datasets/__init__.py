"""
Dataset processing and CRUD operations for BookDB.

This module provides tools for importing datasets and performing
database operations on books, authors, users, and lists.

Quick start:
    from bookdb.datasets import DatasetProcessor

    # Import a dataset
    stats = DatasetProcessor.process_dataset("books.csv")
    print(stats)

    # Preview before importing
    preview = DatasetProcessor.preview("books.parquet", n_rows=5)

    # Validate a file
    info = DatasetProcessor.validate_file("books.json")
"""

from .crud import (
    AuthorCRUD,
    BookCRUD,
    UserCRUD,
    BookListCRUD,
    BaseCRUD,
)
from .processor import (
    DatasetProcessor,
    ColumnMapping,
    ProcessingStats,
)

__all__ = [
    # CRUD operations
    "AuthorCRUD",
    "BookCRUD",
    "UserCRUD",
    "BookListCRUD",
    "BaseCRUD",
    # Dataset processing
    "DatasetProcessor",
    "ColumnMapping",
    "ProcessingStats",
]
