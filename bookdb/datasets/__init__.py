"""
Dataset processing and CRUD operations for BookDB.

Quick start:
    from bookdb.datasets import import_authors, import_books

    # Import Goodreads data
    stats = import_authors("data/raw_goodreads_book_authors.parquet")
    stats = import_books("data/raw_goodreads_book_works.parquet")

    # Or use generic import for simple CSV/Parquet files
    stats = import_dataset("books.csv")
"""

from .crud import (
    AuthorCRUD,
    BookCRUD,
    UserCRUD,
    BookListCRUD,
)
from .processor import (
    import_dataset,
    import_authors,
    import_books,
    preview_dataset,
    read_file,
)

__all__ = [
    # CRUD operations
    "AuthorCRUD",
    "BookCRUD",
    "UserCRUD",
    "BookListCRUD",
    # Goodreads imports
    "import_authors",
    "import_books",
    # Generic imports
    "import_dataset",
    "preview_dataset",
    "read_file",
]
