"""Unified repository wrapping PostgreSQL and ChromaDB.

Provides file-based imports (CSV/Parquet) and single-row inserts that
store structured data in PostgreSQL and embeddings in ChromaDB in one
call.  Use as a context manager for graceful session shutdown.

Usage:
    with Repository() as repo:
        # File-based imports
        preview = repo.preview("authors.parquet")
        stats  = repo.import_authors("authors.parquet")
        stats  = repo.import_books("books.parquet")

        # Single-row inserts
        repo.insert_author(name="George Orwell")
        repo.insert_book(title="1984", publication_year=1949)
        repo.insert_user(email="a@b.com", name="Alice")
"""

from pathlib import Path

from bookdb.db.models import Author, Book, User
from bookdb.db.session import SessionLocal
from bookdb.vector_db.collections import CollectionManager
from bookdb.vector_db.crud import BaseVectorCRUD
from bookdb.vector_db.schemas import CollectionNames

from .crud import AuthorCRUD, BookCRUD, UserCRUD
from .processor import read_file, safe_int, parse_authors_field

# Fake embedding dimension for testing (swap out once a real model is chosen)
FAKE_EMBEDDING_DIM = 4


def _fake_embedding() -> list[float]:
    return [1.0] * FAKE_EMBEDDING_DIM


def _clean_metadata(data: dict) -> dict:
    """Remove None values — ChromaDB rejects them in metadata."""
    return {k: v for k, v in data.items() if v is not None}


class Repository:
    """Unified access to PostgreSQL + ChromaDB for books, authors, and users."""

    def __init__(self):
        # PostgreSQL
        self._session = SessionLocal()

        # ChromaDB
        self._cm = CollectionManager()
        self._cm.initialize_collections()

        self._books_vec = BaseVectorCRUD(
            self._cm.get_collection(CollectionNames.BOOKS)
        )
        self._users_vec = BaseVectorCRUD(
            self._cm.get_collection(CollectionNames.USERS)
        )
        self._authors_vec = BaseVectorCRUD(
            self._cm.get_collection(CollectionNames.AUTHORS)
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self):
        """Close the PostgreSQL session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if exc_type is not None:
            self._session.rollback()
        self.close()
        return False

    # ── preview ──────────────────────────────────────────────────────

    def preview(self, file_path: str | Path, n: int = 5) -> list[dict]:
        """Return the first *n* rows of a CSV/Parquet file as-is."""
        df = read_file(file_path)
        return df.head(n).to_dicts()

    # ── file-based imports ───────────────────────────────────────────

    def import_authors(
        self,
        file_path: str | Path,
        batch_size: int = 1000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Import authors from a CSV/Parquet file into PostgreSQL + ChromaDB.

        Expected columns: name, author_id
        """
        stats = {
            "rows": 0,
            "authors_created": 0,
            "authors_skipped": 0,
            "errors": 0,
        }

        df = read_file(file_path)
        if limit:
            df = df.head(limit)

        rows = df.to_dicts()
        stats["rows"] = len(rows)

        seen_names: set[str] = set()

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]

            pg_authors: list[Author] = []

            for row in batch_rows:
                try:
                    name = row.get("name")
                    if not name or not str(name).strip():
                        stats["errors"] += 1
                        continue

                    name = str(name).strip()
                    name_key = name.lower()

                    if name_key in seen_names:
                        stats["authors_skipped"] += 1
                        continue
                    seen_names.add(name_key)

                    author_kwargs: dict = {}
                    ext_id = row.get("author_id")
                    if ext_id is not None:
                        author_kwargs["external_id"] = str(ext_id)

                    author = AuthorCRUD.create(
                        self._session, name, **author_kwargs
                    )
                    pg_authors.append(author)
                    stats["authors_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if pg_authors:
                self._session.commit()

                self._authors_vec.add_batch(
                    ids=[f"author_{a.id}" for a in pg_authors],
                    documents=[f"Author: {a.name}" for a in pg_authors],
                    metadatas=[
                        _clean_metadata({"name": a.name})
                        for a in pg_authors
                    ],
                    embeddings=[_fake_embedding() for _ in pg_authors],
                )

        return stats

    def import_books(
        self,
        file_path: str | Path,
        batch_size: int = 1000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Import books from a CSV/Parquet file into PostgreSQL + ChromaDB.

        Expected columns: title, publication_year, description, image_url,
                          book_id, authors (list of {author_id, role} dicts)
        """
        stats = {
            "rows": 0,
            "books_created": 0,
            "books_skipped": 0,
            "authors_linked": 0,
            "errors": 0,
        }

        df = read_file(file_path)
        if limit:
            df = df.head(limit)

        rows = df.to_dicts()
        stats["rows"] = len(rows)

        seen_titles: set[str] = set()

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]

            # Bulk-fetch authors for this batch
            batch_author_ids: set[str] = set()
            for row in batch_rows:
                batch_author_ids.update(parse_authors_field(row.get("authors")))

            author_map = AuthorCRUD.bulk_get_by_external_ids(
                self._session, list(batch_author_ids)
            )

            pg_books: list[Book] = []

            for row in batch_rows:
                try:
                    title = row.get("title")
                    if not title or not str(title).strip():
                        stats["errors"] += 1
                        continue

                    title = str(title).strip()
                    title_key = title.lower()

                    if title_key in seen_titles:
                        stats["books_skipped"] += 1
                        continue
                    seen_titles.add(title_key)

                    book_data = {
                        "title": title,
                        "publication_year": safe_int(row.get("publication_year")),
                        "description": row.get("description") or None,
                        "image_url": row.get("image_url") or None,
                        "book_id": str(row["book_id"]) if row.get("book_id") else None,
                    }

                    book = BookCRUD.create(self._session, **book_data)

                    # Link authors via junction table
                    ext_ids = parse_authors_field(row.get("authors"))
                    linked = [author_map[eid] for eid in ext_ids if eid in author_map]
                    if linked:
                        book.authors = linked
                        stats["authors_linked"] += len(linked)

                    pg_books.append(book)
                    stats["books_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if pg_books:
                self._session.commit()

                self._books_vec.add_batch(
                    ids=[f"book_{b.id}" for b in pg_books],
                    documents=[f"Title: {b.title}" for b in pg_books],
                    metadatas=[
                        _clean_metadata({
                            "title": b.title,
                            "publication_year": b.publication_year,
                        })
                        for b in pg_books
                    ],
                    embeddings=[_fake_embedding() for _ in pg_books],
                )

        return stats

    # ── single-row inserts ───────────────────────────────────────────

    def insert_author(self, name: str, **kwargs) -> Author:
        """Insert a single author into PostgreSQL + ChromaDB."""
        author = AuthorCRUD.create(self._session, name, **kwargs)
        self._session.commit()

        self._authors_vec.add(
            id=f"author_{author.id}",
            document=f"Author: {name}",
            metadata=_clean_metadata({"name": name}),
            embedding=_fake_embedding(),
        )
        return author

    def insert_book(self, **kwargs) -> Book:
        """Insert a single book into PostgreSQL + ChromaDB."""
        book = BookCRUD.create(self._session, **kwargs)
        self._session.commit()

        self._books_vec.add(
            id=f"book_{book.id}",
            document=f"Title: {book.title}",
            metadata=_clean_metadata({
                "title": book.title,
                "publication_year": book.publication_year,
            }),
            embedding=_fake_embedding(),
        )
        return book

    def insert_user(self, email: str, name: str) -> User:
        """Insert a single user into PostgreSQL + ChromaDB."""
        user = UserCRUD.create(self._session, email, name)
        self._session.commit()

        self._users_vec.add(
            id=f"user_{user.id}",
            document=f"User: {name}",
            metadata=_clean_metadata({
                "user_id": user.id,
                "name": name,
            }),
            embedding=_fake_embedding(),
        )
        return user
