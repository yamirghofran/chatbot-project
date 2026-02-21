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
        stats  = repo.import_reviews("reviews.parquet")

        # Single-row inserts
        repo.insert_author(name="George Orwell")
        repo.insert_book(title="1984", author_names=["George Orwell"])
        repo.insert_user(email="a@b.com", name="Alice")
        repo.insert_review(user_id=1, book_id=1, text="Great!", rating=5)
        repo.upsert_rating(user_id=1, book_id=1, rating=5)

        # Book list management
        bl = repo.create_book_list(user_id=1, name="Favorites")
        repo.add_book_to_list(list_id=bl.id, book_id=1)
"""

import logging
from pathlib import Path

from bookdb.db.models import Author, Book, BookList, Rating, Review, User
from bookdb.db.session import SessionLocal
from bookdb.vector_db.book_crud import BookVectorCRUD
from bookdb.vector_db.collections import CollectionManager
from bookdb.vector_db.crud import BaseVectorCRUD
from bookdb.vector_db.review_crud import ReviewVectorCRUD
from bookdb.vector_db.schemas import (
    AuthorMetadata,
    BookMetadata,
    CollectionNames,
    ReviewMetadata,
    UserMetadata,
)
from bookdb.vector_db.user_crud import UserVectorCRUD

from .crud import AuthorCRUD, BookCRUD, BookListCRUD, RatingCRUD, ReviewCRUD, UserCRUD
from .processor import parse_authors_field, read_file, safe_int

logger = logging.getLogger(__name__)

# Swap out once a real embedding model is chosen
FAKE_EMBEDDING_DIM = 4


def _fake_embedding() -> list[float]:
    return [1.0] * FAKE_EMBEDDING_DIM


class Repository:
    """Unified access to PostgreSQL + ChromaDB for books, authors, users, and reviews."""

    def __init__(self):
        self._session = SessionLocal()

        self._cm = CollectionManager()
        self._cm.initialize_collections()

        self._books_vec = BookVectorCRUD(
            self._cm.get_collection(CollectionNames.BOOKS)
        )
        self._users_vec = UserVectorCRUD(
            self._cm.get_collection(CollectionNames.USERS)
        )
        self._reviews_vec = ReviewVectorCRUD(
            self._cm.get_collection(CollectionNames.REVIEWS)
        )
        # Authors use BaseVectorCRUD directly (simple metadata, no specialized class)
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

    # ── ChromaDB helpers ─────────────────────────────────────────────

    def _vec_existing_ids(self, crud: BaseVectorCRUD, ids: list[str]) -> set[str]:
        """Return the subset of *ids* that already exist in the collection."""
        if not ids:
            return set()
        result = crud.collection.get(ids=ids)
        return set(result["ids"])

    def _vec_add_batch_skip_existing(
        self,
        crud: BaseVectorCRUD,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """Add a batch to ChromaDB, silently skipping IDs that already exist.

        Returns the number of items actually added.
        """
        existing = self._vec_existing_ids(crud, ids)
        if not existing:
            crud.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            return len(ids)

        new_ids, new_docs, new_metas, new_embeds = [], [], [], []
        for i, id_ in enumerate(ids):
            if id_ not in existing:
                new_ids.append(id_)
                new_docs.append(documents[i])
                new_metas.append(metadatas[i])
                new_embeds.append(embeddings[i])

        if new_ids:
            crud.collection.add(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metas,
                embeddings=new_embeds,
            )
        return len(new_ids)

    def _vec_add_if_missing(
        self,
        crud: BaseVectorCRUD,
        id: str,
        document: str,
        metadata: dict,
        embedding: list[float],
    ) -> bool:
        """Add a single item only if it doesn't already exist. Returns True if added."""
        if crud.exists(id):
            return False
        crud.collection.add(
            ids=[id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return True

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
            "vec_added": 0,
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

                ids = [f"author_{a.id}" for a in pg_authors]
                docs = [f"Author: {a.name}" for a in pg_authors]
                metas = [
                    AuthorMetadata(name=a.name).model_dump(exclude_none=True)
                    for a in pg_authors
                ]
                embeds = [_fake_embedding() for _ in pg_authors]

                stats["vec_added"] += self._vec_add_batch_skip_existing(
                    self._authors_vec, ids, docs, metas, embeds
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
            "vec_added": 0,
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

            batch_author_ids: set[str] = set()
            for row in batch_rows:
                batch_author_ids.update(parse_authors_field(row.get("authors")))

            author_map = AuthorCRUD.bulk_get_by_external_ids(
                self._session, list(batch_author_ids)
            )

            pg_books: list[tuple[Book, list[Author]]] = []

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

                    ext_ids = parse_authors_field(row.get("authors"))
                    linked = [author_map[eid] for eid in ext_ids if eid in author_map]
                    if linked:
                        book.authors = linked
                        stats["authors_linked"] += len(linked)

                    pg_books.append((book, linked))
                    stats["books_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if pg_books:
                self._session.commit()

                ids = [f"book_{b.id}" for b, _ in pg_books]
                docs = [
                    b.description or f"{b.title}"
                    for b, _ in pg_books
                ]
                metas = [
                    BookMetadata(
                        title=b.title,
                        author=", ".join(a.name for a in authors) or "Unknown",
                        publication_year=b.publication_year,
                    ).model_dump(exclude_none=True)
                    for b, authors in pg_books
                ]
                embeds = [_fake_embedding() for _ in pg_books]

                stats["vec_added"] += self._vec_add_batch_skip_existing(
                    self._books_vec, ids, docs, metas, embeds
                )

        return stats

    def import_reviews(
        self,
        file_path: str | Path,
        batch_size: int = 1000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Import reviews from a CSV/Parquet file into PostgreSQL + ChromaDB.

        Expected columns: user_id, book_id, review_id, rating,
                          review_text, date_added, date_updated, read_at

        Users are resolved via ``User.external_id``.  If a user doesn't
        exist yet a **stub** row is created automatically (synthetic email
        ``<prefix>@import.local``, name ``User <prefix>``).  Books are
        resolved via ``Book.book_id``.  Reviews whose book cannot be
        mapped are written to ChromaDB only (the PG insert is skipped).

        Ratings are also upserted into the ``ratings`` table whenever a
        valid PG user + book mapping exists.
        """
        stats = {
            "rows": 0,
            "reviews_created_pg": 0,
            "reviews_created_vec": 0,
            "users_created": 0,
            "ratings_upserted": 0,
            "reviews_skipped": 0,
            "vec_added": 0,
            "errors": 0,
        }

        df = read_file(file_path)
        if limit:
            df = df.head(limit)

        rows = df.to_dicts()
        stats["rows"] = len(rows)

        seen_ids: set[str] = set()

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]

            # Bulk-resolve external user_ids and book_ids for this batch
            batch_ext_user_ids: set[str] = set()
            batch_ext_book_ids: set[str] = set()
            for row in batch_rows:
                uid = row.get("user_id")
                if uid:
                    batch_ext_user_ids.add(str(uid))
                bid = row.get("book_id")
                if bid:
                    batch_ext_book_ids.add(str(bid))

            # Create stub users for any unknown external IDs
            user_map = UserCRUD.bulk_get_or_create_from_external(
                self._session, list(batch_ext_user_ids)
            )
            new_users_this_batch = sum(
                1 for u in user_map.values()
                if u.name.startswith("User ")
                and u.email.endswith("@import.local")
            )

            # Map book external IDs → PG Book objects
            book_map: dict[str, Book] = {}
            if batch_ext_book_ids:
                from sqlalchemy import select
                stmt = select(Book).where(
                    Book.book_id.in_(list(batch_ext_book_ids))
                )
                book_map = {
                    b.book_id: b
                    for b in self._session.scalars(stmt).all()
                    if b.book_id
                }

            pg_reviews: list[Review] = []
            vec_ids: list[str] = []
            vec_docs: list[str] = []
            vec_metas: list[dict] = []
            vec_embeds: list[list[float]] = []

            for row in batch_rows:
                try:
                    review_id = row.get("review_id")
                    if not review_id:
                        stats["errors"] += 1
                        continue

                    review_id = str(review_id)
                    if review_id in seen_ids:
                        stats["reviews_skipped"] += 1
                        continue
                    seen_ids.add(review_id)

                    ext_user_id = str(row.get("user_id", ""))
                    ext_book_id = str(row.get("book_id", ""))
                    rating = safe_int(row.get("rating"))
                    review_text = str(row.get("review_text") or "")

                    if not ext_user_id or not ext_book_id or not rating or not review_text:
                        stats["errors"] += 1
                        continue

                    # PG insert — only if both user and book could be mapped
                    pg_user = user_map.get(ext_user_id)
                    pg_book = book_map.get(ext_book_id)
                    if pg_user and pg_book:
                        review = ReviewCRUD.create(
                            self._session, pg_user.id, pg_book.id, review_text
                        )
                        pg_reviews.append(review)
                        stats["reviews_created_pg"] += 1

                        RatingCRUD.upsert(
                            self._session, pg_user.id, pg_book.id, rating
                        )
                        stats["ratings_upserted"] += 1

                    # ChromaDB insert — always (uses external string IDs)
                    meta = ReviewMetadata(
                        user_id=ext_user_id,
                        book_id=ext_book_id,
                        rating=rating,
                        date_added=str(row["date_added"]) if row.get("date_added") else None,
                        date_updated=str(row["date_updated"]) if row.get("date_updated") else None,
                        read_at=str(row["read_at"]) if row.get("read_at") else None,
                    )

                    chroma_id = f"review_{review_id}"
                    vec_ids.append(chroma_id)
                    vec_docs.append(review_text)
                    vec_metas.append(meta.model_dump(exclude_none=True))
                    vec_embeds.append(_fake_embedding())
                    stats["reviews_created_vec"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if pg_reviews or new_users_this_batch:
                self._session.commit()
                stats["users_created"] += new_users_this_batch

            if vec_ids:
                stats["vec_added"] += self._vec_add_batch_skip_existing(
                    self._reviews_vec, vec_ids, vec_docs, vec_metas, vec_embeds
                )

        return stats

    def import_users(
        self,
        file_path: str | Path,
        batch_size: int = 1000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Import users from a CSV/Parquet file into PostgreSQL + ChromaDB.

        Expected columns: user_id (external), and optionally name, email.

        If name/email are missing, stub values are generated from the
        external user_id (e.g. ``User abc123`` / ``abc123@import.local``).
        This is the same convention used by ``import_reviews`` when it
        creates stub users on the fly.
        """
        stats = {
            "rows": 0,
            "users_created": 0,
            "users_skipped": 0,
            "vec_added": 0,
            "errors": 0,
        }

        df = read_file(file_path)
        if limit:
            df = df.head(limit)

        rows = df.to_dicts()
        stats["rows"] = len(rows)

        seen_ext_ids: set[str] = set()

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]

            pg_users: list[User] = []

            for row in batch_rows:
                try:
                    ext_id = row.get("user_id")
                    if not ext_id or not str(ext_id).strip():
                        stats["errors"] += 1
                        continue

                    ext_id = str(ext_id).strip()
                    if ext_id in seen_ext_ids:
                        stats["users_skipped"] += 1
                        continue
                    seen_ext_ids.add(ext_id)

                    # Skip if already in PG
                    if UserCRUD.get_by_external_id(self._session, ext_id):
                        stats["users_skipped"] += 1
                        continue

                    prefix = ext_id[:12]
                    name = str(row.get("name") or f"User {prefix}").strip()
                    email = str(row.get("email") or f"{prefix}@import.local").strip()

                    user = UserCRUD.create(
                        self._session, email=email, name=name, external_id=ext_id
                    )
                    pg_users.append(user)
                    stats["users_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if pg_users:
                self._session.commit()

                ids = [f"user_{u.id}" for u in pg_users]
                docs = [f"User: {u.name}" for u in pg_users]
                metas = [
                    UserMetadata(
                        user_id=u.id, name=u.name
                    ).model_dump(exclude_none=True)
                    for u in pg_users
                ]
                embeds = [_fake_embedding() for _ in pg_users]

                stats["vec_added"] += self._vec_add_batch_skip_existing(
                    self._users_vec, ids, docs, metas, embeds
                )

        return stats

    # ── single-row inserts ───────────────────────────────────────────

    def insert_author(self, name: str, **kwargs) -> Author:
        """Insert a single author into PostgreSQL + ChromaDB."""
        author = AuthorCRUD.create(self._session, name, **kwargs)
        self._session.commit()

        meta = AuthorMetadata(name=name)
        self._vec_add_if_missing(
            self._authors_vec,
            id=f"author_{author.id}",
            document=f"Author: {name}",
            metadata=meta.model_dump(exclude_none=True),
            embedding=_fake_embedding(),
        )
        return author

    def insert_book(
        self,
        title: str,
        author_names: list[str] | None = None,
        description: str | None = None,
        **kwargs,
    ) -> Book:
        """Insert a single book into PostgreSQL + ChromaDB."""
        if author_names:
            book = BookCRUD.create_with_authors(
                self._session,
                author_names=author_names,
                title=title,
                description=description,
                **kwargs,
            )
        else:
            book = BookCRUD.create(
                self._session, title=title, description=description, **kwargs
            )
        self._session.commit()

        author_str = (
            ", ".join(a.name for a in book.authors) if book.authors else "Unknown"
        )
        meta = BookMetadata(
            title=book.title,
            author=author_str,
            publication_year=book.publication_year,
        )
        self._vec_add_if_missing(
            self._books_vec,
            id=f"book_{book.id}",
            document=book.description or book.title,
            metadata=meta.model_dump(exclude_none=True),
            embedding=_fake_embedding(),
        )
        return book

    def insert_user(self, email: str, name: str) -> User:
        """Insert a single user into PostgreSQL + ChromaDB."""
        user = UserCRUD.create(self._session, email, name)
        self._session.commit()

        meta = UserMetadata(user_id=user.id, name=name)
        self._vec_add_if_missing(
            self._users_vec,
            id=f"user_{user.id}",
            document=f"User: {name}",
            metadata=meta.model_dump(exclude_none=True),
            embedding=_fake_embedding(),
        )
        return user

    def insert_review(
        self,
        user_id: int,
        book_id: int,
        text: str,
        rating: int,
    ) -> Review:
        """Insert a review into PostgreSQL + ChromaDB and upsert the rating."""
        review = ReviewCRUD.create(self._session, user_id, book_id, text)
        RatingCRUD.upsert(self._session, user_id, book_id, rating)
        self._session.commit()

        meta = ReviewMetadata(
            user_id=str(user_id),
            book_id=str(book_id),
            rating=rating,
        )
        self._vec_add_if_missing(
            self._reviews_vec,
            id=f"review_{review.id}",
            document=text,
            metadata=meta.model_dump(exclude_none=True),
            embedding=_fake_embedding(),
        )
        return review

    def upsert_rating(self, user_id: int, book_id: int, rating: int) -> Rating:
        """Create or update a rating in PostgreSQL (no ChromaDB — ratings have no embeddings)."""
        r = RatingCRUD.upsert(self._session, user_id, book_id, rating)
        self._session.commit()
        return r

    # ── book list management (PostgreSQL only) ───────────────────────

    def create_book_list(self, user_id: int, name: str) -> BookList:
        """Create a new book list for a user."""
        bl = BookListCRUD.create(self._session, user_id, name)
        self._session.commit()
        return bl

    def add_book_to_list(self, list_id: int, book_id: int) -> bool:
        """Add a book to an existing list. Returns True on success."""
        ok = BookListCRUD.add_book(self._session, list_id, book_id)
        if ok:
            self._session.commit()
        return ok

    def remove_book_from_list(self, list_id: int, book_id: int) -> bool:
        """Remove a book from a list. Returns True if removed."""
        ok = BookListCRUD.remove_book(self._session, list_id, book_id)
        if ok:
            self._session.commit()
        return ok
