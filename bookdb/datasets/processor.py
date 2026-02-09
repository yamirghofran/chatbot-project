"""
Dataset importer for books and authors.

Supports:
- Generic CSV/Parquet import via import_dataset()
- Goodreads-specific imports via import_authors() and import_books()
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy.orm import Session

from bookdb.db.session import SessionLocal
from .crud import BookCRUD, AuthorCRUD

logger = logging.getLogger(__name__)


# Column mappings for generic imports

DEFAULT_COLUMNS = {
    "title": "title",
    "authors": "authors",
    "pages_number": "pages_number",
    "publisher_name": "publisher_name",
    "publish_year": "publish_year",
}


# Helpers

def read_file(path: str | Path) -> pl.DataFrame:
    """Read CSV or Parquet file."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pl.read_csv(path, ignore_errors=True)
    if ext == ".parquet":
        return pl.read_parquet(path)

    raise ValueError(f"Unsupported file format: {ext}")


def split_authors(raw: Any) -> list[str]:
    # Split author column is multiple are included
    if not raw:
        return []

    if isinstance(raw, str):
        return [a.strip() for a in raw.split(",") if a.strip()]

    if isinstance(raw, list):
        return [str(a).strip() for a in raw if a]

    return []


def safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def extract_book(row: dict, columns: dict) -> tuple[dict, list[str]]:
    title = row.get(columns["title"])
    if not title or not str(title).strip():
        raise ValueError("Missing title")

    authors = split_authors(row.get(columns["authors"]))

    book_data = {
        "title": str(title).strip(),
        "pages_number": safe_int(row.get(columns["pages_number"])),
        "publisher_name": row.get(columns["publisher_name"]),
        "publish_year": safe_int(row.get(columns["publish_year"])),
    }

    return book_data, authors


# Main processor
def import_dataset(
    file_path: str | Path,
    batch_size: int = 500,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import books + authors from a dataset file.
    """

    stats = {
        "rows": 0,
        "books_created": 0,
        "books_skipped": 0,
        "authors_created": 0,
        "errors": 0,
    }

    df = read_file(file_path)

    if limit:
        df = df.head(limit)

    rows = df.to_dicts()
    stats["rows"] = len(rows)

    own_session = session is None
    if own_session:
        session = SessionLocal()

    seen_titles: set[str] = set()

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for row in batch:
                try:
                    book_data, author_names = extract_book(row, DEFAULT_COLUMNS)

                    title_key = book_data["title"].lower()
                    if title_key in seen_titles:
                        stats["books_skipped"] += 1
                        continue

                    seen_titles.add(title_key)

                    BookCRUD.create_with_authors(
                        session,
                        author_names=author_names,
                        **book_data,
                    )
                    stats["books_created"] += 1

                except Exception:
                    logger.exception(f"Failed to process row: {row}")
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats


# Preview helper to inspect datasets before upload
def preview_dataset(file_path: str | Path, n: int = 5) -> list[dict]:
    """Return the first n rows of a dataset file as-is."""
    df = read_file(file_path)
    return df.head(n).to_dicts()


# Importers for individual models (not paired)
def parse_rating_dist(rating_str: str | None) -> dict[str, int]:
    """
    Parse Goodreads rating_dist string.
    Format: "5:count|4:count|3:count|2:count|1:count|total:count"
    """
    result = {
        "rating_dist_1": 0,
        "rating_dist_2": 0,
        "rating_dist_3": 0,
        "rating_dist_4": 0,
        "rating_dist_5": 0,
        "rating_dist_total": 0,
    }

    if not rating_str:
        return result

    try:
        for part in rating_str.split("|"):
            key, val = part.split(":")
            count = int(val)
            if key == "total":
                result["rating_dist_total"] = count
            elif key in ("1", "2", "3", "4", "5"):
                result[f"rating_dist_{key}"] = count
    except (ValueError, AttributeError):
        pass

    return result


def import_authors(
    file_path: str | Path = None,
    batch_size: int = 1000,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import authors from Goodreads authors dataset.

    Expected columns: name, author_id, average_rating, ratings_count, text_reviews_count
    """
    stats = {
        "rows": 0,
        "authors_created": 0,
        "authors_skipped": 0,
        "errors": 0,
    }

    df = pl.read_parquet(file_path)

    if limit:
        df = df.head(limit)

    rows = df.to_dicts()
    stats["rows"] = len(rows)

    own_session = session is None
    if own_session:
        session = SessionLocal()

    seen_names: set[str] = set()

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for row in batch:
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

                    author_data = {}
                    avg = row.get("average_rating")
                    if avg is not None:
                        try:
                            author_data["average_rating"] = float(avg)
                        except (TypeError, ValueError):
                            pass
                    rc = safe_int(row.get("ratings_count"))
                    if rc is not None:
                        author_data["ratings_count"] = rc
                    trc = safe_int(row.get("text_reviews_count"))
                    if trc is not None:
                        author_data["text_reviews_count"] = trc

                    AuthorCRUD.create(session, name, **author_data)
                    stats["authors_created"] += 1

                except Exception:
                    logger.exception(f"Failed to process row: {row}")
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats


def import_books(
    file_path: str | Path = None,
    batch_size: int = 1000,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import books from Goodreads book works dataset.

    Expected columns: original_title, original_publication_year/month/day,
                      rating_dist, ratings_count, reviews_count, work_id
    """
    stats = {
        "rows": 0,
        "books_created": 0,
        "books_skipped": 0,
        "errors": 0,
    }

    df = pl.read_parquet(file_path)

    if limit:
        df = df.head(limit)

    rows = df.to_dicts()
    stats["rows"] = len(rows)

    own_session = session is None
    if own_session:
        session = SessionLocal()

    seen_titles: set[str] = set()

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for row in batch:
                try:
                    title = row.get("original_title")
                    if not title or not str(title).strip():
                        stats["errors"] += 1
                        continue

                    title = str(title).strip()
                    title_key = title.lower()

                    if title_key in seen_titles:
                        stats["books_skipped"] += 1
                        continue

                    seen_titles.add(title_key)

                    # Parse rating distribution
                    ratings = parse_rating_dist(row.get("rating_dist"))

                    book_data = {
                        "title": title,
                        "publish_year": safe_int(row.get("original_publication_year")),
                        "publish_month": safe_int(row.get("original_publication_month")),
                        "publish_day": safe_int(row.get("original_publication_day")),
                        "num_reviews": safe_int(row.get("reviews_count")) or 0,
                        "text_reviews_count": safe_int(row.get("text_reviews_count")),
                        "ratings_count": safe_int(row.get("ratings_count")),
                        "ratings_sum": safe_int(row.get("ratings_sum")),
                        "books_count": safe_int(row.get("books_count")),
                        "media_type": row.get("media_type") or None,
                        "best_book_id": row.get("best_book_id") or None,
                        "work_id": row.get("work_id") or None,
                        "original_language_id": row.get("original_language_id") or None,
                        "default_description_language_code": row.get("default_description_language_code") or None,
                        "default_chaptering_book_id": row.get("default_chaptering_book_id") or None,
                        **ratings,
                    }

                    BookCRUD.create(session, **book_data)
                    stats["books_created"] += 1

                except Exception:
                    logger.exception(f"Failed to process row: {row}")
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats



