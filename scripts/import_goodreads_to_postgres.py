from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from bookdb.db.models import Author, Book, BookAuthor, BookList, BookRating, ListBook, Review, User
from bookdb.db.session import SessionLocal

DEFAULT_PASSWORD_HASH = "imported-user-password"
DEFAULT_EMAIL_DOMAIN = "import.bookdb.local"


def resolve_project_root(start: Path) -> Path:
    current = start.resolve()
    if (current / "data").exists() and (current / "bookdb").exists():
        return current
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "bookdb").exists():
            return parent
    return current


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def ts_to_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    ts = to_int(value)
    if ts is None:
        return fallback or utcnow()
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def extract_author_ids(raw_authors: Any) -> list[int]:
    if raw_authors is None:
        return []

    ids: list[int] = []
    if isinstance(raw_authors, list):
        for item in raw_authors:
            if isinstance(item, dict):
                author_id = to_int(item.get("author_id"))
            else:
                author_id = to_int(item)
            if author_id is not None:
                ids.append(author_id)
    elif isinstance(raw_authors, dict):
        author_id = to_int(raw_authors.get("author_id"))
        if author_id is not None:
            ids.append(author_id)
    elif isinstance(raw_authors, str):
        for token in raw_authors.replace("{", "").replace("}", "").replace('"', "").split(","):
            author_id = to_int(token)
            if author_id is not None:
                ids.append(author_id)

    return list(dict.fromkeys(ids))


def chunk_reader(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: Iterable[Any],
    batch_size: int,
):
    cursor = con.execute(query, list(params))
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def affected_rows(result: Any, fallback: int) -> int:
    rowcount = int(getattr(result, "rowcount", 0) or 0)
    return fallback if rowcount < 0 else rowcount


def ensure_users(session: Session, user_ids: set[int]) -> int:
    if not user_ids:
        return 0

    now = utcnow()
    payload = [
        {
            "goodreads_id": user_id,
            "name": f"User {user_id}",
            "username": f"user_{user_id}",
            "email": f"user_{user_id}@{DEFAULT_EMAIL_DOMAIN}",
            "password_hash": DEFAULT_PASSWORD_HASH,
            "created_at": now,
            "updated_at": now,
        }
        for user_id in sorted(user_ids)
    ]

    stmt = pg_insert(User).values(payload)
    stmt = stmt.on_conflict_do_nothing(index_elements=[User.goodreads_id])
    result = session.execute(stmt)
    return affected_rows(result, len(payload))


def map_books(session: Session, goodreads_ids: set[int]) -> dict[int, int]:
    if not goodreads_ids:
        return {}
    rows = session.execute(
        select(Book.goodreads_id, Book.id).where(Book.goodreads_id.in_(goodreads_ids))
    ).all()
    return {int(gr_id): int(book_id) for gr_id, book_id in rows}


def map_users(session: Session, goodreads_ids: set[int]) -> dict[int, int]:
    if not goodreads_ids:
        return {}
    rows = session.execute(
        select(User.goodreads_id, User.id).where(User.goodreads_id.in_(goodreads_ids))
    ).all()
    return {int(gr_id): int(user_id) for gr_id, user_id in rows if gr_id is not None}


def map_authors(session: Session, goodreads_ids: set[int]) -> dict[int, int]:
    if not goodreads_ids:
        return {}
    rows = session.execute(
        select(Author.goodreads_id, Author.id).where(Author.goodreads_id.in_(goodreads_ids))
    ).all()
    return {int(gr_id): int(author_id) for gr_id, author_id in rows if gr_id is not None}


def map_lists(session: Session, user_ids: set[int], title: str) -> dict[int, int]:
    if not user_ids:
        return {}
    rows = session.execute(
        select(BookList.user_id, BookList.id).where(
            BookList.user_id.in_(user_ids),
            BookList.title == title,
        )
    ).all()
    return {int(user_id): int(list_id) for user_id, list_id in rows}


def import_authors(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    authors_path: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, int]:
    stats = {"rows": 0, "upserted": 0, "skipped": 0}
    limit_sql = f" LIMIT {limit}" if limit is not None else ""
    query = (
        "SELECT author_id, name "
        "FROM read_parquet(?) "
        "WHERE author_id IS NOT NULL "
        f"{limit_sql}"
    )

    for batch in chunk_reader(con, query, [str(authors_path)], batch_size):
        stats["rows"] += len(batch)
        now = utcnow()
        payload: list[dict[str, Any]] = []
        for author_id, name in batch:
            parsed_author_id = to_int(author_id)
            parsed_name = to_text(name)
            if parsed_author_id is None or parsed_name is None:
                stats["skipped"] += 1
                continue
            payload.append(
                {
                    "goodreads_id": parsed_author_id,
                    "name": parsed_name,
                    "description": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        if not payload:
            continue

        stmt = pg_insert(Author).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Author.goodreads_id],
            set_={
                "name": stmt.excluded.name,
                "description": stmt.excluded.description,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        result = session.execute(stmt)
        session.commit()
        stats["upserted"] += affected_rows(result, len(payload))

    return stats


def import_books(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    books_path: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, int]:
    stats = {"rows": 0, "upserted_books": 0, "linked_authors": 0, "skipped": 0}
    limit_sql = f" LIMIT {limit}" if limit is not None else ""
    query = (
        "SELECT book_id, title, description, image_url, format, publisher, publication_year, isbn13, authors "
        "FROM read_parquet(?) "
        f"{limit_sql}"
    )

    for batch in chunk_reader(con, query, [str(books_path)], batch_size):
        stats["rows"] += len(batch)
        now = utcnow()

        book_payload: list[dict[str, Any]] = []
        book_author_map: dict[int, list[int]] = {}

        for (
            goodreads_id,
            title,
            description,
            image_url,
            format_value,
            publisher,
            publication_year,
            isbn13,
            authors_raw,
        ) in batch:
            parsed_goodreads_id = to_int(goodreads_id)
            parsed_title = to_text(title)
            if parsed_goodreads_id is None or parsed_title is None:
                stats["skipped"] += 1
                continue

            book_payload.append(
                {
                    "goodreads_id": parsed_goodreads_id,
                    "title": parsed_title,
                    "description": to_text(description),
                    "image_url": to_text(image_url),
                    "format": to_text(format_value),
                    "publisher": to_text(publisher),
                    "publication_year": to_int(publication_year),
                    "isbn13": to_text(isbn13),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            book_author_map[parsed_goodreads_id] = extract_author_ids(authors_raw)

        if not book_payload:
            continue

        stmt = pg_insert(Book).values(book_payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Book.goodreads_id],
            set_={
                "title": stmt.excluded.title,
                "description": stmt.excluded.description,
                "image_url": stmt.excluded.image_url,
                "format": stmt.excluded.format,
                "publisher": stmt.excluded.publisher,
                "publication_year": stmt.excluded.publication_year,
                "isbn13": stmt.excluded.isbn13,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        result = session.execute(stmt)
        stats["upserted_books"] += affected_rows(result, len(book_payload))

        internal_book_ids = map_books(session, set(book_author_map.keys()))
        all_author_ids = {
            author_id
            for author_ids in book_author_map.values()
            for author_id in author_ids
        }
        existing_author_ids = map_authors(session, all_author_ids)

        relation_payload: list[dict[str, Any]] = []
        for goodreads_id, author_ids in book_author_map.items():
            internal_book_id = internal_book_ids.get(goodreads_id)
            if internal_book_id is None:
                continue
            for goodreads_author_id in author_ids:
                internal_author_id = existing_author_ids.get(goodreads_author_id)
                if internal_author_id is None:
                    continue
                relation_payload.append(
                    {
                        "book_id": internal_book_id,
                        "author_id": internal_author_id,
                        "created_at": now,
                        "updated_at": now,
                    }
                )

        if relation_payload:
            rel_stmt = pg_insert(BookAuthor).values(relation_payload)
            rel_stmt = rel_stmt.on_conflict_do_nothing(index_elements=[BookAuthor.book_id, BookAuthor.author_id])
            rel_result = session.execute(rel_stmt)
            stats["linked_authors"] += affected_rows(rel_result, len(relation_payload))

        session.commit()

    return stats


def import_reviews(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    reviews_path: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, int]:
    stats = {"rows": 0, "upserted_reviews": 0, "users_created": 0, "skipped": 0}
    limit_sql = f" LIMIT {limit}" if limit is not None else ""
    query = (
        "SELECT review_id, review_text, book_id, user_id, ts_updated "
        "FROM read_parquet(?) "
        f"{limit_sql}"
    )

    for batch in chunk_reader(con, query, [str(reviews_path)], batch_size):
        stats["rows"] += len(batch)
        goodreads_book_ids: set[int] = set()
        goodreads_user_ids: set[int] = set()
        parsed_rows: list[tuple[str, str, int, int, datetime]] = []

        for review_id, review_text, book_id, user_id, ts_updated in batch:
            parsed_review_id = to_text(review_id)
            parsed_review_text = to_text(review_text)
            parsed_book_id = to_int(book_id)
            parsed_user_id = to_int(user_id)
            if (
                parsed_review_id is None
                or parsed_review_text is None
                or parsed_book_id is None
                or parsed_user_id is None
            ):
                stats["skipped"] += 1
                continue

            parsed_rows.append(
                (
                    parsed_review_id,
                    parsed_review_text,
                    parsed_book_id,
                    parsed_user_id,
                    ts_to_datetime(ts_updated),
                )
            )
            goodreads_book_ids.add(parsed_book_id)
            goodreads_user_ids.add(parsed_user_id)

        if not parsed_rows:
            continue

        stats["users_created"] += ensure_users(session, goodreads_user_ids)
        internal_book_ids = map_books(session, goodreads_book_ids)
        internal_user_ids = map_users(session, goodreads_user_ids)

        payload: list[dict[str, Any]] = []
        for review_id, review_text, goodreads_book_id, goodreads_user_id, event_time in parsed_rows:
            internal_book_id = internal_book_ids.get(goodreads_book_id)
            internal_user_id = internal_user_ids.get(goodreads_user_id)
            if internal_book_id is None or internal_user_id is None:
                stats["skipped"] += 1
                continue
            payload.append(
                {
                    "goodreads_id": review_id,
                    "user_id": internal_user_id,
                    "book_id": internal_book_id,
                    "review_text": review_text,
                    "created_at": event_time,
                    "updated_at": event_time,
                }
            )

        if payload:
            stmt = pg_insert(Review).values(payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Review.goodreads_id],
                set_={
                    "user_id": stmt.excluded.user_id,
                    "book_id": stmt.excluded.book_id,
                    "review_text": stmt.excluded.review_text,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            result = session.execute(stmt)
            stats["upserted_reviews"] += affected_rows(result, len(payload))

        session.commit()

    return stats


def import_interactions(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    interactions_path: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, int]:
    stats = {
        "rows": 0,
        "users_created": 0,
        "lists_created": 0,
        "list_books_added": 0,
        "ratings_upserted": 0,
        "skipped": 0,
    }
    limit_sql = f" LIMIT {limit}" if limit is not None else ""
    query = (
        "SELECT user_id, book_id, is_read, rating, timestamp "
        "FROM read_parquet(?) "
        f"{limit_sql}"
    )

    for batch in chunk_reader(con, query, [str(interactions_path)], batch_size):
        stats["rows"] += len(batch)
        parsed_rows: list[tuple[int, int, int, int, datetime]] = []
        goodreads_user_ids: set[int] = set()
        goodreads_book_ids: set[int] = set()

        for user_id, book_id, is_read, rating, ts_value in batch:
            parsed_user_id = to_int(user_id)
            parsed_book_id = to_int(book_id)
            if parsed_user_id is None or parsed_book_id is None:
                stats["skipped"] += 1
                continue

            parsed_is_read = to_int(is_read) or 0
            parsed_rating = to_int(rating) or 0
            event_time = ts_to_datetime(ts_value)

            parsed_rows.append((parsed_user_id, parsed_book_id, parsed_is_read, parsed_rating, event_time))
            goodreads_user_ids.add(parsed_user_id)
            goodreads_book_ids.add(parsed_book_id)

        if not parsed_rows:
            continue

        stats["users_created"] += ensure_users(session, goodreads_user_ids)
        internal_book_ids = map_books(session, goodreads_book_ids)
        internal_user_ids = map_users(session, goodreads_user_ids)

        read_rows: list[tuple[int, int, datetime]] = []
        rating_payload: list[dict[str, Any]] = []
        read_user_ids: set[int] = set()

        for goodreads_user_id, goodreads_book_id, is_read, rating, event_time in parsed_rows:
            internal_book_id = internal_book_ids.get(goodreads_book_id)
            internal_user_id = internal_user_ids.get(goodreads_user_id)
            if internal_book_id is None or internal_user_id is None:
                stats["skipped"] += 1
                continue

            if is_read == 1:
                read_rows.append((internal_user_id, internal_book_id, event_time))
                read_user_ids.add(internal_user_id)

            if 1 <= rating <= 5:
                rating_payload.append(
                    {
                        "user_id": internal_user_id,
                        "book_id": internal_book_id,
                        "rating": rating,
                        "created_at": event_time,
                        "updated_at": event_time,
                    }
                )

        if read_user_ids:
            now = utcnow()
            list_payload = [
                {
                    "user_id": user_id,
                    "title": "read",
                    "description": None,
                    "created_at": now,
                    "updated_at": now,
                }
                for user_id in sorted(read_user_ids)
            ]
            list_stmt = pg_insert(BookList).values(list_payload)
            list_stmt = list_stmt.on_conflict_do_nothing(index_elements=[BookList.user_id, BookList.title])
            list_result = session.execute(list_stmt)
            stats["lists_created"] += affected_rows(list_result, len(list_payload))

            list_map = map_lists(session, read_user_ids, "read")
            list_book_payload: list[dict[str, Any]] = []
            for user_id, internal_book_id, event_time in read_rows:
                list_id = list_map.get(user_id)
                if list_id is None:
                    continue
                list_book_payload.append(
                    {
                        "list_id": list_id,
                        "book_id": internal_book_id,
                        "added_at": event_time,
                        "created_at": event_time,
                        "updated_at": event_time,
                    }
                )

            if list_book_payload:
                list_book_stmt = pg_insert(ListBook).values(list_book_payload)
                list_book_stmt = list_book_stmt.on_conflict_do_nothing(index_elements=[ListBook.list_id, ListBook.book_id])
                list_book_result = session.execute(list_book_stmt)
                stats["list_books_added"] += affected_rows(list_book_result, len(list_book_payload))

        if rating_payload:
            # Deduplicate by (user_id, book_id), keeping the entry with the latest updated_at
            seen: dict[tuple[int, int], dict] = {}
            for row in rating_payload:
                key = (row["user_id"], row["book_id"])
                if key not in seen or row["updated_at"] > seen[key]["updated_at"]:
                    seen[key] = row
            rating_payload = list(seen.values())

            rating_stmt = pg_insert(BookRating).values(rating_payload)
            rating_stmt = rating_stmt.on_conflict_do_update(
                index_elements=[BookRating.user_id, BookRating.book_id],
                set_={
                    "rating": rating_stmt.excluded.rating,
                    "updated_at": rating_stmt.excluded.updated_at,
                },
            )
            rating_result = session.execute(rating_stmt)
            stats["ratings_upserted"] += affected_rows(rating_result, len(rating_payload))

        session.commit()

    return stats


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root(Path.cwd())
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser(description="Import Goodreads parquet datasets into PostgreSQL.")
    parser.add_argument(
        "--authors-path",
        type=Path,
        default=data_dir / "raw_goodreads_book_authors.parquet",
    )
    parser.add_argument(
        "--books-path",
        type=Path,
        default=data_dir / "3_goodreads_books_with_metrics.parquet",
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=data_dir / "3_goodreads_reviews_dedup_clean.parquet",
    )
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=data_dir / "3_goodreads_interactions_reduced.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-authors", action="store_true")
    parser.add_argument("--skip-books", action="store_true")
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument("--skip-interactions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = duckdb.connect(database=":memory:")
    session = SessionLocal()

    try:
        if not args.skip_authors:
            author_stats = import_authors(session, con, args.authors_path, args.batch_size, args.limit)
            print("authors:", author_stats)
        if not args.skip_books:
            book_stats = import_books(session, con, args.books_path, args.batch_size, args.limit)
            print("books:", book_stats)
        if not args.skip_reviews:
            review_stats = import_reviews(session, con, args.reviews_path, args.batch_size, args.limit)
            print("reviews:", review_stats)
        if not args.skip_interactions:
            interaction_stats = import_interactions(
                session,
                con,
                args.interactions_path,
                args.batch_size,
                args.limit,
            )
            print("interactions:", interaction_stats)
    finally:
        session.close()
        con.close()


if __name__ == "__main__":
    main()
