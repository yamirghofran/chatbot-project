from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
from sqlalchemy import func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from bookdb.db.base import Base
from bookdb.db.models import Author, Book, book_authors
from bookdb.db.session import SessionLocal, engine


def resolve_project_root(start: Path) -> Path:
    current = start.resolve()
    if (current / "data").exists() and (current / "bookdb").exists():
        return current

    for parent in current.parents:
        if (parent / "data").exists() and (parent / "bookdb").exists():
            return parent

    return current


def parse_jsonish(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        value = raw.strip()
        if not value or value.lower() == "null":
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return raw


def parquet_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> set[str]:
    rel = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(parquet_path)])
    return {str(desc[0]) for desc in rel.description}


def extract_author_ids(raw: Any) -> list[str]:
    parsed = parse_jsonish(raw)

    if parsed is None:
        return []

    ids: list[str] = []

    if isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, dict):
                aid = entry.get("author_id")
                if aid is not None:
                    aid_s = str(aid).strip()
                    if aid_s:
                        ids.append(aid_s)
            elif entry is not None:
                aid_s = str(entry).strip()
                if aid_s:
                    ids.append(aid_s)
    elif isinstance(parsed, str):
        for part in parsed.split(","):
            part = part.strip()
            if part:
                ids.append(part)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(ids))


def normalize_similar_books(raw: Any) -> str | None:
    parsed = parse_jsonish(raw)

    if parsed is None:
        return None

    if isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, str):
        values = [v.strip() for v in parsed.split(",")]
    else:
        return None

    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return json.dumps(cleaned)


def ensure_tables() -> None:
    import bookdb.db.models  # noqa: F401 - ensures models are registered

    Base.metadata.create_all(engine)


def preview_datasets(books_parquet: Path, authors_parquet: Path, preview_rows: int) -> None:
    con = duckdb.connect(database=":memory:")
    try:
        books_cols_rel = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(books_parquet)]
        )
        books_columns = [d[0] for d in books_cols_rel.description]

        authors_cols_rel = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(authors_parquet)]
        )
        authors_columns = [d[0] for d in authors_cols_rel.description]

        books_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(books_parquet)]
        ).fetchone()[0]
        authors_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(authors_parquet)]
        ).fetchone()[0]

        books_sample = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?", [str(books_parquet), preview_rows]
        ).fetchall()
        authors_sample = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?", [str(authors_parquet), preview_rows]
        ).fetchall()

        print("## Dataset Preview")
        print(f"Books:   {books_count:,} rows")
        print(f"Columns: {', '.join(books_columns)}")
        print(f"Authors: {authors_count:,} rows")
        print(f"Columns: {', '.join(authors_columns)}")

        if preview_rows > 0:
            print("\nBooks sample:")
            for row in books_sample:
                print(row)
            print("\nAuthors sample:")
            for row in authors_sample:
                print(row)
            print()
    finally:
        con.close()


def import_authors_fast(
    authors_parquet: Path,
    read_batch_size: int,
    write_batch_size: int,
) -> dict[str, int]:
    session = SessionLocal()
    con = duckdb.connect(database=":memory:")

    stats = {
        "rows": 0,
        "authors_created": 0,
        "authors_skipped": 0,
        "errors": 0,
    }

    try:
        existing_names = {
            row[0]
            for row in session.execute(select(func.lower(Author.name))).all()
            if row[0] is not None
        }
        existing_external_ids = {
            row[0]
            for row in session.execute(
                select(Author.external_id).where(Author.external_id.is_not(None))
            ).all()
            if row[0] is not None
        }

        stats["rows"] = int(
            con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(authors_parquet)]
            ).fetchone()[0]
        )

        cursor = con.execute(
            """
            SELECT
                TRIM(CAST(name AS VARCHAR)) AS name,
                CASE WHEN author_id IS NULL THEN NULL ELSE CAST(author_id AS VARCHAR) END AS external_id
            FROM read_parquet(?)
            """,
            [str(authors_parquet)],
        )

        seen_names = set(existing_names)
        seen_external_ids = set(existing_external_ids)
        pending: list[dict[str, str | None]] = []

        while True:
            rows = cursor.fetchmany(read_batch_size)
            if not rows:
                break

            for name, external_id in rows:
                if not name:
                    stats["errors"] += 1
                    continue

                name_key = name.lower()
                ext_id = external_id.strip() if external_id else None

                if name_key in seen_names:
                    stats["authors_skipped"] += 1
                    continue
                if ext_id is not None and ext_id in seen_external_ids:
                    stats["authors_skipped"] += 1
                    continue

                pending.append({"name": name, "external_id": ext_id})
                seen_names.add(name_key)
                if ext_id is not None:
                    seen_external_ids.add(ext_id)

                if len(pending) >= write_batch_size:
                    session.execute(insert(Author), pending)
                    stats["authors_created"] += len(pending)
                    pending.clear()

        if pending:
            session.execute(insert(Author), pending)
            stats["authors_created"] += len(pending)

        session.commit()
        return stats
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        con.close()


def import_books_fast(
    books_parquet: Path,
    read_batch_size: int,
    write_batch_size: int,
) -> dict[str, int]:
    session = SessionLocal()
    con = duckdb.connect(database=":memory:")

    stats = {
        "rows": 0,
        "books_created": 0,
        "books_skipped": 0,
        "authors_linked": 0,
        "errors": 0,
    }

    try:
        columns = parquet_columns(con, books_parquet)

        title_parts: list[str] = []
        if "original_title" in columns:
            title_parts.append("NULLIF(TRIM(CAST(original_title AS VARCHAR)), '')")
        if "title" in columns:
            title_parts.append("NULLIF(TRIM(CAST(title AS VARCHAR)), '')")
        if not title_parts:
            raise ValueError(
                "Books parquet must include at least one of: 'original_title' or 'title'."
            )

        title_expr = (
            f"COALESCE({', '.join(title_parts)})"
            if len(title_parts) > 1
            else title_parts[0]
        )
        publication_year_expr = (
            "TRY_CAST(publication_year AS INTEGER)"
            if "publication_year" in columns
            else "CAST(NULL AS INTEGER)"
        )
        description_expr = (
            "NULLIF(TRIM(CAST(description AS VARCHAR)), '')"
            if "description" in columns
            else "CAST(NULL AS VARCHAR)"
        )
        image_url_expr = (
            "NULLIF(TRIM(CAST(image_url AS VARCHAR)), '')"
            if "image_url" in columns
            else "CAST(NULL AS VARCHAR)"
        )
        book_id_expr = (
            "CASE WHEN book_id IS NULL THEN NULL ELSE CAST(book_id AS VARCHAR) END"
            if "book_id" in columns
            else "CAST(NULL AS VARCHAR)"
        )
        authors_expr = (
            "CASE WHEN authors IS NULL THEN NULL ELSE to_json(authors) END"
            if "authors" in columns
            else "CAST(NULL AS VARCHAR)"
        )

        author_map = {
            row.external_id: row.id
            for row in session.execute(
                select(Author.id, Author.external_id).where(Author.external_id.is_not(None))
            ).all()
        }
        existing_book_ids = {
            row[0]
            for row in session.execute(
                select(Book.book_id).where(Book.book_id.is_not(None))
            ).all()
            if row[0] is not None
        }
        existing_title_keys = {
            row[0]
            for row in session.execute(select(func.lower(Book.title))).all()
            if row[0] is not None
        }

        seen_book_ids = set(existing_book_ids)
        seen_title_keys = set(existing_title_keys)

        stats["rows"] = int(
            con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(books_parquet)]).fetchone()[0]
        )

        cursor = con.execute(
            f"""
            SELECT
                {title_expr} AS title,
                {publication_year_expr} AS publication_year,
                {description_expr} AS description,
                {image_url_expr} AS image_url,
                {book_id_expr} AS book_id,
                {authors_expr} AS authors_json
            FROM read_parquet(?)
            """,
            [str(books_parquet)],
        )

        pending_books: list[dict[str, Any]] = []
        pending_links: list[tuple[str | None, str, list[str]]] = []

        def flush_batch() -> None:
            if not pending_books:
                return

            session.execute(insert(Book), pending_books)

            ext_ids = {item[0] for item in pending_links if item[0] is not None}
            title_keys = {item[1] for item in pending_links if item[1]}

            books_by_ext: dict[str, int] = {}
            books_by_title: dict[str, int] = {}

            if ext_ids:
                for row in session.execute(
                    select(Book.id, Book.book_id).where(Book.book_id.in_(list(ext_ids)))
                ).all():
                    books_by_ext[row.book_id] = row.id

            if title_keys:
                for row in session.execute(
                    select(Book.id, Book.title).where(func.lower(Book.title).in_(list(title_keys)))
                ).all():
                    books_by_title[row.title.lower()] = row.id

            assoc_rows: list[dict[str, int]] = []
            seen_assoc: set[tuple[int, int]] = set()

            for ext_book_id, title_key, author_ext_ids in pending_links:
                book_id = books_by_ext.get(ext_book_id) if ext_book_id else None
                if book_id is None:
                    book_id = books_by_title.get(title_key)
                if book_id is None:
                    continue

                for author_ext_id in author_ext_ids:
                    author_id = author_map.get(author_ext_id)
                    if author_id is None:
                        continue
                    pair = (book_id, author_id)
                    if pair in seen_assoc:
                        continue
                    seen_assoc.add(pair)
                    assoc_rows.append({"book_id": book_id, "author_id": author_id})

            if assoc_rows:
                result = session.execute(
                    pg_insert(book_authors)
                    .values(assoc_rows)
                    .on_conflict_do_nothing(index_elements=["book_id", "author_id"])
                )
                if result.rowcount is not None and result.rowcount >= 0:
                    stats["authors_linked"] += result.rowcount
                else:
                    stats["authors_linked"] += len(assoc_rows)

            stats["books_created"] += len(pending_books)
            pending_books.clear()
            pending_links.clear()

        while True:
            rows = cursor.fetchmany(read_batch_size)
            if not rows:
                break

            for title, publication_year, description, image_url, book_id, authors_json in rows:
                if not title:
                    stats["errors"] += 1
                    continue

                title_key = title.lower()
                ext_book_id = book_id.strip() if book_id else None

                if title_key in seen_title_keys:
                    stats["books_skipped"] += 1
                    continue
                if ext_book_id is not None and ext_book_id in seen_book_ids:
                    stats["books_skipped"] += 1
                    continue

                pending_books.append(
                    {
                        "title": title,
                        "publication_year": publication_year,
                        "description": description,
                        "image_url": image_url,
                        "book_id": ext_book_id,
                    }
                )
                pending_links.append((ext_book_id, title_key, extract_author_ids(authors_json)))

                seen_title_keys.add(title_key)
                if ext_book_id is not None:
                    seen_book_ids.add(ext_book_id)

                if len(pending_books) >= write_batch_size:
                    flush_batch()

        flush_batch()
        session.commit()
        return stats
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        con.close()


def backfill_similar_books_fast(
    books_parquet: Path,
    read_batch_size: int,
    write_batch_size: int,
) -> dict[str, int]:
    session = SessionLocal()
    con = duckdb.connect(database=":memory:")

    stats = {
        "updated": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        columns = parquet_columns(con, books_parquet)
        if "book_id" not in columns or "similar_books" not in columns:
            print(
                "Skipping similar_books backfill: missing required parquet columns "
                "('book_id' and/or 'similar_books')."
            )
            return stats

        book_ext_to_int = {
            str(row.book_id): row.id
            for row in session.execute(
                select(Book.id, Book.book_id).where(Book.book_id.is_not(None))
            ).all()
        }

        cursor = con.execute(
            """
            SELECT
                CAST(book_id AS VARCHAR) AS book_id,
                to_json(similar_books) AS similar_books_json
            FROM read_parquet(?)
            WHERE similar_books IS NOT NULL
              AND book_id IS NOT NULL
            """,
            [str(books_parquet)],
        )

        pending_updates: list[dict[str, Any]] = []

        while True:
            rows = cursor.fetchmany(read_batch_size)
            if not rows:
                break

            for ext_id, similar_raw in rows:
                int_id = book_ext_to_int.get(str(ext_id))
                if int_id is None:
                    stats["skipped"] += 1
                    continue

                similar_json = normalize_similar_books(similar_raw)
                if similar_json is None:
                    stats["errors"] += 1
                    continue

                pending_updates.append({"id": int_id, "similar_books": similar_json})

                if len(pending_updates) >= write_batch_size:
                    session.execute(
                        text("UPDATE books SET similar_books = :similar_books WHERE id = :id"),
                        pending_updates,
                    )
                    stats["updated"] += len(pending_updates)
                    pending_updates.clear()

        if pending_updates:
            session.execute(
                text("UPDATE books SET similar_books = :similar_books WHERE id = :id"),
                pending_updates,
            )
            stats["updated"] += len(pending_updates)

        session.commit()
        return stats
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        con.close()


def verify_counts() -> dict[str, int]:
    session = SessionLocal()
    try:
        return {
            "total_books": session.scalar(select(func.count()).select_from(Book)) or 0,
            "total_authors": session.scalar(select(func.count()).select_from(Author)) or 0,
            "book_author_associations": session.scalar(
                select(func.count()).select_from(book_authors)
            )
            or 0,
            "books_with_authors": session.scalar(
                select(func.count(Book.id.distinct()))
                .select_from(Book)
                .join(book_authors, Book.id == book_authors.c.book_id)
            )
            or 0,
            "books_with_similar_books": session.scalar(
                select(func.count()).where(Book.similar_books.is_not(None))
            )
            or 0,
        }
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast backpopulate pipeline for authors, books, relations, and similar_books."
    )
    default_root = resolve_project_root(Path.cwd())
    default_data = default_root / "data"

    parser.add_argument(
        "--books-parquet",
        type=Path,
        default=default_data / "3_goodreads_books_with_metrics.parquet",
        help="Path to books parquet file.",
    )
    parser.add_argument(
        "--authors-parquet",
        type=Path,
        default=default_data / "raw_goodreads_book_authors.parquet",
        help="Path to authors parquet file.",
    )
    parser.add_argument(
        "--authors-read-batch-size",
        type=int,
        default=50_000,
        help="DuckDB fetch batch size for authors import.",
    )
    parser.add_argument(
        "--books-read-batch-size",
        type=int,
        default=20_000,
        help="DuckDB fetch batch size for books import.",
    )
    parser.add_argument(
        "--similar-read-batch-size",
        type=int,
        default=50_000,
        help="DuckDB fetch batch size for similar_books backfill.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10_000,
        help="DB executemany batch size.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of rows to show in dataset preview.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip dataset preview output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    books_parquet = args.books_parquet.expanduser().resolve()
    authors_parquet = args.authors_parquet.expanduser().resolve()

    if not books_parquet.exists():
        raise FileNotFoundError(f"Books parquet not found: {books_parquet}")
    if not authors_parquet.exists():
        raise FileNotFoundError(f"Authors parquet not found: {authors_parquet}")

    print("== Ensuring tables ==")
    ensure_tables()

    if not args.skip_preview:
        print("\n== Preview datasets ==")
        preview_datasets(books_parquet, authors_parquet, args.preview_rows)

    print("\n== Import authors ==")
    author_stats = import_authors_fast(
        authors_parquet=authors_parquet,
        read_batch_size=args.authors_read_batch_size,
        write_batch_size=args.write_batch_size,
    )
    print(author_stats)

    print("\n== Import books + author links ==")
    book_stats = import_books_fast(
        books_parquet=books_parquet,
        read_batch_size=args.books_read_batch_size,
        write_batch_size=args.write_batch_size,
    )
    print(book_stats)

    print("\n== Backfill similar_books ==")
    similar_stats = backfill_similar_books_fast(
        books_parquet=books_parquet,
        read_batch_size=args.similar_read_batch_size,
        write_batch_size=args.write_batch_size,
    )
    print(similar_stats)

    print("\n== Verification ==")
    verification = verify_counts()
    print(verification)


if __name__ == "__main__":
    main()
