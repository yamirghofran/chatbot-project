import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Load Books & Authors from Parquet

    Full import pipeline — runs on an empty database.

    **Steps:**
    1. Configure parquet file paths below
    2. Ensure tables exist
    3. Preview datasets
    4. Import authors
    5. Import books (author links created automatically)
    6. Populate `similar_books` column
    7. Verify results
    """)
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## Configuration — set your parquet file paths here
    """)
    return


@app.cell
def _():
    from pathlib import Path

    # ── Edit these paths to point at your finalized parquet files ──────────────
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "data").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent.parent

    DATA_DIR = PROJECT_ROOT / "data"

    BOOKS_PARQUET = DATA_DIR / "3_goodreads_books_with_metrics.parquet"
    AUTHORS_PARQUET = DATA_DIR / "raw_goodreads_book_authors.parquet"
    # ───────────────────────────────────────────────────────────────────────────
    return AUTHORS_PARQUET, BOOKS_PARQUET


@app.cell
def _():
    import polars as pl
    import json
    from sqlalchemy import select, func, text

    from bookdb.db.base import Base
    from bookdb.db.session import SessionLocal, engine
    from bookdb.db.models import Book, Author, book_authors
    from bookdb.datasets.processor import parse_authors_field, import_authors, import_books

    return (
        Author,
        Base,
        Book,
        SessionLocal,
        book_authors,
        engine,
        func,
        import_authors,
        import_books,
        json,
        pl,
        select,
    )


@app.cell
def _(Base, engine, mo):
    import bookdb.db.models  # noqa: ensure all models are registered

    Base.metadata.create_all(engine)
    mo.md("## Tables ensured")
    return


@app.cell
def _(AUTHORS_PARQUET, BOOKS_PARQUET, mo, pl):
    books_df = pl.read_parquet(BOOKS_PARQUET)
    authors_df = pl.read_parquet(AUTHORS_PARQUET)

    mo.vstack([
        mo.md("## Dataset Preview"),
        mo.md(f"**Books:** {books_df.shape[0]:,} rows — columns: `{'`, `'.join(books_df.columns)}`"),
        mo.md(f"**Authors:** {authors_df.shape[0]:,} rows — columns: `{'`, `'.join(authors_df.columns)}`"),
        mo.md("### Books sample"),
        books_df.head(5),
        mo.md("### Authors sample"),
        authors_df.head(5),
    ])
    return (books_df,)


@app.cell
def _(AUTHORS_PARQUET, SessionLocal, import_authors, mo):
    _session = SessionLocal()
    mo.output.append(mo.md("## Importing authors..."))
    author_stats = import_authors(AUTHORS_PARQUET, session=_session)
    _session.commit()
    _session.close()

    mo.vstack([
        mo.md(f"- **Rows processed:** {author_stats['rows']:,}"),
        mo.md(f"- **Created:** {author_stats['authors_created']:,}"),
        mo.md(f"- **Skipped (duplicates):** {author_stats['authors_skipped']:,}"),
        mo.md(f"- **Errors:** {author_stats['errors']:,}"),
    ])
    return


@app.cell
def _(BOOKS_PARQUET, SessionLocal, import_books, mo):
    _session = SessionLocal()
    mo.output.append(mo.md("## Importing books (with author links)..."))
    book_stats = import_books(BOOKS_PARQUET, session=_session)
    _session.commit()
    _session.close()

    mo.vstack([
        mo.md(f"- **Rows processed:** {book_stats['rows']:,}"),
        mo.md(f"- **Books created:** {book_stats['books_created']:,}"),
        mo.md(f"- **Authors linked:** {book_stats['authors_linked']:,}"),
        mo.md(f"- **Skipped (duplicates):** {book_stats['books_skipped']:,}"),
        mo.md(f"- **Errors:** {book_stats['errors']:,}"),
    ])
    return


@app.cell
def _(Book, SessionLocal, books_df, json, mo, pl, select):
    # Populate similar_books column from parquet
    _session = SessionLocal()

    # Build external_id -> internal_id map
    _db_books = _session.execute(
        select(Book.id, Book.book_id).where(Book.book_id.isnot(None))
    ).all()
    _book_ext_to_int = {str(row.book_id): row.id for row in _db_books}

    similar_data = (
        books_df
        .select("book_id", "similar_books")
        .filter(pl.col("similar_books").is_not_null())
        .with_columns(pl.col("book_id").cast(pl.Utf8))
        .to_dicts()
    )

    sb_updated = 0
    sb_skipped = 0
    sb_errors = 0

    for _sb_row in similar_data:
        _ext_id = str(_sb_row["book_id"])
        _int_id = _book_ext_to_int.get(_ext_id)
        if _int_id is None:
            sb_skipped += 1
            continue
        try:
            similar = _sb_row["similar_books"]
            if isinstance(similar, list):
                similar_json = json.dumps([str(s) for s in similar])
            elif isinstance(similar, str):
                try:
                    parsed = json.loads(similar)
                    similar_json = json.dumps([str(s) for s in parsed])
                except json.JSONDecodeError:
                    similar_json = json.dumps([s.strip() for s in similar.split(",") if s.strip()])
            else:
                sb_errors += 1
                continue

            book = _session.get(Book, _int_id)
            if book:
                book.similar_books = similar_json
                sb_updated += 1
        except Exception:
            sb_errors += 1

    _session.commit()
    _session.close()

    mo.vstack([
        mo.md("## Similar Books Backfill"),
        mo.md(f"- **Updated:** {sb_updated:,}"),
        mo.md(f"- **Skipped (not in DB):** {sb_skipped:,}"),
        mo.md(f"- **Errors:** {sb_errors:,}"),
    ])
    return


@app.cell
def _(Author, Book, SessionLocal, book_authors, func, mo, select):
    _session = SessionLocal()

    total_books = _session.scalar(select(func.count()).select_from(Book))
    total_authors_count = _session.scalar(
        select(func.count()).select_from(Author)
    )
    total_assoc = _session.scalar(select(func.count()).select_from(book_authors))
    books_with_authors = _session.scalar(
        select(func.count(Book.id.distinct()))
        .select_from(Book)
        .join(book_authors, Book.id == book_authors.c.book_id)
    )
    books_with_similar = _session.scalar(
        select(func.count()).where(Book.similar_books.isnot(None))
    )

    _session.close()

    mo.vstack([
        mo.md("## Verification"),
        mo.md(f"- **Total books:** {total_books:,}"),
        mo.md(f"- **Total authors:** {total_authors_count:,}"),
        mo.md(f"- **Book-author associations:** {total_assoc:,}"),
        mo.md(f"- **Books with at least one author linked:** {books_with_authors:,}"),
        mo.md(f"- **Books with similar_books populated:** {books_with_similar:,}"),
    ])
    return


if __name__ == "__main__":
    app.run()
