import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Back-populate Relations from Parquet Data

    This notebook is **self-contained** — it works on an empty database.

    Steps:
    1. Ensure all tables exist (creates them if needed)
    2. Import authors and books from parquet if tables are empty
    3. Populate **book_authors** junction table
    4. Populate **similar_books** column on books

    **Data sources:**
    - `data/raw_goodreads_books.parquet` — book metadata including author_id and similar_books
    - `data/raw_goodreads_book_authors.parquet` — author metadata including author_id
    """)
    return (mo,)


@app.cell
def _():
    import polars as pl
    import json
    from pathlib import Path
    from sqlalchemy import select, func

    from bookdb.db.base import Base
    from bookdb.db.session import SessionLocal, engine
    from bookdb.db.models import Book, Author, book_authors, list_books
    from bookdb.datasets.processor import parse_authors_field, import_authors, import_books

    return (
        Author,
        Base,
        Book,
        Path,
        SessionLocal,
        book_authors,
        engine,
        func,
        import_authors,
        import_books,
        json,
        list_books,
        parse_authors_field,
        pl,
        select,
    )


@app.cell
def _(Base, engine, mo):
    # Ensure all tables exist (safe no-op if they already do)
    import bookdb.db.models  # noqa: ensure all models are registered

    Base.metadata.create_all(engine)
    mo.md("## Tables ensured")
    return


@app.cell
def _(Path, mo, pl):
    # Resolve project root (handles running from notebooks/data/ or project root)
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "data").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent.parent

    DATA_DIR = PROJECT_ROOT / "data"

    books_df = pl.read_parquet(DATA_DIR / "raw_goodreads_books.parquet")
    authors_df = pl.read_parquet(DATA_DIR / "raw_goodreads_book_authors.parquet")

    mo.vstack([
        mo.md("## Dataset Overview"),
        mo.md(f"**Books dataset:** {books_df.shape[0]:,} rows, {books_df.shape[1]} columns"),
        mo.md(f"**Authors dataset:** {authors_df.shape[0]:,} rows, {authors_df.shape[1]} columns"),
        mo.md("**Books columns:** " + ", ".join(books_df.columns)),
        mo.md("**Authors columns:** " + ", ".join(authors_df.columns)),
    ])
    return DATA_DIR, books_df


@app.cell
def _(books_df, mo):
    # Preview the key columns used for linking
    preview = books_df.select("book_id", "title", "authors", "similar_books").head(10)
    mo.vstack([
        mo.md("## Preview: Books with authors and similar_books"),
        preview,
    ])
    return


@app.cell
def _(
    Author,
    Book,
    DATA_DIR,
    SessionLocal,
    book_authors,
    func,
    import_authors,
    import_books,
    list_books,
    mo,
    select,
):
    from sqlalchemy import delete

    session = SessionLocal()

    # Import authors if the table is empty
    author_count = session.scalar(select(func.count()).select_from(Author))
    if author_count == 0:
        mo.output.append(mo.md("**Importing authors from parquet...**"))
        author_stats = import_authors(
            DATA_DIR / "raw_goodreads_book_authors.parquet", session=session
        )
        session.commit()
        mo.output.append(mo.md(f"Authors imported: {author_stats}"))
    else:
        mo.output.append(mo.md(f"**Authors already in DB:** {author_count:,}"))

    # Import books — clear and re-import if existing books lack book_id
    book_count = session.scalar(select(func.count()).select_from(Book))
    books_with_ext_id = session.scalar(
        select(func.count()).where(Book.book_id.isnot(None))
    )

    if book_count > 0 and books_with_ext_id == 0:
        mo.output.append(mo.md(
            f"**{book_count:,} books found but none have book_id set. "
            f"Clearing and re-importing...**"
        ))
        session.execute(delete(book_authors))
        session.execute(delete(list_books))
        session.execute(delete(Book))
        session.commit()
        book_count = 0

    if book_count == 0:
        mo.output.append(mo.md("**Importing books from parquet...**"))
        book_stats = import_books(
            DATA_DIR / "raw_goodreads_books.parquet", session=session
        )
        session.commit()
        mo.output.append(mo.md(f"Books imported: {book_stats}"))
    else:
        mo.output.append(mo.md(f"**Books already in DB:** {book_count:,}"))
    return (session,)


@app.cell
def _(Author, Book, mo, select, session):
    # Build lookup dicts: external_id -> internal_id
    db_books = session.execute(
        select(Book.id, Book.book_id)
        .where(Book.book_id.isnot(None))
    ).all()

    db_authors = session.execute(
        select(Author.id, Author.external_id)
        .where(Author.external_id.isnot(None))
    ).all()

    book_ext_to_int = {str(row.book_id): row.id for row in db_books}
    author_ext_to_int = {str(row.external_id): row.id for row in db_authors}

    mo.vstack([
        mo.md("## Database State"),
        mo.md(f"**Books in DB with external IDs:** {len(book_ext_to_int):,}"),
        mo.md(f"**Authors in DB with external IDs:** {len(author_ext_to_int):,}"),
    ])
    return author_ext_to_int, book_ext_to_int


@app.cell
def _(
    author_ext_to_int,
    book_authors,
    book_ext_to_int,
    books_df,
    mo,
    parse_authors_field,
    pl,
    select,
    session,
):
    # Extract book_id -> author_id pairs from the parquet authors struct column
    book_rows = (
        books_df
        .select("book_id", "authors")
        .filter(pl.col("book_id").is_not_null() & pl.col("authors").is_not_null())
        .with_columns(pl.col("book_id").cast(pl.Utf8))
        .to_dicts()
    )

    # Flatten: one row per (book_id, author_id) pair
    book_author_pairs = []
    for row in book_rows:
        ext_book_id = str(row["book_id"])
        for ext_author_id in parse_authors_field(row["authors"]):
            book_author_pairs.append({"book_id": ext_book_id, "author_id": ext_author_id})

    # Check existing associations to avoid duplicates
    existing_assoc = set()
    existing_rows = session.execute(
        select(book_authors.c.book_id, book_authors.c.author_id)
    ).all()
    for row in existing_rows:
        existing_assoc.add((row.book_id, row.author_id))

    # Build insert batch
    to_insert = []
    ba_matched = 0
    ba_skipped_no_book = 0
    ba_skipped_no_author = 0
    ba_skipped_exists = 0

    for pair in book_author_pairs:
        ext_book_id = pair["book_id"]
        ext_author_id = pair["author_id"]

        int_book_id = book_ext_to_int.get(ext_book_id)
        int_author_id = author_ext_to_int.get(ext_author_id)

        if int_book_id is None:
            ba_skipped_no_book += 1
            continue
        if int_author_id is None:
            ba_skipped_no_author += 1
            continue

        if (int_book_id, int_author_id) in existing_assoc:
            ba_skipped_exists += 1
            continue

        to_insert.append({"book_id": int_book_id, "author_id": int_author_id})
        existing_assoc.add((int_book_id, int_author_id))
        ba_matched += 1

    # Batch insert
    BATCH_SIZE = 5000
    for i in range(0, len(to_insert), BATCH_SIZE):
        batch = to_insert[i : i + BATCH_SIZE]
        session.execute(book_authors.insert(), batch)
        session.commit()

    mo.vstack([
        mo.md("## Book-Author Junction Table Results"),
        mo.md(f"- **Matched and inserted:** {ba_matched:,}"),
        mo.md(f"- **Skipped (book not in DB):** {ba_skipped_no_book:,}"),
        mo.md(f"- **Skipped (author not in DB):** {ba_skipped_no_author:,}"),
        mo.md(f"- **Skipped (already exists):** {ba_skipped_exists:,}"),
    ])
    return


@app.cell
def _(Book, book_ext_to_int, books_df, json, mo, pl, session):
    # Extract similar_books from parquet
    similar_data = (
        books_df
        .select("book_id", "similar_books")
        .filter(pl.col("similar_books").is_not_null())
        .with_columns(pl.col("book_id").cast(pl.Utf8))
        .to_dicts()
    )

    sb_updated = 0
    sb_skipped_no_book = 0
    sb_errors = 0

    SB_BATCH_SIZE = 1000
    for e in range(0, len(similar_data), SB_BATCH_SIZE):
        sb_batch = similar_data[e : e + SB_BATCH_SIZE]

        for sb_row in sb_batch:
            exte_book_id = str(sb_row["book_id"])
            inte_book_id = book_ext_to_int.get(exte_book_id)

            if inte_book_id is None:
                sb_skipped_no_book += 1
                continue

            try:
                similar = sb_row["similar_books"]
                # Handle different formats
                if isinstance(similar, list):
                    similar_json = json.dumps([str(s) for s in similar])
                elif isinstance(similar, str):
                    try:
                        parsed = json.loads(similar)
                        similar_json = json.dumps([str(s) for s in parsed])
                    except json.JSONDecodeError:
                        similar_json = json.dumps(
                            [s.strip() for s in similar.split(",") if s.strip()]
                        )
                else:
                    sb_errors += 1
                    continue

                book = session.get(Book, inte_book_id)
                if book:
                    book.similar_books = similar_json
                    sb_updated += 1

            except Exception:
                sb_errors += 1
                continue

        session.commit()

    mo.vstack([
        mo.md("## Similar Books Backfill Results"),
        mo.md(f"- **Updated:** {sb_updated:,}"),
        mo.md(f"- **Skipped (book not in DB):** {sb_skipped_no_book:,}"),
        mo.md(f"- **Errors:** {sb_errors:,}"),
    ])
    return


@app.cell
def _(Book, book_authors, func, mo, select, session):
    # Verify results
    total_associations = session.scalar(
        select(func.count()).select_from(book_authors)
    )

    books_with_similar = session.scalar(
        select(func.count()).where(Book.similar_books.isnot(None))
    )

    books_with_authors = session.scalar(
        select(func.count(Book.id.distinct()))
        .select_from(Book)
        .join(book_authors, Book.id == book_authors.c.book_id)
    )

    mo.vstack([
        mo.md("## Verification"),
        mo.md(f"- **Total book-author associations:** {total_associations:,}"),
        mo.md(f"- **Books with at least one author linked:** {books_with_authors:,}"),
        mo.md(f"- **Books with similar_books populated:** {books_with_similar:,}"),
    ])
    return


@app.cell
def _(session):
    session.close()
    return


if __name__ == "__main__":
    app.run()
