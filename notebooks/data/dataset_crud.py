import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Dataset CRUD Testing

    Testing the CRUD operations and dataset import functionality.

    **Prerequisites:** Run `make setup` to start PostgreSQL and run migrations.
    """)
    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import pyarrow.parquet as pq
    from pathlib import Path

    from bookdb.db.session import SessionLocal
    from bookdb.datasets.crud import BookCRUD, AuthorCRUD, UserCRUD, BookListCRUD
    from bookdb.datasets.processor import import_dataset, preview_dataset, read_file, import_authors, import_books

    session = SessionLocal()
    return (
        AuthorCRUD,
        BookCRUD,
        BookListCRUD,
        Path,
        SessionLocal,
        UserCRUD,
        import_authors,
        import_books,
        import_dataset,
        pl,
        pq,
        preview_dataset,
        session,
    )


@app.cell
def _(Path, pq):
    # Preview new dataset
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"

    books_file_path = DATA_DIR / "raw_goodreads_books.parquet"

    pq_file = pq.ParquetFile(books_file_path)

    table = pq_file.read_row_group(0).slice(0, 5)
    df = table.to_pandas()
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Import GoodReads data to DBs
    """)
    return


@app.cell
def _(Path, preview_dataset):
    # Preview authors dataset
    PROJECT_ROOT = Path.cwd().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    authors_file_path = DATA_DIR / "raw_goodreads_book_authors.parquet"

    preview = preview_dataset(authors_file_path)
    for item in preview:
        print(item)
        print(f"{item['name']}")
    return (authors_file_path,)


@app.cell
def _(authors_file_path, import_authors):
    # Import authors dataset
    author_stats = import_authors(authors_file_path)
    print(author_stats)
    return


@app.cell
def _(Path, preview_dataset):
    # Preview books dataset
    PROJECT_ROOT = Path.cwd().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    book_file_path = DATA_DIR / "raw_goodreads_book_works.parquet"

    preview = preview_dataset(book_file_path)
    for item in preview:
        print(item)
        print(f"{item['original_title']}")
    return (book_file_path,)


@app.cell
def _(book_file_path, import_books):
    # Import books dataset
    book_stats = import_books(book_file_path)
    print(book_stats)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test CRUD Operations
    """)
    return


@app.cell
def _(AuthorCRUD, BookCRUD, SessionLocal):
    session = SessionLocal()

    # Get an author
    author = AuthorCRUD.get_or_create(session, "Ronald J. Fields")
    print(f"Author: {author.name} (ID: {author.id})")

    # Create a book with author
    book = BookCRUD.create_with_authors(
        session,
        author_names=["Ronald J. Fields"],
        title="W.C. Fields by Himself",
        pages_number=510,
        publish_year=1973,
    )
    print(f"Book: {book.title} by {[a.name for a in book.authors]}")

    session.commit()
    return book, session


@app.cell
def _(BookCRUD, session):
    # Search for books
    results = BookCRUD.search_by_title(session, "W.C. Fields")
    print(f"Found {len(results)} books:")
    for b in results:
        print(f"  - {b.title} ({b.publish_year})")
    return


@app.cell
def _(BookListCRUD, UserCRUD, book, session):
    # Create user and book list
    user = UserCRUD.get_or_create(session, "reader@example.com", "Book Reader")
    book_list = BookListCRUD.create(session, user.id, "Classics")
    BookListCRUD.add_book(session, book_list.id, book.id)

    print(f"User: {user.name}")
    print(f"List '{book_list.name}' has {len(book_list.books)} book(s)")

    session.commit()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test csv support
    """)
    return


@app.cell
def _(Path, pl):
    # Create a sample CSV
    sample = pl.DataFrame({
        "title": ["The Great Gatsby", "1984", "Pride and Prejudice", "Brave New World"],
        "authors": ["F. Scott Fitzgerald", "George Orwell", "Jane Austen", "Aldous Huxley"],
        "pages_number": [180, 328, 279, 268],
        "publisher_name": ["Scribner", "Secker & Warburg", "T. Egerton", "Chatto & Windus"],
        "publish_year": [1925, 1949, 1813, 1932],
    })

    sample_path = Path("/tmp/sample_books.csv")
    sample.write_csv(sample_path)
    print(f"Created: {sample_path}")
    return (sample_path,)


@app.cell
def _(preview_dataset, sample_path):
    # Preview the dataset before importing
    preview = preview_dataset(sample_path, n=3)
    for item in preview:
        print(f"Book: {item['book']['title']}")
        print(f"  Authors: {item['authors']}")
        print(f"  Year: {item['book']['publish_year']}\n")
    return


@app.cell
def _(import_dataset, sample_path):
    # Import the dataset
    stats = import_dataset(sample_path)
    print("Import stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return


@app.cell
def _(BookCRUD, SessionLocal):
    # Verify the imported books
    session = SessionLocal()

    gatsby = BookCRUD.get_by_title(session, "The Great Gatsby")
    if gatsby:
        b = gatsby[0]
        print(f"{b.title}")
        print(f"  Authors: {[a.name for a in b.authors]}")
        print(f"  Published: {b.publish_year}")
        print(f"  Pages: {b.pages_number}")
    return (session,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Cleanup
    """)
    return


@app.cell
def _(session):
    session.close()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
