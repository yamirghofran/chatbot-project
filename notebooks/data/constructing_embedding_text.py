import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import polars as pl

    return mo, os, pl


@app.cell
def _(os):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    raw_books_path = os.path.join(project_root, "data", "raw_goodreads_books.parquet")
    raw_authors_path = os.path.join(
        project_root, "data", "raw_goodreads_book_authors.parquet"
    )
    output_path = os.path.join(project_root, "data", "books_embedding_texts.parquet")
    return output_path, raw_authors_path, raw_books_path


@app.cell
def _(pl):
    # Canonical genre labels and robust shelf-name regexes.
    GENRE_PATTERNS = [
        ("science fiction", r"\b(?:science[\s_-]*fiction|sci[\s_-]?fi|scifi|sf)\b"),
        ("historical fiction", r"\bhistorical[\s_-]*fiction\b"),
        ("literary fiction", r"\bliterary[\s_-]*fiction\b"),
        ("young adult", r"\b(?:young[\s_-]*adult|ya)\b"),
        ("middle grade", r"\bmiddle[\s_-]*grade\b"),
        ("graphic novel", r"\bgraphic[\s_-]*novel\b"),
        ("self-help", r"\bself[\s_-]*help\b"),
        ("true crime", r"\btrue[\s_-]*crime\b"),
        ("urban fantasy", r"\burban[\s_-]*fantasy\b"),
        ("chick lit", r"\bchick[\s_-]*lit\b"),
        ("fantasy", r"\bfantasy\b"),
        ("romance", r"\bromance\b"),
        ("mystery", r"\bmystery\b"),
        ("thriller", r"\bthriller\b"),
        ("horror", r"\bhorror\b"),
        ("dystopian", r"\bdystopian\b"),
        ("paranormal", r"\bparanormal\b"),
        ("adventure", r"\badventure\b"),
        ("crime", r"\bcrime\b"),
        ("fiction", r"\bfiction\b"),
        ("nonfiction", r"\bnon[\s_-]?fiction\b"),
        ("biography", r"\bbiograph(?:y|ies)\b"),
        ("memoir", r"\bmemoir\b"),
        ("poetry", r"\bpoetry\b"),
        ("classics", r"\bclassics?\b"),
        ("contemporary", r"\bcontemporary\b"),
        ("children", r"\bchildren(?:s)?\b"),
        ("comics", r"\bcomics?\b"),
        ("manga", r"\bmanga\b"),
        ("history", r"\bhistory\b"),
        ("philosophy", r"\bphilosophy\b"),
        ("business", r"\bbusiness\b"),
        ("science", r"\bscience\b"),
    ]

    IGNORE_SHELF_REGEX = (
        r"(?:^|\b)(?:to[\s_-]*read|currently[\s_-]*reading|owned|my[\s_-]*books?|"
        r"favorites?|favourites?|wishlist|wish[\s_-]*list|library|kindle|e[\s_-]?book|"
        r"audiobooks?|book[\s_-]*club|did[\s_-]*not[\s_-]*finish|dnf|series|default)"
        r"(?:\b|$)"
    )

    def clean_text(expr: pl.Expr) -> pl.Expr:
        return (
            expr.fill_null("")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def extract_genres_expr() -> pl.Expr:
        normalized_shelves = (
            pl.col("popular_shelves")
            .list.eval(
                pl.element()
                .struct.field("name")
                .fill_null("")
                .str.to_lowercase()
                .str.replace_all(r"[_/]", " ")
                .str.replace_all(r"[^a-z0-9+\-\s]", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
            .list.eval(
                pl.when(
                    (pl.element().str.len_chars() > 0)
                    & ~pl.element().str.contains(IGNORE_SHELF_REGEX)
                )
                .then(pl.element())
                .otherwise(None)
            )
            .list.drop_nulls()
        )

        shelf_text = normalized_shelves.list.join(" | ")

        return (
            pl.concat_list(
                [
                    pl.when(shelf_text.str.contains(pattern))
                    .then(pl.lit(label))
                    .otherwise(None)
                    for label, pattern in GENRE_PATTERNS
                ]
            )
            .list.drop_nulls()
            .list.unique()
            .list.sort()
        )

    return clean_text, extract_genres_expr


@app.cell
def _(pl, raw_authors_path, raw_books_path):
    books_lf = pl.scan_parquet(raw_books_path).select(
        [
            "book_id",
            "title",
            "title_without_series",
            "description",
            "popular_shelves",
            "authors",
        ]
    )
    authors_lf = (
        pl.scan_parquet(raw_authors_path)
        .select("author_id", pl.col("name").alias("author_name"))
        .unique(subset=["author_id"], keep="first")
    )
    return authors_lf, books_lf


@app.cell
def _(authors_lf, books_lf, pl):
    # Resolve each book's author ids -> ordered author names.
    book_authors_lf = (
        books_lf.select("book_id", "authors")
        .with_columns(
            author_ids=pl.col("authors").list.eval(
                pl.element().struct.field("author_id")
            ),
            author_positions=pl.int_ranges(pl.lit(0), pl.col("authors").list.len()),
        )
        .select("book_id", "author_ids", "author_positions")
        .explode("author_ids", "author_positions")
        .rename({"author_ids": "author_id", "author_positions": "author_position"})
        .filter(
            pl.col("author_id").is_not_null() & (pl.col("author_id").str.len_chars() > 0)
        )
        .join(authors_lf, on="author_id", how="left")
        .group_by("book_id")
        .agg(pl.col("author_name").sort_by("author_position").drop_nulls().alias("author_names"))
        .with_columns(pl.col("author_names").list.join(", ").fill_null("").alias("authors_text"))
        .select("book_id", "authors_text")
    )
    return (book_authors_lf,)


@app.cell
def _(book_authors_lf, books_lf, clean_text, extract_genres_expr, pl):
    books_text_lf = (
        books_lf.select(
            "book_id",
            "title",
            "title_without_series",
            "description",
            "popular_shelves",
        )
        .with_columns(
            title_without_series_text=clean_text(pl.col("title_without_series")),
            title_fallback_text=clean_text(pl.col("title")),
        )
        .with_columns(
            title_text=pl.when(pl.col("title_without_series_text").str.len_chars() > 0)
            .then(pl.col("title_without_series_text"))
            .otherwise(pl.col("title_fallback_text")),
            description_text=clean_text(pl.col("description")),
            genre_list=extract_genres_expr(),
        )
        .with_columns(pl.col("genre_list").list.join(", ").fill_null("").alias("genres_text"))
        .select("book_id", "title_text", "description_text", "genres_text")
    )

    embedding_lf = (
        books_text_lf.join(book_authors_lf, on="book_id", how="left")
        .with_columns(pl.col("authors_text").fill_null(""))
        .with_columns(
            pl.format(
                "TITLE: {}\nGENRES: {}\nAUTHORS: {}\nDESCRIPTION: {}",
                pl.col("title_text"),
                pl.col("genres_text"),
                pl.col("authors_text"),
                pl.col("description_text"),
            ).alias("book_embedding_text")
        )
        .select("book_id", "book_embedding_text")
    )
    return (embedding_lf,)


@app.cell
def _(embedding_lf, mo, output_path, pl):
    embedding_lf.sink_parquet(output_path)

    preview = pl.scan_parquet(output_path).limit(5).collect()
    row_count = (
        pl.scan_parquet(output_path).select(pl.len().alias("rows")).collect().item(0, "rows")
    )

    mo.vstack(
        [
            mo.md(f"Wrote `{output_path}`"),
            mo.md(f"Rows: **{row_count}**"),
            preview,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
