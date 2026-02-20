import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import duckdb
    import polars as pl

    return duckdb, mo, os, pl


@app.cell
def _(duckdb, os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    book_texts = os.path.join(project_root, "data", "books_embedding_texts.parquet")
    book_embeddings = os.path.join(
        project_root, "data", "books_finetuned_embeddings.parquet"
    )
    output_path = os.path.join(project_root, "data", "books_texts_embeddings.parquet")
    if not os.path.exists(book_texts):
        raise FileNotFoundError(f"Missing input parquet: {book_texts}")
    if not os.path.exists(book_embeddings):
        raise FileNotFoundError(f"Missing input parquet: {book_embeddings}")

    book_texts_sql = book_texts.replace("'", "''")
    book_embeddings_sql = book_embeddings.replace("'", "''")
    output_path_sql = output_path.replace("'", "''")

    with duckdb.connect() as conn:
        join_sql = f"""
        COPY (
            SELECT
                CAST(texts.book_id AS BIGINT) AS book_id,
                texts.book_embedding_text,
                embeddings.embedding
            FROM read_parquet('{book_texts_sql}') AS texts
            INNER JOIN read_parquet('{book_embeddings_sql}') AS embeddings
                ON texts.book_id = TRY_CAST(embeddings.book_id AS BIGINT)
        ) TO '{output_path_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        conn.execute(join_sql)

        books_texts_count = conn.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [book_texts]
        ).fetchone()[0]
        books_embeddings_count = conn.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [book_embeddings]
        ).fetchone()[0]
        joined_count = conn.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [output_path]
        ).fetchone()[0]

    books_texts_df = pl.read_parquet(book_texts, n_rows=3)
    books_embeddings_df = pl.read_parquet(book_embeddings, n_rows=3)
    joined_preview_df = pl.read_parquet(output_path, n_rows=3)
    row_counts = {
        "books_embedding_texts": books_texts_count,
        "books_finetuned_embeddings": books_embeddings_count,
        "joined_rows": joined_count,
    }
    return (
        books_embeddings_df,
        books_texts_df,
        joined_preview_df,
        output_path,
        row_counts,
    )


@app.cell
def _(books_texts_df, mo):
    mo.vstack([
        mo.md("Texts Dataset"),
        books_texts_df.head(3),
    ])
    return


@app.cell
def _(books_embeddings_df, mo):
    mo.vstack([
        mo.md("Books Embeddings"),
        books_embeddings_df.head(3)
    ])
    return


@app.cell
def _(joined_preview_df, mo, output_path, row_counts):
    mo.vstack([
        mo.md(f"Joined Dataset ({output_path})"),
        mo.md(
            f"- books_embedding_texts rows: {row_counts['books_embedding_texts']:,}\n"
            f"- books_finetuned_embeddings rows: {row_counts['books_finetuned_embeddings']:,}\n"
            f"- joined rows: {row_counts['joined_rows']:,}"
        ),
        joined_preview_df,
    ])
    return


if __name__ == "__main__":
    app.run()
