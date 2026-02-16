import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # This notebook standardized book_ids to the integer id in the csv mapping
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from scipy import stats
    import json
    import os

    return mo, os, pl


@app.cell
def _(os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_path = os.path.join(project_root, "data", "goodreads_books_cleaned.parquet")
    df_cleaned = pl.read_parquet(data_path)
    return df_cleaned, project_root


@app.cell
def _(df_cleaned, mo, os, pl, project_root):
    """Standardize Book IDs to CSV Integer Versions"""
    # Load book_id_map.parquet
    book_id_map_path = os.path.join(project_root, "data", "raw_book_id_map.parquet")
    book_id_map = pl.read_parquet(book_id_map_path)

    # Create lookup dictionary: {original_book_id (as string): csv_book_id (int)}
    book_id_lookup = dict(
        zip(
            book_id_map["book_id"].cast(str),  # Convert to string to match df_cleaned
            book_id_map["book_id_csv"],
        )
    )

    # Check coverage
    original_count = df_cleaned.shape[0]
    book_ids_in_data = set(df_cleaned["book_id"].unique().to_list())
    book_ids_in_map = set(book_id_map["book_id"].cast(str).unique().to_list())
    unmapped_count = len(book_ids_in_data - book_ids_in_map)


    # Drop rows with unmapped book_ids if any
    df_mapped = df_cleaned.filter(pl.col("book_id").is_in(book_ids_in_map))

    # Map book_id column to CSV integer version
    df_mapped = df_mapped.with_columns(
        pl.col("book_id")
        .map_elements(lambda x: book_id_lookup.get(x), return_dtype=pl.Int64)
        .alias("book_id")
    )

    # Map similar_books list
    def map_similar_books(book_ids_list):
        """Map list of book ID strings to CSV integers, removing unmapped IDs"""
        mapped_ids = []
        for bid in book_ids_list:
            if bid in book_id_lookup:
                mapped_ids.append(book_id_lookup[bid])
        return mapped_ids

    df_mapped = df_mapped.with_columns(
        pl.col("similar_books")
        .map_elements(map_similar_books, return_dtype=pl.List(pl.Int64))
        .alias("similar_books")
    )

    # Verify and display results
    book_id_type_after = df_mapped.schema["book_id"]
    similar_books_type_after = df_mapped.schema["similar_books"]
    books_removed = original_count - df_mapped.shape[0]

    mo.vstack(
        [
            mo.md("### Book ID Standardization"),
            mo.md(f"**Mapping coverage:** {len(book_ids_in_map):,} book IDs in map"),
            mo.md(f"**Books in dataset:** {len(book_ids_in_data):,} book IDs"),
            mo.md(f"**Unmapped book IDs:** {unmapped_count}"),
            mo.md(
                f"**Mapping success rate:** {(1 - unmapped_count / len(book_ids_in_data)) * 100:.2f}%"
            ),
            mo.md("### Standardization Results"),
            mo.md(f"**Before:** {original_count:,} books"),
            mo.md(
                f"**After:** {df_mapped.shape[0]:,} books (removed {books_removed} with unmapped IDs)"
            ),
            mo.md(f"**book_id type:** {book_id_type_after} (was String)"),
            mo.md(f"**similar_books type:** {similar_books_type_after}"),
            mo.md("### Standardized Dataset Sample"),
            df_mapped.head(),
            mo.md("### Key Transformations"),
            mo.md("- `book_id`: String → Int64 (mapped to CSV version)"),
            mo.md(
                "- `similar_books`: List(String) → List(Int64) (mapped to CSV versions)"
            ),
            mo.md("### Verification Checks"),
            mo.md(f"**Minimum book_id:** {df_mapped['book_id'].min()}"),
            mo.md(f"**Maximum book_id:** {df_mapped['book_id'].max()}"),
            mo.md(f"**Unique book_ids:** {df_mapped['book_id'].n_unique():,}"),
            mo.md(
                f"**Books with similar_books:** {df_mapped.filter(pl.col('similar_books').list.len() > 0).shape[0]:,}"
            ),
            mo.md(
                f"**Empty similar_books:** {df_mapped.filter(pl.col('similar_books').list.len() == 0).shape[0]:,}"
            ),
        ]
    )
    return


app._unparsable_cell(
    r"""
    Uncomment the code below to save the standardized dataset to the data directory.
    """,
    name="_"
)


@app.cell
def _():
    # Save the standardized dataframe to parquet
    # output_path = os.path.join(project_root, "data", "goodreads_books_standardized.parquet")
    # df_mapped.write_parquet(output_path)
    # mo.md(f"**Saved standardized dataset to:** `{output_path}`")
    return


if __name__ == "__main__":
    app.run()
