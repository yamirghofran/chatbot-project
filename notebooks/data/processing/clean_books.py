import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


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

    return json, mo, os, pl


@app.cell
def _(mo, os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_path = os.path.join(project_root, "data", "raw_goodreads_books.parquet")
    df = pl.read_parquet(data_path)

    df = df.with_columns(
        [
            pl.when(pl.col(c).str.len_chars() == 0)
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in [
                "publication_month",
                "publication_year",
                "publication_day",
            ]
        ]
    )

    df = df.with_columns(
        [
            pl.col("text_reviews_count").cast(pl.Int64, strict=False),
            pl.col("ratings_count").cast(pl.Int64, strict=False),
            pl.col("average_rating").cast(pl.Float64, strict=False),
            pl.col("num_pages").cast(pl.Int64, strict=False),
            pl.col("publication_day").cast(pl.Int64, strict=False),
            pl.col("publication_month").cast(pl.Int64, strict=False),
            pl.col("publication_year").cast(pl.Int64, strict=False),
        ]
    )

    mo.vstack(
        [
            df.head(),
            mo.md(f"Dataset contains {df.shape[0]} books and {df.shape[1]} columns."),
        ]
    )
    return df, project_root


@app.cell
def _(df, json, mo, os, pl, project_root):
    best_book_map_path = os.path.join(project_root, "data", "best_book_id_map.json")

    # Load the best book ID map
    with open(best_book_map_path, "r") as f:
        best_book_map = json.load(f)

    # Extract non-best book IDs (values from the JSON to drop)
    # Convert to strings to match the book_id column type in the dataframe
    non_best_book_ids = set()
    for book_id_list in best_book_map.values():
        non_best_book_ids.update(str(book_id) for book_id in book_id_list)

    # Filter to EXCLUDE non-best book IDs
    # This keeps: (1) best book IDs and (2) books with only one version
    df_filtered = df.filter(~pl.col("book_id").is_in(non_best_book_ids))

    mo.vstack(
        [
            mo.md(f"**Before filtering:** {df.shape[0]} books"),
            mo.md(f"**After filtering:** {df_filtered.shape[0]} books"),
            mo.md(
                f"**Removed:** {df.shape[0] - df_filtered.shape[0]} books ({((df.shape[0] - df_filtered.shape[0]) / df.shape[0] * 100):.1f}%)"
            ),
            mo.md("### Filtered Dataset Sample"),
            df_filtered.head(),
        ]
    )
    return (df_filtered,)


@app.cell
def _(df_filtered, mo):
    # Define columns to drop
    columns_to_drop = [
        "text_reviews_count",
        "series",
        "country_code",
        "asin",
        "is_ebook",
        "average_rating",
        "kindle_asin",
        "publication_day",
        "publication_month",
        "edition_information",
        "ratings_count",
        "title_without_series",
        "isbn",  # Keep isbn13 instead
    ]

    # Drop the specified columns
    df_cleaned = df_filtered.drop(columns_to_drop)

    # Get remaining columns
    remaining_columns = df_cleaned.columns

    mo.vstack(
        [
            mo.md(f"**Dropped {len(columns_to_drop)} columns:**"),
            mo.md(", ".join([f"`{col}`" for col in columns_to_drop])),
            mo.md(f"**Remaining {len(remaining_columns)} columns:**"),
            mo.md(", ".join([f"`{col}`" for col in remaining_columns])),
            mo.md(
                f"**Before dropping:** {df_filtered.shape[0]} rows, {df_filtered.shape[1]} columns"
            ),
            mo.md(
                f"**After dropping:** {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns"
            ),
            mo.md("### Cleaned Dataset Sample"),
            df_cleaned.head(),
        ]
    )
    return (df_cleaned,)


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


@app.cell
def _(mo):
    mo.md("""
    Uncomment the code below to save the cleaned dataset in the data directory.
    """)
    return


@app.cell
def _():
    # Save the final cleaned and standardized dataset to parquet
    # output_path = os.path.join(project_root, "data", "goodreads_books_cleaned.parquet")
    # df_mapped.write_parquet(output_path)
    # mo.md(f"**Dataset saved successfully!**")
    # mo.md(f"**Location:** `{output_path}`")
    # mo.md(f"**File size:** {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    # mo.md(f"**Final dataset:** {df_mapped.shape[0]:,} books × {df_mapped.shape[1]} columns")
    return


if __name__ == "__main__":
    app.run()
