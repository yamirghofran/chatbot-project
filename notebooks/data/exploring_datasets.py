import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Exploring BookDB Datasets

    This notebook performs basic EDA on all parquet datasets in the data directory.
    """)
    return


@app.cell
def _():
    import polars as pl
    import pathlib
    import os

    # Get all parquet files using absolute path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = pathlib.Path(project_root) / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    parquet_files
    return parquet_files, pl


@app.cell
def _(parquet_files, pl):
    # Load all datasets into a dictionary
    datasets = {}

    for _file_path in parquet_files:
        _dataset_name = _file_path.stem
        print(f"Loading {_dataset_name}...")
        datasets[_dataset_name] = pl.scan_parquet(_file_path)

    print(f"\nLoaded {len(datasets)} datasets")
    datasets
    return (datasets,)


@app.cell
def _(parquet_files, pl):
    # Get basic info for all datasets
    dataset_info = {}

    for _file_path in parquet_files:
        _name = _file_path.stem
        _lazy_df = pl.scan_parquet(_file_path)

        _schema = _lazy_df.collect_schema()
        _dtypes = {field: dtype for field, dtype in zip(_schema.names(), _schema.dtypes())}
        _row_count = _lazy_df.select(pl.len()).collect().item()

        dataset_info[_name] = {
            "row_count": _row_count,
            "column_count": len(_schema.names()),
            "columns": list(_schema.names()),
            "dtypes": _dtypes,
        }

    dataset_info
    return (dataset_info,)


@app.cell
def _(dataset_info, pl):
    # Display summary table

    summary_data = {
        "Dataset": [],
        "Rows": [],
        "Columns": [],
        "Size (approx)": [],
    }

    for name, info in sorted(dataset_info.items()):
        summary_data["Dataset"].append(name)
        summary_data["Rows"].append(info["row_count"])
        summary_data["Columns"].append(info["column_count"])

        # Calculate approximate size in memory
        _size_bytes = info["row_count"] * info["column_count"] * 8
        if _size_bytes >= 1_000_000_000:
            _size_str = f"{_size_bytes / 1_000_000_000:.2f} GB"
        elif _size_bytes >= 1_000_000:
            _size_str = f"{_size_bytes / 1_000_000:.2f} MB"
        else:
            _size_str = f"{_size_bytes / 1_000:.2f} KB"
        summary_data["Size (approx)"].append(_size_str)

    summary_df = pl.DataFrame(summary_data)
    summary_df
    return info, name


@app.cell
def _(pl):
    # Create detailed EDA function for each dataset
    def analyze_dataset(_name, _lazy_df, _limit_rows=5):
        """Return detailed analysis of a dataset."""

        # Get schema info
        _schema = _lazy_df.collect_schema()

        # Collect sample data
        _sample = _lazy_df.head(_limit_rows).collect()

        # Get null counts (may be expensive for large datasets)
        _null_counts = _lazy_df.select(pl.all().null_count()).collect().row(0)
        _null_percentages = [
            f"{(_null_count / _lazy_df.select(pl.len()).collect().item() * 100):.2f}%"
            for _null_count in _null_counts
        ]

        # Get numeric statistics for numeric columns
        _numeric_cols = [
            _col_name
            for _col_name, _dtype in zip(_schema.names(), _schema.dtypes())
            if _dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ]
        ]

        _numeric_stats = None
        if _numeric_cols:
            try:
                _numeric_stats = (
                    _lazy_df.select(_numeric_cols)
                    .describe()
                    .collect()
                    .filter(pl.col("statistic").is_in(["mean", "min", "max", "std"]))
                )
            except Exception as _e:
                _numeric_stats = f"Could not compute: {str(_e)}"

        # Get unique value counts for string columns
        _string_cols = [
            _col_name
            for _col_name, _dtype in zip(_schema.names(), _schema.dtypes())
            if _dtype == pl.String
        ]

        _unique_counts = None
        if _string_cols:
            try:
                _unique_counts = (
                    _lazy_df.select([pl.col(c).n_unique().alias(c) for c in _string_cols])
                    .collect()
                    .row(0)
                )
            except Exception as _e:
                _unique_counts = f"Could not compute: {str(_e)}"

        return {
            "name": _name,
            "schema": _schema,
            "sample": _sample,
            "null_counts": dict(zip(_schema.names(), _null_counts)),
            "null_percentages": dict(zip(_schema.names(), _null_percentages)),
            "numeric_cols": _numeric_cols,
            "numeric_stats": _numeric_stats,
            "string_cols": _string_cols,
            "unique_counts": _unique_counts,
        }
    return (analyze_dataset,)


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_book_id_map
    mo.md("## Dataset: raw_book_id_map")
    result = analyze_dataset("raw_book_id_map",  datasets["raw_book_id_map"])
    result
    return (result,)


@app.cell
def _(result):
    result["schema"]
    return


@app.cell
def _(result):
    result["sample"]
    return


@app.cell
def _(pl, result):
    pl.DataFrame(
        {
            "Column": list(result["null_counts"].keys()),
            "Null Count": list(result["null_counts"].values()),
            "Null %": list(result["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_book_authors
    mo.md("## Dataset: raw_goodreads_book_authors")
    result_authors = analyze_dataset("raw_goodreads_book_authors", datasets["raw_goodreads_book_authors"])
    result_authors
    return (result_authors,)


@app.cell
def _(result_authors):
    result_authors["schema"]
    return


@app.cell
def _(result_authors):
    result_authors["sample"]
    return


@app.cell
def _(pl, result_authors):
    pl.DataFrame(
        {
            "Column": list(result_authors["null_counts"].keys()),
            "Null Count": list(result_authors["null_counts"].values()),
            "Null %": list(result_authors["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(pl, result_authors):
    # Display unique counts for string columns
    if result_authors["unique_counts"] and isinstance(result_authors["unique_counts"], tuple):
        pl.DataFrame(
            {
                "Column": result_authors["string_cols"],
                "Unique Count": result_authors["unique_counts"],
            }
        )
    else:
        result_authors["unique_counts"]
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_book_works
    mo.md("## Dataset: raw_goodreads_book_works")
    result_works = analyze_dataset("raw_goodreads_book_works", datasets["raw_goodreads_book_works"])
    result_works
    return (result_works,)


@app.cell
def _(result_works):
    result_works["schema"]
    return


@app.cell
def _(result_works):
    result_works["sample"]
    return


@app.cell
def _(pl, result_works):
    pl.DataFrame(
        {
            "Column": list(result_works["null_counts"].keys()),
            "Null Count": list(result_works["null_counts"].values()),
            "Null %": list(result_works["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(pl, result_works):
    # Display numeric statistics if available
    if result_works["numeric_stats"] and isinstance(_result_works["numeric_stats"], pl.DataFrame):
        result_works["numeric_stats"]
    else:
        result_works["numeric_stats"]
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_interactions
    mo.md("## Dataset: raw_goodreads_interactions")
    result_interactions = analyze_dataset("raw_goodreads_interactions", datasets["raw_goodreads_interactions"])
    result_interactions
    return (result_interactions,)


@app.cell
def _(result_interactions):
    result_interactions["schema"]
    return


@app.cell
def _(result_interactions):
    result_interactions["sample"]
    return


@app.cell
def _(pl, result_interactions):
    pl.DataFrame(
        {
            "Column": list(result_interactions["null_counts"].keys()),
            "Null Count": list(result_interactions["null_counts"].values()),
            "Null %": list(result_interactions["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(pl, result_interactions):
    # Display numeric statistics if available
    if result_interactions["numeric_stats"] and isinstance(result_interactions["numeric_stats"], pl.DataFrame):
        result_interactions["numeric_stats"]
    else:
        result_interactions["numeric_stats"]
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_interactions_dedup
    mo.md("## Dataset: raw_goodreads_interactions_dedup")
    result_interactions_dedup = analyze_dataset("raw_goodreads_interactions_dedup", datasets["raw_goodreads_interactions_dedup"])
    result_interactions_dedup
    return (result_interactions_dedup,)


@app.cell
def _(result_interactions_dedup):
    result_interactions_dedup["schema"]
    return


@app.cell
def _(result_interactions_dedup):
    result_interactions_dedup["sample"]
    return


@app.cell
def _(pl, result_interactions_dedup):
    pl.DataFrame(
        {
            "Column": list(result_interactions_dedup["null_counts"].keys()),
            "Null Count": list(result_interactions_dedup["null_counts"].values()),
            "Null %": list(result_interactions_dedup["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(pl, result_interactions_dedup):
    # Display numeric statistics if available
    if result_interactions_dedup["numeric_stats"] and isinstance(result_interactions_dedup["numeric_stats"], pl.DataFrame):
        result_interactions_dedup["numeric_stats"]
    else:
        result_interactions_dedup["numeric_stats"]
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_reviews_spoiler
    mo.md("## Dataset: raw_goodreads_reviews_spoiler")
    result_spoiler = analyze_dataset("raw_goodreads_reviews_spoiler", datasets["raw_goodreads_reviews_spoiler"])
    result_spoiler
    return (result_spoiler,)


@app.cell
def _(result_spoiler):
    result_spoiler["schema"]
    return


@app.cell
def _(result_spoiler):
    result_spoiler["sample"]
    return


@app.cell
def _(pl, result_spoiler):
    pl.DataFrame(
        {
            "Column": list(result_spoiler["null_counts"].keys()),
            "Null Count": list(result_spoiler["null_counts"].values()),
            "Null %": list(result_spoiler["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_goodreads_reviews_dedup
    mo.md("## Dataset: raw_goodreads_reviews_dedup")
    result_reviews = analyze_dataset("raw_goodreads_reviews_dedup", datasets["raw_goodreads_reviews_dedup"])
    result_reviews
    return (result_reviews,)


@app.cell
def _(result_reviews):
    result_reviews["schema"]
    return


@app.cell
def _(result_reviews):
    result_reviews["sample"]
    return


@app.cell
def _(pl, result_reviews):
    pl.DataFrame(
        {
            "Column": list(result_reviews["null_counts"].keys()),
            "Null Count": list(result_reviews["null_counts"].values()),
            "Null %": list(result_reviews["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(pl, result_reviews):
    # Display numeric statistics if available
    if result_reviews["numeric_stats"] and isinstance(result_reviews["numeric_stats"], pl.DataFrame):
        result_reviews["numeric_stats"]
    else:
        result_reviews["numeric_stats"]
    return


@app.cell
def _(analyze_dataset, datasets, mo):
    # Analyze raw_user_id_map
    mo.md("## Dataset: raw_user_id_map")
    result_user_map = analyze_dataset("raw_user_id_map", datasets["raw_user_id_map"])
    result_user_map
    return (result_user_map,)


@app.cell
def _(result_user_map):
    result_user_map["schema"]
    return


@app.cell
def _(result_user_map):
    result_user_map["sample"]
    return


@app.cell
def _(pl, result_user_map):
    pl.DataFrame(
        {
            "Column": list(result_user_map["null_counts"].keys()),
            "Null Count": list(result_user_map["null_counts"].values()),
            "Null %": list(result_user_map["null_percentages"].values()),
        }
    )
    return


@app.cell
def _(dataset_info):
    def _():
        content = "## Summary for README\n\nBased on the EDA above, here's a summary for the README:\n"

        for name, info in sorted(dataset_info.items()):
            content += f"\n### {name}\n"
            content += f"- **Rows:** {info['row_count']:,}\n"
            content += f"- **Columns:** {info['column_count']}\n"
            content += f"- **Features:** {', '.join(info['columns'][:10])}\n"
        
            #if len(info['columns']) > 10:
                #content += f"  ... and {len(info['columns']) - 10} more\n"
            
            content += "- **Data Types:**\n"
            for col, dtype in list(info['dtypes'].items())[:8]:
                content += f"  - {col}: {dtype}\n"

        print(content)
        #return mo.md(content)
    _()
    return


@app.cell
def _(dataset_info, info, mo, name):
    mo.md(
        """
        ## Summary for README

        Based on the EDA above, here's a summary for the README:
        """
    )

    # Create a comprehensive summary
    for _name, _info in sorted(dataset_info.items()):
        print(f"\n### {name}")
        print(f"- **Rows:** {info['row_count']:,}")
        print(f"- **Columns:** {info['column_count']}")
        print(f"- **Features:** {', '.join(info['columns'][:10])}")
        if len(info['columns']) > 10:
            print(f"  ... and {len(info['columns']) - 10} more")
        print(f"- **Data Types:**")
        for _col, _dtype in list(info['dtypes'].items())[:8]:
            print(f"  - {_col}: {_dtype}")
    return


if __name__ == "__main__":
    app.run()
