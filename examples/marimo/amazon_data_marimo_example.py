import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Polars Tutorial — Amazon Furniture Dataset

    A hands-on tutorial using a real Amazon furniture product dataset to learn
    Polars operations: loading, cleaning, filtering, aggregating, and visualizing.
    """)
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Loading the CSV
    """)
    return


@app.cell
def _():
    import polars as pl

    df = pl.read_csv("data/furniture_amazon_dataset_sample copy.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    df.head(50)
    return df, pl


@app.cell
def _(mo):
    mo.md("""
    ## 2. Exploring the Data

    Let's look at the schema and basic statistics for the key columns.
    """)
    return


@app.cell
def _(df, mo):
    mo.vstack([
        mo.md("**Schema (column name → dtype):**"),
        mo.plain_text(str(df.schema)),
        mo.md("**Describe numeric/string columns:**"),
        df.describe(),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Selecting & Cleaning Columns
    """)
    return


@app.cell
def _(df, mo, pl):
    # Select useful columns and clean the price column
    products = df.select(
        "asin",
        "title",
        "brand",
        "price",
        "availability",
        "categories",
        "color",
        "material",
        "country_of_origin",
        "date_first_available",
    ).with_columns(
        # Strip '$' and convert price to float
        pl.col("price")
        .str.replace_all(r"[\$,]", "")
        .cast(pl.Float64, strict=False)
        .alias("price"),
    )

    mo.vstack([
        mo.md("**Cleaned product table (first 10 rows):**"),
        products.head(10),
    ])
    return (products,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Filtering
    """)
    return


@app.cell
def _(mo, pl, products):
    # Products with a price
    with_price = products.filter(pl.col("price").is_not_null())

    # Affordable items (under $50)
    affordable = with_price.filter(pl.col("price") < 50)

    # Expensive items (over $200)
    expensive = with_price.filter(pl.col("price") > 200)

    mo.vstack([
        mo.md(f"**Products with a price:** {with_price.height} rows"),
        mo.md(f"**Affordable (< $50):** {affordable.height} rows"),
        affordable.select("title", "brand", "price").head(5),
        mo.md(f"**Expensive (> $200):** {expensive.height} rows"),
        expensive.select("title", "brand", "price").head(5),
    ])
    return (with_price,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. String Operations & Categories
    """)
    return


@app.cell
def _(mo, pl, products):
    # Extract the first category from the list string
    categorized = products.with_columns(
        pl.col("categories")
        .str.extract(r"'([^']+)'", 1)
        .alias("top_category"),
    )

    category_counts = (
        categorized
        .group_by("top_category")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    mo.vstack([
        mo.md("**Top-level category distribution:**"),
        category_counts,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. GroupBy & Aggregation
    """)
    return


@app.cell
def _(mo, pl, with_price):
    brand_stats = (
        with_price
        .filter(pl.col("brand").is_not_null() & (pl.col("brand") != ""))
        .group_by("brand")
        .agg(
            pl.len().alias("product_count"),
            pl.col("price").mean().round(2).alias("avg_price"),
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
        )
        .sort("product_count", descending=True)
        .head(15)
    )

    mo.vstack([
        mo.md("**Top 15 brands by product count (with price stats):**"),
        brand_stats,
    ])
    return


@app.cell
def _(mo, pl, with_price):
    country_stats = (
        with_price
        .filter(pl.col("country_of_origin").is_not_null() & (pl.col("country_of_origin") != ""))
        .group_by("country_of_origin")
        .agg(
            pl.len().alias("count"),
            pl.col("price").mean().round(2).alias("avg_price"),
        )
        .sort("count", descending=True)
    )

    mo.vstack([
        mo.md("**Products by country of origin:**"),
        country_stats,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Sorting & Ranking
    """)
    return


@app.cell
def _(mo, pl, with_price):
    ranked = (
        with_price
        .filter(pl.col("brand").is_not_null() & (pl.col("brand") != ""))
        .with_columns(
            pl.col("price").rank(descending=True).alias("price_rank"),
        )
        .sort("price_rank")
        .select("price_rank", "title", "brand", "price")
        .head(10)
    )

    mo.vstack([
        mo.md("**Top 10 most expensive products:**"),
        ranked,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Window Functions
    """)
    return


@app.cell
def _(mo, pl, with_price):
    windowed = (
        with_price
        .filter(
            pl.col("material").is_not_null()
            & (pl.col("material") != "")
        )
        .with_columns(
            pl.col("price").mean().over("material").round(2).alias("material_avg_price"),
            pl.col("price").rank(descending=True).over("material").alias("rank_in_material"),
        )
        .sort("material", "rank_in_material")
        .select("title", "material", "price", "material_avg_price", "rank_in_material")
        .head(15)
    )

    mo.vstack([
        mo.md("**Price ranked within each material type:**"),
        windowed,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Lazy Evaluation
    """)
    return


@app.cell
def _(mo, pl, with_price):
    mo.md("""
    Polars lazy mode builds a query plan and optimizes it before execution.
    """)

    result = (
        with_price.lazy()
        .filter(pl.col("country_of_origin") == "China")
        .filter(pl.col("price") < 100)
        .group_by("material")
        .agg(
            pl.len().alias("count"),
            pl.col("price").mean().round(2).alias("avg_price"),
        )
        .filter(pl.col("count") > 1)
        .sort("avg_price", descending=True)
        .collect()
    )

    mo.vstack([
        mo.md("**Lazy query: Chinese-made products under $100, grouped by material:**"),
        result,
    ])
    return


if __name__ == "__main__":
    app.run()
