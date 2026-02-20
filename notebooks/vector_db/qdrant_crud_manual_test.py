import marimo

__generated_with = "0.19.11"
app = marimo.App(width="wide")


@app.cell
def _():
    import json
    import os
    import random
    import uuid

    import marimo as mo
    import polars as pl

    from bookdb.vector_db import (
        BookVectorCRUD,
        CollectionManager,
        CollectionNames,
        QdrantConfig,
        ReviewVectorCRUD,
        UserVectorCRUD,
        get_qdrant_client,
        reset_client,
    )

    return (
        BookVectorCRUD,
        CollectionManager,
        CollectionNames,
        QdrantConfig,
        ReviewVectorCRUD,
        UserVectorCRUD,
        get_qdrant_client,
        json,
        mo,
        os,
        pl,
        random,
        reset_client,
        uuid,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Qdrant Vector CRUD Manual Smoke Test

    This notebook uses fake data and fake embeddings to exercise vector CRUD
    behavior against a local Qdrant instance.

    **Prerequisites**
    - Start Qdrant: `make qdrant-up`
    - Ensure environment variables are set (see `.env.example`)
    """)
    return


@app.cell
def _(mo, os):
    default_vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    vector_size_input = mo.ui.number(
        start=1,
        stop=4096,
        step=1,
        value=default_vector_size,
        label="Vector size",
    )
    reset_collections = mo.ui.checkbox(
        label="Reset books/users/reviews before test run",
        value=True,
    )
    run_crud = mo.ui.run_button(label="Run CRUD smoke test", kind="success")
    return reset_collections, run_crud, vector_size_input


@app.cell
def _(mo, reset_collections, run_crud, vector_size_input):
    mo.hstack(
        [
            vector_size_input,
            reset_collections,
            run_crud,
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(
    BookVectorCRUD,
    CollectionManager,
    CollectionNames,
    QdrantConfig,
    ReviewVectorCRUD,
    UserVectorCRUD,
    get_qdrant_client,
    json,
    mo,
    pl,
    random,
    reset_client,
    reset_collections,
    run_crud,
    uuid,
    vector_size_input,
):
    def empty_records_df():
        return pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "document": pl.Utf8,
                "metadata": pl.Utf8,
                "embedding_len": pl.Int64,
                "embedding_head": pl.Utf8,
            }
        )

    def records_df(records):
        if not records:
            return empty_records_df()

        rows = []
        for item in records:
            embedding = item.get("embedding")
            rows.append(
                {
                    "id": str(item.get("id")),
                    "document": item.get("document"),
                    "metadata": json.dumps(item.get("metadata") or {}, sort_keys=True),
                    "embedding_len": len(embedding) if embedding else 0,
                    "embedding_head": str(embedding[:5]) if embedding else "[]",
                }
            )
        return pl.DataFrame(rows)

    operations = []

    def log(operation, status, detail):
        operations.append(
            {
                "operation": operation,
                "status": status,
                "detail": detail,
            }
        )

    counts_df = pl.DataFrame({"books": [0], "users": [0], "reviews": [0]})
    books_snapshots = []
    users_snapshots = []
    reviews_snapshots = []
    books_filtered = []
    reviews_filtered = []
    batch_records = []
    run_id = uuid.uuid4().hex[:8]
    vector_size = int(vector_size_input.value or 384)
    status_block = mo.md("Click **Run CRUD smoke test** to execute against local Qdrant.")

    if run_crud.value:
        try:
            def fake_embedding(seed, size):
                rng = random.Random(seed)
                return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(size)]

            reset_client()
            config = QdrantConfig.from_env()
            get_qdrant_client(config)

            manager = CollectionManager(config=config, vector_size=vector_size)
            manager.initialize_collections()

            if reset_collections.value:
                for collection_name in CollectionNames:
                    manager.reset_collection(collection_name)
                log("reset_collections", "ok", "books/users/reviews reset")

            log(
                "connect_qdrant",
                "ok",
                f"mode={config.mode}, host={config.host}, port={config.port}",
            )
            log("list_collections", "ok", ", ".join(manager.list_collections()))

            books_crud = BookVectorCRUD()
            users_crud = UserVectorCRUD()
            reviews_crud = ReviewVectorCRUD()

            def make_point_id(scope, index):
                return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:{scope}:{index}"))

            fake_books = [
                {
                    "id": make_point_id("book", 1),
                    "title": "Graph Databases in Practice",
                    "description": "Practical guide to graph modeling and querying.",
                    "sql_book_id": 10001,
                },
                {
                    "id": make_point_id("book", 2),
                    "title": "Distributed Systems Field Guide",
                    "description": "Failure modes, retries, and consistency tradeoffs.",
                    "sql_book_id": 10002,
                },
                {
                    "id": make_point_id("book", 3),
                    "title": "Applied Recommendation Systems",
                    "description": "Ranking, retrieval, and offline evaluation methods.",
                    "sql_book_id": 10003,
                },
            ]

            fake_users = [
                {
                    "id": make_point_id("user", 1),
                    "name": "Manual Tester A",
                    "pg_user_id": 50001,
                    "description": "Enjoys systems design and technical books.",
                },
                {
                    "id": make_point_id("user", 2),
                    "name": "Manual Tester B",
                    "pg_user_id": 50002,
                    "description": "Reads reviews before picking books.",
                },
            ]

            fake_reviews = [
                {
                    "id": make_point_id("review", 1),
                    "user_id": fake_users[0]["pg_user_id"],
                    "book_id": fake_books[0]["sql_book_id"],
                    "review_text": "Useful and concise. Great examples.",
                },
                {
                    "id": make_point_id("review", 2),
                    "user_id": fake_users[0]["pg_user_id"],
                    "book_id": fake_books[1]["sql_book_id"],
                    "review_text": "Dense but worth reading twice.",
                },
                {
                    "id": make_point_id("review", 3),
                    "user_id": fake_users[1]["pg_user_id"],
                    "book_id": fake_books[0]["sql_book_id"],
                    "review_text": "Clear writing and practical tips.",
                },
            ]

            for book in fake_books:
                books_crud.add_book(
                    book_id=book["id"],
                    title=book["title"],
                    description=book["description"],
                    embedding=fake_embedding(f"{book['id']}_seed", vector_size),
                )
            log("books.add_book", "ok", f"inserted={len(fake_books)}")

            for user in fake_users:
                users_crud.add_user(
                    user_id=user["id"],
                    name=user["name"],
                    pg_user_id=user["pg_user_id"],
                    description=user["description"],
                    embedding=fake_embedding(f"{user['id']}_seed", vector_size),
                )
            log("users.add_user", "ok", f"inserted={len(fake_users)}")

            for review in fake_reviews:
                reviews_crud.add_review(
                    review_id=review["id"],
                    user_id=review["user_id"],
                    book_id=review["book_id"],
                    review_text=review["review_text"],
                    embedding=fake_embedding(f"{review['id']}_seed", vector_size),
                )
            log("reviews.add_review", "ok", f"inserted={len(fake_reviews)}")

            books_crud.update_book(
                book_id=fake_books[0]["id"],
                description="Updated description for manual smoke test.",
                embedding=fake_embedding(f"{fake_books[0]['id']}_updated", vector_size),
            )
            log("books.update_book", "ok", fake_books[0]["id"])

            users_crud.update_user(
                user_id=fake_users[0]["id"],
                name="Manual Tester A+",
                description="Updated preferences after a new reading session.",
                embedding=fake_embedding(f"{fake_users[0]['id']}_updated", vector_size),
            )
            log("users.update_user", "ok", fake_users[0]["id"])

            reviews_crud.update_review(
                review_id=fake_reviews[0]["id"],
                review_text="Updated review text for manual testing.",
                embedding=fake_embedding(f"{fake_reviews[0]['id']}_updated", vector_size),
            )
            log("reviews.update_review", "ok", fake_reviews[0]["id"])

            books_filtered = books_crud.search_by_metadata(limit=10)
            log("books.search_by_metadata", "ok", f"matched={len(books_filtered)}")

            reviews_filtered = reviews_crud.search_by_metadata(
                book_id=fake_books[0]["sql_book_id"],
                limit=10,
            )
            log("reviews.search_by_metadata", "ok", f"matched={len(reviews_filtered)}")

            books_snapshots = [
                item
                for item in [
                    books_crud.get(fake_books[0]["id"]),
                    books_crud.get(fake_books[1]["id"]),
                ]
                if item is not None
            ]
            users_snapshots = [
                item
                for item in [
                    users_crud.get(fake_users[0]["id"]),
                    users_crud.get(fake_users[1]["id"]),
                ]
                if item is not None
            ]
            reviews_snapshots = reviews_crud.get_all(limit=10)
            log(
                "base.get/get_all",
                "ok",
                f"books={len(books_snapshots)} users={len(users_snapshots)} reviews={len(reviews_snapshots)}",
            )

            batch_ids = [
                make_point_id("batch_review", 1),
                make_point_id("batch_review", 2),
            ]
            reviews_crud.add_batch(
                ids=batch_ids,
                documents=[
                    "Batch review A",
                    "Batch review B",
                ],
                metadatas=[
                    {"user_id": 60001, "book_id": 20001},
                    {"user_id": 60002, "book_id": 20002},
                ],
                embeddings=[
                    fake_embedding(f"{batch_ids[0]}_seed", vector_size),
                    fake_embedding(f"{batch_ids[1]}_seed", vector_size),
                ],
            )
            batch_records = reviews_crud.get_batch(batch_ids)
            reviews_crud.delete_batch(batch_ids)
            log(
                "base.add_batch/get_batch/delete_batch",
                "ok",
                "reviews batch round-trip",
            )

            reviews_crud.delete(fake_reviews[-1]["id"])
            log("base.delete", "ok", fake_reviews[-1]["id"])

            counts_df = pl.DataFrame(
                {
                    "books": [books_crud.count()],
                    "users": [users_crud.count()],
                    "reviews": [reviews_crud.count()],
                }
            )
            log(
                "base.count",
                "ok",
                f"books={counts_df['books'][0]}, users={counts_df['users'][0]}, reviews={counts_df['reviews'][0]}",
            )

            status_block = mo.md(
                f"""
    ### Status
    Run `{run_id}` completed.

    - Qdrant mode: `{config.mode}`
    - Collections reset: `{reset_collections.value}`
    - Vector size used: `{vector_size}`
    """
            )
        except Exception as exc:
            status_block = mo.md(
                f"""
    ### Status
    Run `{run_id}` failed.

    Error:
    `{type(exc).__name__}: {exc}`
    """
            )
            log("pipeline", "error", f"{type(exc).__name__}: {exc}")

    operations_df = (
        pl.DataFrame(operations)
        if operations
        else pl.DataFrame(
            schema={
                "operation": pl.Utf8,
                "status": pl.Utf8,
                "detail": pl.Utf8,
            }
        )
    )

    mo.vstack(
        [
            status_block,
            mo.md("### Operation log"),
            mo.ui.dataframe(operations_df),
            mo.md("### Collection counts"),
            mo.ui.dataframe(counts_df),
            mo.md("### Book snapshots (`get`)"),
            mo.ui.dataframe(records_df(books_snapshots)),
            mo.md("### User snapshots (`get`)"),
            mo.ui.dataframe(records_df(users_snapshots)),
            mo.md("### Review snapshots (`get_all`)"),
            mo.ui.dataframe(records_df(reviews_snapshots)),
            mo.md("### Book metadata search results"),
            mo.ui.dataframe(records_df(books_filtered)),
            mo.md("### Review metadata search results"),
            mo.ui.dataframe(records_df(reviews_filtered)),
            mo.md("### Batch round-trip results (`add_batch/get_batch/delete_batch`)"),
            mo.ui.dataframe(records_df(batch_records)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
