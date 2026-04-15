"""Generate benchmark dataset for embedding model evaluation.

Creates 200 diverse book queries with ground truth similar books
using the similar_books field from the dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import polars as pl


def book_has_genre(book_shelves: list | None, genre: str) -> bool:
    """Check if a book has a genre in its popular shelves."""
    if not book_shelves:
        return False
    genre_lower = genre.lower()
    return any(genre_lower in str(shelf).lower() for shelf in book_shelves)


def sample_books_with_genre(
    books_df: pl.DataFrame, genre: str, n: int = 100
) -> pl.DataFrame:
    """Sample n books that have a genre in their shelves."""
    # Sample more books and filter in Python
    sampled = books_df.sample(min(n * 10, books_df.height))
    books_list = sampled.to_dicts()
    matching = [
        b for b in books_list if book_has_genre(b.get("popular_shelves"), genre)
    ]

    if matching:
        selected = random.sample(matching, min(n, len(matching)))
        return pl.DataFrame(selected)
    return pl.DataFrame()


def generate_genre_queries(
    books_df: pl.DataFrame, n_queries: int = 50
) -> list[dict[str, Any]]:
    """Generate genre-based queries."""
    queries = []

    # Common genre patterns
    genres = [
        "fantasy",
        "romance",
        "mystery",
        "thriller",
        "sci-fi",
        "historical",
        "young adult",
        "horror",
        "contemporary",
        "adventure",
        "literary",
        "dystopian",
        "paranormal",
    ]

    # Sample books that have these genres
    for genre in random.sample(genres, min(n_queries, len(genres))):
        genre_books = sample_books_with_genre(books_df, genre, 50)

        if genre_books.height > 0:
            book = genre_books.sample(1).to_dicts()[0]
            queries.append(
                {
                    "query_id": f"genre_{len(queries)}",
                    "query_text": f"{genre} books",
                    "query_type": "genre",
                    "source_book_id": book["book_id"],
                    "ground_truth_ids": book.get("similar_books", [])[:20],
                    "difficulty": "medium",
                }
            )

    # Fill remaining with random genre + modifier combinations
    modifiers = ["best", "popular", "new", "classic", "award-winning"]
    while len(queries) < n_queries:
        genre = random.choice(genres)
        modifier = random.choice(modifiers)

        genre_books = sample_books_with_genre(books_df, genre, 50)

        if genre_books.height > 0:
            book = genre_books.sample(1).to_dicts()[0]
            queries.append(
                {
                    "query_id": f"genre_{len(queries)}",
                    "query_text": f"{modifier} {genre} books",
                    "query_type": "genre",
                    "source_book_id": book["book_id"],
                    "ground_truth_ids": book.get("similar_books", [])[:20],
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                }
            )

    return queries[:n_queries]


def generate_description_queries(
    books_df: pl.DataFrame, n_queries: int = 75
) -> list[dict[str, Any]]:
    """Generate description-based queries from book descriptions."""
    queries = []

    themes = [
        "love",
        "war",
        "family",
        "friendship",
        "death",
        "magic",
        "journey",
        "secret",
        "mystery",
        "power",
        "betrayal",
        "survival",
        "identity",
        "dream",
        "time",
        "world",
    ]

    # Sample books with descriptions
    books_with_desc = books_df.filter(
        pl.col("description").is_not_null()
        & (pl.col("description").str.len_chars() > 100)
    )

    sampled = books_with_desc.sample(min(n_queries * 3, books_with_desc.height))

    for book in sampled.iter_rows(named=True):
        if len(queries) >= n_queries:
            break

        description = book.get("description", "")
        if not description:
            continue

        sentences = str(description).split(".")
        query_sentence = None

        for sentence in sentences:
            if (
                any(theme in sentence.lower() for theme in themes)
                and 20 < len(sentence) < 150
            ):
                query_sentence = sentence.strip()
                break

        if query_sentence and book.get("similar_books"):
            queries.append(
                {
                    "query_id": f"desc_{len(queries)}",
                    "query_text": query_sentence,
                    "query_type": "description",
                    "source_book_id": book["book_id"],
                    "ground_truth_ids": book["similar_books"][:20],
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                }
            )

    return queries[:n_queries]


def generate_author_style_queries(
    books_df: pl.DataFrame, n_queries: int = 50
) -> list[dict[str, Any]]:
    """Generate author/style similarity queries."""
    queries = []

    books_with_authors = books_df.filter(
        pl.col("authors").is_not_null()
        & (pl.col("authors").list.len() > 0)
        & pl.col("similar_books").is_not_null()
    )

    sampled = books_with_authors.sample(min(n_queries * 2, books_with_authors.height))

    for book in sampled.iter_rows(named=True):
        if len(queries) >= n_queries:
            break

        authors = book.get("authors", [])
        if not authors:
            continue

        author_name = authors[0]
        title = book.get("title", "")

        if len(title) > 50:
            title = title[:47] + "..."

        queries.append(
            {
                "query_id": f"author_{len(queries)}",
                "query_text": f"books similar to {title} by {author_name}",
                "query_type": "author_style",
                "source_book_id": book["book_id"],
                "ground_truth_ids": book["similar_books"][:20],
                "difficulty": "medium",
            }
        )

    return queries[:n_queries]


def generate_hybrid_queries(
    books_df: pl.DataFrame, n_queries: int = 25
) -> list[dict[str, Any]]:
    """Generate hybrid queries combining multiple elements."""
    queries = []

    genres = ["fantasy", "romance", "mystery", "thriller", "science fiction"]
    audiences = ["young adult", "adult"]

    for _ in range(n_queries):
        genre = random.choice(genres)
        audience = random.choice(audiences)

        # Sample and filter
        sampled = books_df.sample(min(500, books_df.height))
        books_list = sampled.to_dicts()
        matching = [
            b
            for b in books_list
            if book_has_genre(b.get("popular_shelves"), genre)
            and b.get("similar_books")
        ]

        if matching:
            book = random.choice(matching)
            queries.append(
                {
                    "query_id": f"hybrid_{len(queries)}",
                    "query_text": f"{audience} {genre} books",
                    "query_type": "hybrid",
                    "source_book_id": book["book_id"],
                    "ground_truth_ids": book["similar_books"][:20],
                    "difficulty": "medium",
                }
            )

    return queries


def main():
    """Generate benchmark dataset."""
    random.seed(42)

    books_path = Path("data/3_goodreads_books_with_metrics.parquet")
    if not books_path.exists():
        print(f"Error: {books_path} not found")
        return

    print(f"Loading books from {books_path}...")
    books_df = pl.read_parquet(books_path)

    books_with_similar = books_df.filter(
        pl.col("similar_books").is_not_null() & (pl.col("similar_books").list.len() > 0)
    )

    print(f"Found {books_with_similar.height} books with similar_books")

    print("Generating benchmark queries...")

    all_queries = []
    all_queries.extend(generate_genre_queries(books_with_similar, 50))
    print(f"  Generated {len(all_queries)} genre queries")

    all_queries.extend(generate_description_queries(books_with_similar, 75))
    print(f"  Generated {len(all_queries)} total queries (added description)")

    all_queries.extend(generate_author_style_queries(books_with_similar, 50))
    print(f"  Generated {len(all_queries)} total queries (added author)")

    all_queries.extend(generate_hybrid_queries(books_with_similar, 25))
    print(f"  Generated {len(all_queries)} total queries (added hybrid)")

    random.shuffle(all_queries)

    for i, query in enumerate(all_queries):
        query["query_id"] = f"q{i:03d}"

    output_path = Path("data/benchmark_ground_truth.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_data = {
        "metadata": {
            "total_queries": len(all_queries),
            "generated_from": str(books_path),
            "ground_truth_source": "similar_books field",
            "query_types": {
                "genre": 50,
                "description": 75,
                "author_style": 50,
                "hybrid": 25,
            },
        },
        "queries": all_queries,
    }

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nGenerated {len(all_queries)} benchmark queries")
    print(f"Saved to {output_path}")

    type_counts = {}
    for query in all_queries:
        query_type = query["query_type"]
        type_counts[query_type] = type_counts.get(query_type, 0) + 1

    print("\nQuery type distribution:")
    for query_type, count in sorted(type_counts.items()):
        print(f"  {query_type}: {count}")


if __name__ == "__main__":
    main()
