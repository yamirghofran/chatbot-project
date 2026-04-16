"""Entity extraction for book titles and author names from user queries.

This module uses an LLM to extract book titles and author names from natural language,
then performs fuzzy matching against the database to find corresponding books/authors.
"""

from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Any, Optional

from cachetools import TTLCache
from openai import OpenAI
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from bookdb.db.models import Author, Book, BookAuthor, BookTag, Tag
from bookdb.models.chatbot_llm import (
    _create_structured_completion_with_retries_sync,
    _json_schema_with_strict_mode,
    _parse_structured_content,
    create_llm_client,
)

# ============================================================================
# Entity Extraction with LLM
# ============================================================================

_ENTITY_EXTRACTION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_ENTITY_EXTRACTION_RETRIES = 2

_ENTITY_EXTRACTION_PROMPT = """Extract book titles and author names from the user's query.

Rules:
1. Extract ONLY specific book titles and author names explicitly mentioned
2. Do NOT extract general preferences, genres, or themes
3. Do NOT extract books/authors that are hypothetical or mentioned as examples
4. Be conservative - if unsure about whether something is a book/author, don't include it
5. Include full titles (e.g., "A Game of Thrones" not just "Game of Thrones")
6. Include full author names when possible (e.g., "George R.R. Martin" not just "Martin")

Examples:
- "I love Harry Potter" → book_titles: ["Harry Potter"], author_names: []
- "I enjoy Stephen King novels" → book_titles: [], author_names: ["Stephen King"]
- "I like Game of Thrones and Lord of the Rings" → book_titles: ["Game of Thrones", "Lord of the Rings"], author_names: []
- "I want something like The Shining by Stephen King" → book_titles: ["The Shining"], author_names: ["Stephen King"]
- "I want an exciting adventure story" → book_titles: [], author_names: []
"""

_ENTITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_entities",
        "schema": {
            "type": "object",
            "properties": {
                "book_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific book titles mentioned in the query",
                },
                "author_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Author names mentioned in the query",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score (0-1) that the query contains specific entities",
                },
            },
            "required": ["book_titles", "author_names", "confidence"],
            "additionalProperties": False,
        },
    },
}


def extract_book_entities(query: str, client: Optional[OpenAI] = None) -> dict[str, Any]:
    """Extract book titles and author names from user query using LLM.

    Args:
        query: User's search query
        client: Optional OpenAI-compatible client (will create one if not provided)

    Returns:
        Dictionary with keys:
            - book_titles: List[str] - Extracted book titles
            - author_names: List[str] - Extracted author names
            - confidence: float - Confidence score (0-1)

    Example:
        >>> extract_book_entities("I love Harry Potter")
        {'book_titles': ['Harry Potter'], 'author_names': [], 'confidence': 0.95}
    """
    if not query or not query.strip():
        return {"book_titles": [], "author_names": [], "confidence": 0.0}

    if client is None:
        client = create_llm_client()

    response_format = _json_schema_with_strict_mode(
        name="book_entities",
        model=_ENTITY_EXTRACTION_MODEL,
        schema=_ENTITY_SCHEMA["json_schema"]["schema"],
    )

    try:
        response = _create_structured_completion_with_retries_sync(
            client,
            model=_ENTITY_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,  # Lower temperature for more consistent extraction
            max_completion_tokens=500,
            response_format=response_format,
        )

        data = _parse_structured_content(response, label="entity_extraction")
        if data is None:
            return {"book_titles": [], "author_names": [], "confidence": 0.0}

        book_titles = data.get("book_titles", [])
        author_names = data.get("author_names", [])
        confidence = data.get("confidence", 0.0)

        # Validate and sanitize
        if not isinstance(book_titles, list):
            book_titles = []
        if not isinstance(author_names, list):
            author_names = []
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        book_titles = [str(t).strip() for t in book_titles if t and str(t).strip()]
        author_names = [str(n).strip() for n in author_names if n and str(n).strip()]
        confidence = float(max(0.0, min(1.0, confidence)))

        return {
            "book_titles": book_titles,
            "author_names": author_names,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"Entity extraction LLM call failed: {e}")
        return {"book_titles": [], "author_names": [], "confidence": 0.0}


# ============================================================================
# Fuzzy Lookup with pg_trgm
# ============================================================================

# Cache for entity lookups to reduce database queries
_entity_lookup_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL


def _string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher.

    This is a fallback for when pg_trgm similarity() function is not available.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_books_by_title(
    db: Session,
    title: str,
    limit: int = 3,
    similarity_threshold: float = 0.3,
    use_cache: bool = True,
) -> list[tuple[Book, float]]:
    """Find books by title using PostgreSQL trigram similarity.

    Args:
        db: Database session
        title: Book title to search for
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1) to include results
        use_cache: Whether to use lookup cache

    Returns:
        List of (Book, similarity_score) tuples sorted by similarity score (descending)

    Note:
        This function uses PostgreSQL's similarity() function from the pg_trgm extension.
        If the extension is not available, it falls back to Python-based similarity.
    """
    if not title or not title.strip():
        return []

    cache_key = f"book:{title.lower()}:{limit}:{similarity_threshold}"
    if use_cache and cache_key in _entity_lookup_cache:
        return _entity_lookup_cache[cache_key]

    # Try to use pg_trgm similarity() function
    try:
        # Use PostgreSQL's similarity() function from pg_trgm
        query = text("""
            SELECT id, goodreads_id, title, description, image_url, format,
                   publisher, publication_year, isbn13,
                   similarity(lower(title), lower(:search_term)) as score
            FROM books
            WHERE similarity(lower(title), lower(:search_term)) >= :threshold
            ORDER BY score DESC
            LIMIT :limit
        """)

        result = db.execute(
            query,
            {
                "search_term": title,
                "threshold": similarity_threshold,
                "limit": limit,
            },
        ).fetchall()

        matches = []
        for row in result:
            book = Book(
                id=row.id,
                goodreads_id=row.goodreads_id,
                title=row.title,
                description=row.description,
                image_url=row.image_url,
                format=row.format,
                publisher=row.publisher,
                publication_year=row.publication_year,
                isbn13=row.isbn13,
            )
            score = float(row.score or 0.0)
            matches.append((book, score))

    except Exception as e:
        # Fallback to simple LIKE query + Python similarity scoring
        print(f"pg_trgm similarity not available, using fallback: {e}")

        matches = []

        # Get candidates with LIKE pattern
        candidates = db.scalars(
            select(Book).where(Book.title.ilike(f"%{title}%")).limit(limit * 5)
        ).all()

        # Score with Python similarity
        for book in candidates:
            score = _string_similarity(title, book.title)
            if score >= similarity_threshold:
                matches.append((book, score))

        # Sort by score
        matches.sort(key=lambda x: -x[1])
        matches = matches[:limit]

    # Cache the result
    if use_cache:
        _entity_lookup_cache[cache_key] = matches

    return matches


def find_authors_by_name(
    db: Session,
    name: str,
    limit: int = 3,
    similarity_threshold: float = 0.3,
    use_cache: bool = True,
) -> list[tuple[Author, float]]:
    """Find authors by name using PostgreSQL trigram similarity.

    Args:
        db: Database session
        name: Author name to search for
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1) to include results
        use_cache: Whether to use lookup cache

    Returns:
        List of (Author, similarity_score) tuples sorted by similarity score (descending)
    """
    if not name or not name.strip():
        return []

    cache_key = f"author:{name.lower()}:{limit}:{similarity_threshold}"
    if use_cache and cache_key in _entity_lookup_cache:
        return _entity_lookup_cache[cache_key]

    # Try to use pg_trgm similarity() function
    try:
        query = text("""
            SELECT id, goodreads_id, name, description,
                   similarity(lower(name), lower(:search_term)) as score
            FROM authors
            WHERE similarity(lower(name), lower(:search_term)) >= :threshold
            ORDER BY score DESC
            LIMIT :limit
        """)

        result = db.execute(
            query,
            {
                "search_term": name,
                "threshold": similarity_threshold,
                "limit": limit,
            },
        ).fetchall()

        matches = []
        for row in result:
            author = Author(
                id=row.id,
                goodreads_id=row.goodreads_id,
                name=row.name,
                description=row.description,
            )
            score = float(row.score or 0.0)
            matches.append((author, score))

    except Exception as e:
        # Fallback to simple LIKE query + Python similarity scoring
        print(f"pg_trgm similarity not available, using fallback: {e}")

        matches = []

        # Get candidates with LIKE pattern
        candidates = db.scalars(
            select(Author).where(Author.name.ilike(f"%{name}%")).limit(limit * 5)
        ).all()

        # Score with Python similarity
        for author in candidates:
            score = _string_similarity(name, author.name)
            if score >= similarity_threshold:
                matches.append((author, score))

        # Sort by score
        matches.sort(key=lambda x: -x[1])
        matches = matches[:limit]

    # Cache the result
    if use_cache:
        _entity_lookup_cache[cache_key] = matches

    return matches


# ============================================================================
# Context Generation
# ============================================================================


def get_book_context_string(
    book: Book, score: float = 0.0, db: Optional[Session] = None
) -> str:
    """Format book metadata for use as LLM context.

    Args:
        book: Book object (may be a partial object)
        score: Similarity score (0-1) indicating match quality
        db: Optional database session to fetch additional metadata (authors, tags)

    Returns:
        Formatted string with book metadata suitable for LLM context
    """
    # Load full book with relationships if db session provided
    if db and book.id:
        full_book = db.scalar(select(Book).where(Book.id == book.id))
        if full_book:
            book = full_book

    # Extract authors
    authors = []
    if hasattr(book, "authors") and book.authors:
        authors = [ba.author.name for ba in book.authors if ba.author]
    elif db and book.id:
        author_rows = db.execute(
            select(Author.name)
            .select_from(Author)
            .join(BookAuthor, BookAuthor.author_id == Author.id)
            .where(BookAuthor.book_id == book.id)
        ).fetchall()
        authors = [row.name for row in author_rows if row.name]

    author_str = ", ".join(authors) if authors else "Unknown"

    # Extract tags
    tags = []
    if hasattr(book, "tags") and book.tags:
        tags = [bt.tag.name for bt in book.tags if bt.tag]
    elif db and book.id:
        tag_rows = db.execute(
            select(Tag.name)
            .select_from(Tag)
            .join(BookTag, BookTag.tag_id == Tag.id)
            .where(BookTag.book_id == book.id)
        ).fetchall()
        tags = [row.name for row in tag_rows if row.name]

    tags_str = ", ".join(tags[:5]) if tags else "unspecified"

    # Get description
    description = getattr(book, "description", "") or ""
    description = description.strip() or "No description available."

    # Build context string
    context_parts = [
        f"TITLE: {book.title}",
        f"AUTHOR: {author_str}",
        f"GENRE: {tags_str}",
    ]

    if score > 0:
        context_parts.append(f"MATCH_SCORE: {score:.2f}")

    context_parts.append(f"DESCRIPTION: {description}")

    return "\n".join(context_parts)


def get_author_context_string(
    author: Author,
    score: float = 0.0,
    db: Optional[Session] = None,
    max_books: int = 3,
) -> str:
    """Format author metadata with their books for LLM context.

    Args:
        author: Author object
        score: Similarity score (0-1) indicating match quality
        db: Database session to fetch author's books
        max_books: Maximum number of books to include

    Returns:
        Formatted string with author and book metadata
    """
    author_context = f"AUTHOR: {author.name}"

    if score > 0:
        author_context += f" (MATCH_SCORE: {score:.2f})"

    if not db or not author.id:
        return author_context

    # Get author's books
    book_rows = db.execute(
        select(Book.id, Book.title, Book.description)
        .select_from(Book)
        .join(BookAuthor, BookAuthor.book_id == Book.id)
        .where(BookAuthor.author_id == author.id)
        .order_by(Book.id.desc())
        .limit(max_books)
    ).fetchall()

    if not book_rows:
        return author_context

    author_context += "\nBOOKS BY THIS AUTHOR:"
    for row in book_rows:
        description = (row.description or "").strip() or "No description available."
        # Truncate long descriptions
        if len(description) > 200:
            description = description[:197] + "..."

        author_context += f"\n  - {row.title}: {description}"

    return author_context


# ============================================================================
# High-Level Entity Resolution
# ============================================================================


def resolve_entities(
    db: Session,
    query: str,
    max_books: int = 2,
    max_authors: int = 2,
    similarity_threshold: float = 0.4,
    confidence_threshold: float = 0.7,
) -> dict[str, Any]:
    """Extract and resolve entities from a user query.

    This is the main high-level function that:
    1. Extracts book titles and author names using LLM
    2. Looks them up in the database with fuzzy matching
    3. Returns the matched entities and a context string

    Args:
        db: Database session
        query: User's search query
        max_books: Maximum number of books to resolve
        max_authors: Maximum number of authors to resolve
        similarity_threshold: Minimum similarity score for database matches
        confidence_threshold: Minimum LLM confidence to attempt extraction

    Returns:
        Dictionary with keys:
            - books: List of (Book, score) tuples
            - authors: List of (Author, score) tuples
            - entity_context: String formatted for LLM context
            - confidence: float - LLM confidence in extraction
    """
    # Step 1: Extract entities from query
    entities = extract_book_entities(query)

    confidence = entities.get("confidence", 0.0)
    if confidence < confidence_threshold:
        # Low confidence - don't attempt entity resolution
        return {
            "books": [],
            "authors": [],
            "entity_context": None,
            "confidence": confidence,
        }

    # Step 2: Resolve book titles
    found_books = []
    for book_title in entities.get("book_titles", [])[:max_books]:
        matches = find_books_by_title(
            db,
            book_title,
            limit=3,
            similarity_threshold=similarity_threshold,
        )

        # Only include high-quality matches
        for book, score in matches:
            if score > 0.5:  # Higher threshold for inclusion
                found_books.append((book, score))
                break  # Only take best match per title

    # Step 3: Resolve author names
    found_authors = []
    for author_name in entities.get("author_names", [])[:max_authors]:
        matches = find_authors_by_name(
            db,
            author_name,
            limit=3,
            similarity_threshold=similarity_threshold,
        )

        for author, score in matches:
            if score > 0.5:
                found_authors.append((author, score))
                break

    # Step 4: Build context string
    context_parts = []

    # Add book contexts (sorted by similarity)
    for book, score in sorted(found_books, key=lambda x: -x[1])[:max_books]:
        context_parts.append(get_book_context_string(book, score, db))

    # Add author contexts (sorted by similarity)
    for author, score in sorted(found_authors, key=lambda x: -x[1])[:max_authors]:
        context_parts.append(get_author_context_string(author, score, db))

    entity_context = "\n\n".join(context_parts) if context_parts else None

    return {
        "books": found_books,
        "authors": found_authors,
        "entity_context": entity_context,
        "confidence": confidence,
    }


# ============================================================================
# Cache Management
# ============================================================================


def clear_entity_cache() -> None:
    """Clear the entity lookup cache."""
    _entity_lookup_cache.clear()


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the entity lookup cache."""
    return {
        "size": len(_entity_lookup_cache),
        "maxsize": _entity_lookup_cache.maxsize,
        "ttl": _entity_lookup_cache.ttl,
    }
