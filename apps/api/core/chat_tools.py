"""Chat tool implementations for the orchestrator.

Each tool is a plain function that accepts typed arguments plus injected
dependencies (db, qdrant, groq_client, etc.) and returns a standardized
result dict.  Alongside the functions, TOOL_DEFINITIONS provides the
JSON-schema descriptors that Groq's function-calling API requires.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groq import Groq
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from bookdb.db.models import Book, Review
from bookdb.models.chatbot_llm import (
    create_groq_client_sync,
    rewrite_query_sync,
)

_log = logging.getLogger(__name__)

from .book_queries import (
    BOOK_LOAD_OPTIONS,
    load_books_by_goodreads_ids,
    load_books_by_ids,
    serialize_books_with_engagement,
)
from .config import settings
from .embeddings import most_similar, most_similar_by_vector
from .entity_extraction import resolve_entities


# ---------------------------------------------------------------------------
# Groq function-calling tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": (
                "Search the BookDB catalogue using a DESCRIPTIVE natural-language "
                "query. Use ONLY when you have a concrete description of what kind "
                "of book the user wants (genre, theme, mood, specific title, etc.). "
                "The query parameter MUST be a rich, descriptive search phrase — "
                "never a single word, 'yes', or a vague confirmation. If the user "
                "just said 'yes' or 'sure', use get_recommendations instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A descriptive search query (at least 3-4 words) that "
                            "captures what the user is looking for. Expand short "
                            "or contextual requests using the conversation history. "
                            "For example, if the user discussed Harry Potter and "
                            "then said 'yes, recommend me some', write: "
                            "'fantasy books similar to Harry Potter with magic "
                            "and coming-of-age themes'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_book_details",
            "description": (
                "Fetch full details for a specific book by its ID, including "
                "description, tags, authors, and stats. Use this when the user asks "
                "about a particular book you already know the ID of."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "The internal book ID.",
                    },
                },
                "required": ["book_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_books",
            "description": (
                "Find books similar to a given book using vector similarity. "
                "Use when the user says 'more like this' or 'similar to X' and "
                "you already have a book ID from a previous tool result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "The internal book ID to find related books for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of related books to return.",
                        "default": 6,
                    },
                },
                "required": ["book_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Get personalized book recommendations for the current user based "
                "on their reading history, ratings, and shelves. Use this when: "
                "(a) the user gives a vague or short confirmation like 'yes', "
                "'sure', 'recommend me something'; "
                "(b) the user asks for general recommendations without specifying "
                "genre/theme/mood; "
                "(c) you don't have a descriptive query for search_books. "
                "This tool does NOT require a search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recommendations.",
                        "default": 6,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_books",
            "description": (
                "Compare two or three books side by side on dimensions like pacing, "
                "themes, tone, difficulty, and length. You can pass either book IDs "
                "(if you know them from previous results) or book titles. Titles "
                "will be fuzzy-matched against the database, so partial titles and "
                "minor misspellings are fine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of 2-3 book IDs to compare (use if you have IDs).",
                    },
                    "titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of 2-3 book titles to compare. Use this when the "
                            "user mentions books by name and you don't have IDs. "
                            "Partial titles and misspellings are handled."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_via_mcp",
            "description": (
                "Get book recommendations from the team's external recommendation "
                "engine. Use when the user wants personalized recommendations and "
                "you want to leverage the external recommendation system. Falls back "
                "to local recommendations if the external service is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recommendations.",
                        "default": 6,
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Standardized result builder
# ---------------------------------------------------------------------------


def _tool_result(
    *,
    success: bool,
    data: dict[str, Any] | None = None,
    books: list[dict[str, Any]] | None = None,
    source: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "data": data or {},
        "books": books or [],
        "source": source,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Tool: search_books
# ---------------------------------------------------------------------------


def _embed_text_via_service(text: str) -> list[float]:
    """Call the embedding service for a single text.  Returns [] on failure."""
    import requests as http_requests

    service_url = settings.EMBEDDING_SERVICE_URL
    if not service_url:
        return []

    endpoint = service_url.rstrip("/") + "/embed"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.EMBEDDING_SERVICE_API_KEY:
        headers["Authorization"] = f"Bearer {settings.EMBEDDING_SERVICE_API_KEY}"

    body = {
        "texts": [text],
        "model": settings.EMBEDDING_SERVICE_MODEL,
        "normalize_embeddings": True,
    }

    try:
        response = http_requests.post(
            endpoint,
            json=body,
            headers=headers,
            timeout=settings.EMBEDDING_SERVICE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        _log.warning("Embedding request failed: %s", e)
        return []

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        return []
    first = embeddings[0]
    if not isinstance(first, list) or not first:
        return []

    vector: list[float] = []
    for value in first:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            return []
    return vector


def _book_author_names(book: Book) -> str:
    names = [ba.author.name for ba in book.authors if ba.author]
    return ", ".join(names) if names else "Unknown"


def _payload_to_book_context(book: Book, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    document = payload.get("document")
    document = document.strip() if isinstance(document, str) else ""
    tags = [bt.tag.name for bt in book.tags if bt.tag]

    title = str(metadata.get("title") or book.title)
    author = str(metadata.get("author") or _book_author_names(book))
    shelves = str(metadata.get("shelves") or ", ".join(tags[:5]) or "unspecified")
    description = document or (book.description or "")
    description = description.strip() or "No description available."

    return (
        f"TITLE: {title}\n"
        f"AUTHOR: {author}\n"
        f"SHELVES: {shelves}\n"
        f"DESCRIPTION: {description}\n"
    )


def _apply_preferences_to_query(query: str, prefs: dict[str, Any] | None) -> str:
    """Append preference constraints to search query for better embedding."""
    if not prefs:
        return query
    parts = [query]
    if prefs.get("disliked_genres"):
        parts.append(f"excluding {', '.join(prefs['disliked_genres'])}")
    if prefs.get("preferred_mood"):
        parts.append(f"mood: {prefs['preferred_mood']}")
    if prefs.get("max_pages"):
        parts.append(f"under {prefs['max_pages']} pages")
    if prefs.get("standalone_only"):
        parts.append("standalone books only")
    if prefs.get("other_constraints"):
        parts.append("; ".join(prefs["other_constraints"]))
    return ", ".join(parts)


def tool_search_books(
    query: str,
    *,
    db: Session,
    qdrant: QdrantClient | None,
    groq_client: Groq | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Semantic book search via entity extraction → query rewriting → Qdrant."""
    if qdrant is None:
        return _tool_result(success=False, error="Vector search unavailable")

    query = _apply_preferences_to_query(query, preferences)

    groq_client = groq_client or create_groq_client_sync()

    # Entity extraction
    entity_context = None
    if settings.ENTITY_EXTRACTION_ENABLED:
        try:
            resolution = resolve_entities(
                db=db,
                query=query,
                max_books=settings.ENTITY_MAX_BOOKS_PER_QUERY,
                max_authors=settings.ENTITY_MAX_AUTHORS_PER_QUERY,
                similarity_threshold=settings.ENTITY_SIMILARITY_THRESHOLD,
                confidence_threshold=settings.ENTITY_CONFIDENCE_THRESHOLD,
            )
            entity_context = resolution.get("entity_context")
        except Exception as e:
            _log.warning("Entity extraction failed: %s", e)

    # Query rewriting
    try:
        rewritten_description, rewritten_review = rewrite_query_sync(
            groq_client,
            query,
            entity_context=entity_context,
        )
    except Exception as e:
        _log.warning("Query rewrite failed: %s", e)
        return _tool_result(success=False, error="Query rewrite failed")

    rewritten_text = "\n\n".join(
        part.strip()
        for part in [rewritten_description, rewritten_review]
        if part and part.strip()
    )
    if not rewritten_text:
        return _tool_result(success=False, error="Empty rewrite")

    # Embed
    query_embedding = _embed_text_via_service(rewritten_text)
    if not query_embedding:
        return _tool_result(success=False, error="Embedding failed")

    # Vector search
    try:
        qdrant_hits = most_similar_by_vector(
            qdrant,
            query_embedding,
            top_k=settings.CHATBOT_TOP_K,
        )
    except Exception as e:
        _log.warning("Qdrant vector search failed: %s", e)
        return _tool_result(success=False, error="Vector search failed")

    if not qdrant_hits:
        return _tool_result(
            success=True, data={"narrative": None}, source="vector_search"
        )

    goodreads_ids: list[int] = []
    for hit in qdrant_hits:
        try:
            goodreads_ids.append(int(hit["id"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not goodreads_ids:
        return _tool_result(
            success=True, data={"narrative": None}, source="vector_search"
        )

    qdrant_books = load_books_by_goodreads_ids(db, goodreads_ids)
    if not qdrant_books:
        return _tool_result(
            success=True, data={"narrative": None}, source="vector_search"
        )

    books_by_gid = {int(book.goodreads_id): book for book in qdrant_books}
    llm_context_books: list[dict[str, Any]] = []
    ranked_books: list[Book] = []

    for hit in qdrant_hits:
        try:
            gid = int(hit["id"])
        except (KeyError, TypeError, ValueError):
            continue
        book = books_by_gid.get(gid)
        if book is None:
            continue
        ranked_books.append(book)
        llm_context_books.append(
            {
                "book_id": int(book.id),
                "description": _payload_to_book_context(book, hit.get("payload", {})),
            }
        )

    if not ranked_books:
        return _tool_result(
            success=True, data={"narrative": None}, source="vector_search"
        )

    # Fetch reviews for grounding
    reviews: list[dict[str, Any]] = []
    if settings.CHATBOT_MAX_REVIEWS > 0:
        ranked_book_ids = [book.id for book in ranked_books]
        review_rows = db.execute(
            select(Review.id, Review.review_text, Book.title.label("book_title"))
            .join(Book, Book.id == Review.book_id)
            .where(Review.book_id.in_(ranked_book_ids))
            .order_by(Review.created_at.desc())
            .limit(settings.CHATBOT_MAX_REVIEWS)
        ).all()
        reviews = [
            {
                "review_id": int(row.id),
                "book_title": str(row.book_title or ""),
                "review": str(row.review_text or ""),
            }
            for row in review_rows
        ]

    max_books = settings.CHATBOT_MAX_BOOKS
    serialized = serialize_books_with_engagement(db, ranked_books[:max_books])

    return _tool_result(
        success=True,
        data={
            "llm_context_books": llm_context_books[:max_books],
            "reviews": reviews,
            "book_count": len(serialized),
        },
        books=serialized,
        source="vector_search",
    )


# ---------------------------------------------------------------------------
# Tool: get_book_details
# ---------------------------------------------------------------------------


def tool_get_book_details(
    book_id: int,
    *,
    db: Session,
) -> dict[str, Any]:
    """Fetch full details for a single book."""
    book = db.scalar(select(Book).where(Book.id == book_id).options(*BOOK_LOAD_OPTIONS))
    if book is None:
        return _tool_result(success=False, error=f"Book {book_id} not found")

    serialized = serialize_books_with_engagement(db, [book])
    return _tool_result(success=True, books=serialized, source="database")


# ---------------------------------------------------------------------------
# Tool: get_related_books
# ---------------------------------------------------------------------------


def tool_get_related_books(
    book_id: int,
    *,
    db: Session,
    qdrant: QdrantClient | None,
    limit: int = 6,
) -> dict[str, Any]:
    """Find books similar to a given book via Qdrant recommend."""
    row = db.execute(
        select(Book.id, Book.goodreads_id).where(Book.id == book_id)
    ).first()
    if row is None:
        return _tool_result(success=False, error=f"Book {book_id} not found")

    if qdrant is None or row.goodreads_id is None:
        return _tool_result(
            success=False, error="Vector search unavailable for this book"
        )

    try:
        similar_gids = most_similar(qdrant, row.goodreads_id, top_k=limit * 3)
    except Exception as e:
        _log.warning("Qdrant recommend failed for book %s: %s", book_id, e)
        return _tool_result(success=False, error="Vector search failed")

    if not similar_gids:
        return _tool_result(success=True, books=[], source="vector_similarity")

    books = load_books_by_goodreads_ids(db, similar_gids)[:limit]
    serialized = serialize_books_with_engagement(db, books)
    return _tool_result(success=True, books=serialized, source="vector_similarity")


# ---------------------------------------------------------------------------
# Tool: get_recommendations
# ---------------------------------------------------------------------------


def tool_get_recommendations(
    *,
    db: Session,
    qdrant: QdrantClient | None,
    request_app_state: Any | None = None,
    user_id: int | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Personalized recommendations via the discovery pipeline.

    This calls the same logic as GET /discovery/recommendations but as an
    in-process function call instead of an HTTP request.
    """
    from ..routers.discovery import (
        _append_unique_books,
        _bpr_recommendations,
        _cluster_vector_recommendations,
        _cold_start,
        _interaction_vector_recommendations,
    )

    bpr_path: str | None = (
        getattr(request_app_state, "bpr_parquet_path", None)
        if request_app_state
        else None
    )
    metrics_path: str | None = (
        getattr(request_app_state, "book_metrics_parquet_path", None)
        if request_app_state
        else None
    )

    # If no user, return cold start
    if user_id is None:
        cold = _cold_start(db, limit, metrics_path)
        serialized = serialize_books_with_engagement(db, cold)
        return _tool_result(success=True, books=serialized, source="cold_start")

    from bookdb.db.models import User

    user = db.scalar(select(User).where(User.id == user_id))
    if user is None:
        cold = _cold_start(db, limit, metrics_path)
        serialized = serialize_books_with_engagement(db, cold)
        return _tool_result(success=True, books=serialized, source="cold_start")

    # BPR
    bpr_books: list[Book] = []
    bpr_gids: list[int] = []
    if bpr_path and user.goodreads_id is not None:
        bpr_gids = _bpr_recommendations(
            bpr_path, user.goodreads_id, limit=max(limit * 5, 100)
        )
        if bpr_gids:
            bpr_books = load_books_by_goodreads_ids(db, bpr_gids)

    # Vector interaction recs
    interaction_books: list[Book] = []
    if qdrant is not None:
        interaction_gids = _cluster_vector_recommendations(
            db,
            user.id,
            qdrant_client=qdrant,
            limit=max(limit * 4, 80),
            exclude_ids=set(bpr_gids),
        )
        if not interaction_gids:
            interaction_gids = _interaction_vector_recommendations(
                db,
                user.id,
                qdrant_client=qdrant,
                limit=max(limit * 4, 80),
                exclude_ids=set(bpr_gids),
            )
        if interaction_gids:
            interaction_books = load_books_by_goodreads_ids(db, interaction_gids)

    recommendations: list[Book] = []
    if bpr_books and interaction_books:
        interaction_quota = max(1, min(limit // 3, 8)) if limit >= 3 else 0
        interaction_reserved = min(interaction_quota, len(interaction_books))
        bpr_target = max(limit - interaction_reserved, 0)
        _append_unique_books(recommendations, bpr_books, limit=bpr_target)
        _append_unique_books(recommendations, interaction_books, limit=limit)
        _append_unique_books(recommendations, bpr_books, limit=limit)
    else:
        _append_unique_books(recommendations, bpr_books, limit=limit)
        _append_unique_books(recommendations, interaction_books, limit=limit)

    if len(recommendations) < limit:
        cold = _cold_start(db, limit, metrics_path)
        _append_unique_books(recommendations, cold, limit=limit)

    source = (
        "bpr+vector"
        if bpr_books and interaction_books
        else ("bpr" if bpr_books else ("vector" if interaction_books else "cold_start"))
    )
    serialized = serialize_books_with_engagement(db, recommendations[:limit])
    return _tool_result(success=True, books=serialized, source=source)


# ---------------------------------------------------------------------------
# Tool: compare_books
# ---------------------------------------------------------------------------

_COMPARE_SYSTEM_PROMPT = """\
You are a book comparison assistant. Given details about 2-3 books, produce a \
structured comparison. Be specific and grounded in the provided metadata and descriptions.
"""

_COMPARE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_comparison",
        "schema": {
            "type": "object",
            "properties": {
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "values": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "values"],
                        "additionalProperties": False,
                    },
                },
                "verdict": {"type": "string"},
            },
            "required": ["dimensions", "verdict"],
            "additionalProperties": False,
        },
    },
}


def _resolve_books_for_comparison(
    db: Session,
    book_ids: list[int] | None,
    titles: list[str] | None,
) -> tuple[list[Any], list[str]]:
    """Resolve books by IDs or fuzzy title matching. Returns (books, warnings).

    When titles are resolved, books are re-fetched by ID with full
    relationship loading so that authors/tags are available.
    """
    from .entity_extraction import find_books_by_title

    warnings: list[str] = []

    if book_ids:
        books = load_books_by_ids(db, book_ids)
        if len(books) >= 2:
            return books, warnings
        warnings.append(f"Only found {len(books)} of {len(book_ids)} books by ID.")

    if titles:
        resolved_ids: list[int] = []
        seen_ids: set[int] = set()
        for title in titles:
            matches = find_books_by_title(db, title, limit=1, similarity_threshold=0.25)
            if matches:
                book, score = matches[0]
                if book.id not in seen_ids:
                    resolved_ids.append(book.id)
                    seen_ids.add(book.id)
                    if score < 0.5:
                        warnings.append(
                            f'Closest match for "{title}" was "{book.title}" '
                            f"(confidence: {score:.0%})"
                        )
            else:
                warnings.append(
                    f'Could not find a book matching "{title}" in the catalogue.'
                )
        if len(resolved_ids) >= 2:
            books = load_books_by_ids(db, resolved_ids)
            if len(books) >= 2:
                return books, warnings

    return [], warnings


def tool_compare_books(
    book_ids: list[int] | None = None,
    titles: list[str] | None = None,
    *,
    db: Session,
    groq_client: Groq | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison of 2-3 books by ID or title."""
    total = len(book_ids or []) + len(titles or [])
    if total < 2:
        return _tool_result(
            success=False, error="Provide at least 2 books to compare (by ID or title)"
        )
    if total > 3 and not book_ids:
        titles = (titles or [])[:3]

    books, warnings = _resolve_books_for_comparison(db, book_ids, titles)
    if len(books) < 2:
        msg = "Could not find enough books to compare."
        if warnings:
            msg += " " + " ".join(warnings)
        return _tool_result(success=False, error=msg)

    groq_client = groq_client or create_groq_client_sync()

    book_descriptions = []
    for book in books:
        author = _book_author_names(book)
        tags = [bt.tag.name for bt in book.tags if bt.tag]
        desc = (book.description or "").strip() or "No description available."
        book_descriptions.append(
            f"[book_id: {book.id}] TITLE: {book.title}\n"
            f"AUTHOR: {author}\nGENRE: {', '.join(tags[:5]) or 'unspecified'}\n"
            f"DESCRIPTION: {desc}"
        )

    user_message = (
        "Compare these books:\n\n"
        + "\n\n".join(book_descriptions)
        + "\n\nCompare on: pacing, themes, tone/mood, difficulty, target audience, and overall recommendation."
    )

    try:
        from bookdb.models.chatbot_llm import (
            DEFAULT_CHATBOT_MODEL,
            _create_structured_completion_with_retries_sync,
            _parse_structured_content,
        )

        response = _create_structured_completion_with_retries_sync(
            groq_client,
            model=DEFAULT_CHATBOT_MODEL,
            messages=[
                {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_completion_tokens=1024,
            response_format=_COMPARE_RESPONSE_SCHEMA,
        )
        parsed = _parse_structured_content(response, label="comparison")
    except Exception as e:
        _log.warning("Comparison LLM call failed: %s", e)
        parsed = None

    serialized = serialize_books_with_engagement(db, books)

    if parsed is None:
        return _tool_result(
            success=True,
            data={"comparison": None},
            books=serialized,
            source="comparison",
        )

    return _tool_result(
        success=True,
        data={"comparison": parsed},
        books=serialized,
        source="comparison",
    )


# ---------------------------------------------------------------------------
# Tool: recommend_via_mcp
# ---------------------------------------------------------------------------


def tool_recommend_via_mcp(
    *,
    mcp_adapter: Any | None = None,
    db: Session,
    qdrant: QdrantClient | None = None,
    request_app_state: Any | None = None,
    user_id: int | None = None,
    preferences: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Try the MCP recommendation server first; fall back to the local pipeline."""
    if mcp_adapter is not None and mcp_adapter.is_available():
        mcp_results = mcp_adapter.recommend(
            user_id=user_id,
            preferences=preferences,
            constraints=constraints,
            limit=limit,
        )
        if mcp_results:
            # Try to resolve MCP results to local book records for rich cards
            book_ids: list[int] = []
            for item in mcp_results:
                bid = item.get("book_id") or item.get("id")
                if bid is not None:
                    try:
                        book_ids.append(int(bid))
                    except (TypeError, ValueError):
                        continue

            if book_ids:
                books = load_books_by_ids(db, book_ids)
                if books:
                    serialized = serialize_books_with_engagement(db, books[:limit])
                    return _tool_result(success=True, books=serialized, source="mcp")

            return _tool_result(
                success=True,
                data={"raw_recommendations": mcp_results[:limit]},
                source="mcp",
            )

    # Fallback to local discovery pipeline
    result = tool_get_recommendations(
        db=db,
        qdrant=qdrant,
        request_app_state=request_app_state,
        user_id=user_id,
        limit=limit,
    )
    if result["source"] != "mcp":
        result["source"] = f"local_fallback ({result['source']})"
    return result


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "search_books": tool_search_books,
    "get_book_details": tool_get_book_details,
    "get_related_books": tool_get_related_books,
    "get_recommendations": tool_get_recommendations,
    "compare_books": tool_compare_books,
    "recommend_via_mcp": tool_recommend_via_mcp,
}
