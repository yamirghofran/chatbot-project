from groq import AsyncGroq
import os
import asyncio
import json
from typing import Any, Optional

# QUERY REWRITING TO BOOK DESCRIPTIONS AND REVIEWS

DEFAULT_QUERY_REWRITER_MODEL = os.environ.get("DEFAULT_QUERY_REWRITER_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_STRICT_STRUCTURED_OUTPUT_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}
_STRUCTURED_OUTPUT_RETRIES = int(os.environ.get("STRUCTURED_OUTPUT_RETRIES", "2"))

BOOK_DESCRIPTION_PROMPT = """You're a prompt re-writer. Your job is to rewrite \
 user's queries to look like a book description. Make up the title, author, shelves, \
 and description that would fit best the user's query."""

BOOK_REVIEW_PROMPT = """You're a prompt re-writer. Your job is to rewrite user's queries \
 to look like a book review. Make up a human review that would fit best the user's query, \
 make it sound natural and engaging, talk about the things you liked and disliked, do not \
 describe the plot, like you were an experienced reader in the review section of this book. \
 Keep the review around 500 characters."""

_BOOK_DESCRIPTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_description",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "shelves": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title", "author", "shelves", "description"],
            "additionalProperties": False,
        },
    },
}

_BOOK_REVIEW_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_review",
        "schema": {
            "type": "object",
            "properties": {
                "review": {"type": "string"},
            },
            "required": ["review"],
            "additionalProperties": False,
        },
    },
}


def _json_schema_with_strict_mode(
    *,
    name: str,
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    # Strict constrained decoding is only supported on a subset of models.
    strict = model.strip().lower() in _STRICT_STRUCTURED_OUTPUT_MODELS
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def _parse_structured_content(response: Any, *, label: str) -> dict[str, Any] | None:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError):
        print(f"Missing choices/message for {label} response")
        return None

    refusal = getattr(message, "refusal", None)
    if refusal:
        print(f"Model refused {label} response: {refusal}")
        return None

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        print(f"Empty structured content for {label} response")
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response for {label}: {e}")
        return None
    if not isinstance(parsed, dict):
        print(f"Unexpected {label} payload type: {type(parsed).__name__}")
        return None
    return parsed


async def _create_structured_completion_with_retries(
    client: AsyncGroq,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
    response_format: dict[str, Any],
):
    retries = max(0, _STRUCTURED_OUTPUT_RETRIES)
    for attempt in range(retries + 1):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                stream=False,
                response_format=response_format,
            )
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))


def create_groq_client(api_key: Optional[str] = None) -> AsyncGroq:
    return AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"))


async def _rewrite_description(client: AsyncGroq, query: str) -> str:
    response_format = _json_schema_with_strict_mode(
        name="book_description",
        model=DEFAULT_QUERY_REWRITER_MODEL,
        schema=_BOOK_DESCRIPTION_SCHEMA["json_schema"]["schema"],
    )
    response = await _create_structured_completion_with_retries(
        client,
        model=DEFAULT_QUERY_REWRITER_MODEL,
        messages=[
            {"role": "system", "content": BOOK_DESCRIPTION_PROMPT},
            {"role": "user", "content": f"### USER QUERY ###\n{query}\n### END USER QUERY ###"},
        ],
        temperature=1,
        max_completion_tokens=int(os.environ.get("MAX_DESCRIPTION_TOKENS", 1024)),
        response_format=response_format,
    )
    data = _parse_structured_content(response, label="description")
    if data is None:
        return ""
    title = data.get("title")
    author = data.get("author")
    shelves = data.get("shelves")
    description = data.get("description")
    if not all(isinstance(value, str) for value in [title, author, shelves, description]):
        print("Invalid field types in description response")
        return ""
    return f"TITLE: {title}\nAUTHOR: {author}\nSHELVES: {shelves}\nDESCRIPTION: {description}\n"


async def _rewrite_review(client: AsyncGroq, query: str) -> str:
    response_format = _json_schema_with_strict_mode(
        name="book_review",
        model=DEFAULT_QUERY_REWRITER_MODEL,
        schema=_BOOK_REVIEW_SCHEMA["json_schema"]["schema"],
    )
    response = await _create_structured_completion_with_retries(
        client,
        model=DEFAULT_QUERY_REWRITER_MODEL,
        messages=[
            {"role": "system", "content": BOOK_REVIEW_PROMPT},
            {"role": "user", "content": f"### USER QUERY ###\n{query}\n### END USER QUERY ###"},
        ],
        temperature=1,
        max_completion_tokens=int(os.environ.get("MAX_REVIEW_TOKENS", 150)),
        response_format=response_format,
    )
    data = _parse_structured_content(response, label="review")
    if data is None:
        return ""
    review = data.get("review")
    if not isinstance(review, str):
        print("Invalid field types in review response")
        return ""
    return review


async def rewrite_query(client: AsyncGroq, query: str) -> tuple[str, str]:
    description, review = await asyncio.gather(
        _rewrite_description(client, query),
        _rewrite_review(client, query),
    )
    return description, review

# ANSWERING USER QUERIES WITH A CHATBOT LLM

DEFAULT_CHATBOT_MODEL = os.environ.get("DEFAULT_CHATBOT_MODEL", "moonshotai/kimi-k2-instruct-0905")

_CHATBOT_SYSTEM_PROMPT = """\
You are a helpful book recommendation assistant. You are given a list of books \
and reader reviews retrieved from a database that are relevant to the user's query. \
Generate a friendly, conversational, yet brief response that recommends or discusses the most \
relevant books, citing specific books and reviews to support your points.

Rules:
1) Talk about the most relevant books first.
2) `referenced_book_ids` must be in the exact order the books are first mentioned in `response`.
3) Include each referenced book ID at most once.
"""

#4) Only include book IDs that appear in the provided relevant books list and are actually discussed.\

_CHATBOT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "chatbot_response",
        "schema": {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "referenced_book_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "referenced_review_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["response", "referenced_book_ids", "referenced_review_ids"],
            "additionalProperties": False,
        },
    },
}


async def generate_response(
    client: AsyncGroq,
    query: str,
    books: list[dict],    # each: {"book_id": int, "description": str}
    reviews: list[dict],  # each: {"review_id": int, "book_title": str, "review": str}
) -> dict[str, Any]:
    """
    Generate a user-friendly response grounded in the retrieved books and reviews.
    """
    books_text = "\n\n".join(
        f"[book_id: {b['book_id']}]\n{b['description']}" for b in books
    )
    reviews_text = "\n\n".join(
        f"[review_id: {r['review_id']}] (book: \"{r['book_title']}\")\n{r['review']}"
        for r in reviews
    )
    user_message = (
        f"### USER QUERY ###\n{query}\n### END USER QUERY ###\n\n"
        f"### RELEVANT BOOKS ###\n{books_text}\n### END RELEVANT BOOKS ###\n\n"
        f"### RELEVANT REVIEWS ###\n{reviews_text}\n### END RELEVANT REVIEWS ###"
    )

    response_format = _json_schema_with_strict_mode(
        name="chatbot_response",
        model=DEFAULT_CHATBOT_MODEL,
        schema=_CHATBOT_RESPONSE_SCHEMA["json_schema"]["schema"],
    )
    response = await _create_structured_completion_with_retries(
        client,
        model=DEFAULT_CHATBOT_MODEL,
        messages=[
            {"role": "system", "content": _CHATBOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=float(os.environ.get("CHATBOT_TEMPERATURE", 0.7)),
        max_completion_tokens=int(os.environ.get("MAX_CHATBOT_TOKENS", 1024)),
        response_format=response_format,
    )
    data = _parse_structured_content(response, label="chatbot")
    if data is None:
        return {
            "response": "An error occurred while generating the response.",
            "referenced_book_ids": [],
            "referenced_review_ids": [],
        }

    response_text = data.get("response")
    referenced_book_ids = data.get("referenced_book_ids")
    referenced_review_ids = data.get("referenced_review_ids")
    if (
        not isinstance(response_text, str)
        or not isinstance(referenced_book_ids, list)
        or not isinstance(referenced_review_ids, list)
    ):
        print("Invalid field types in chatbot response")
        return {
            "response": "An error occurred while generating the response.",
            "referenced_book_ids": [],
            "referenced_review_ids": [],
        }

    normalized_book_ids: list[int] = []
    for value in referenced_book_ids:
        try:
            normalized_book_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    normalized_review_ids: list[int] = []
    for value in referenced_review_ids:
        try:
            normalized_review_ids.append(int(value))
        except (TypeError, ValueError):
            continue

    return {
        "response": response_text,
        "referenced_book_ids": normalized_book_ids,
        "referenced_review_ids": normalized_review_ids,
    }
