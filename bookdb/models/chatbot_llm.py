from groq import AsyncGroq
import os
import asyncio
import json
from typing import Optional

# QUERY REWRITING TO BOOK DESCRIPTIONS AND REVIEWS

DEFAULT_QUERY_REWRITER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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


def create_groq_client(api_key: Optional[str] = None) -> AsyncGroq:
    return AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"))


async def _rewrite_description(client: AsyncGroq, query: str) -> str:
    response = await client.chat.completions.create(
        model=DEFAULT_QUERY_REWRITER_MODEL,
        messages=[
            {"role": "system", "content": BOOK_DESCRIPTION_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=1,
        max_completion_tokens=1024,
        stream=False,
        response_format=_BOOK_DESCRIPTION_SCHEMA,
    )
    data = json.loads(response.choices[0].message.content)
    return f"TITLE: {data['title']}\nAUTHOR: {data['author']}\nSHELVES: {data['shelves']}\nDESCRIPTION: {data['description']}\n"


async def _rewrite_review(client: AsyncGroq, query: str) -> str:
    response = await client.chat.completions.create(
        model=DEFAULT_QUERY_REWRITER_MODEL,
        messages=[
            {"role": "system", "content": BOOK_REVIEW_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=1,
        max_completion_tokens=1024,
        stream=False,
        response_format=_BOOK_REVIEW_SCHEMA,
    )
    return json.loads(response.choices[0].message.content)["review"]


async def rewrite_query(client: AsyncGroq, query: str) -> tuple[str, str]:
    description, review = await asyncio.gather(
        _rewrite_description(client, query),
        _rewrite_review(client, query),
    )
    return description, review

# ANSWERING USER QUERIES WITH A CHATBOT LLM

DEFAULT_CHATBOT_MODEL = "moonshotai/kimi-k2-instruct-0905"

_CHATBOT_SYSTEM_PROMPT = """\
You are a helpful book recommendation assistant. You are given a list of books \
and reader reviews retrieved from a database that are relevant to the user's query. \
Generate a friendly, conversational response that recommends or discusses the most \
relevant books, citing specific books and reviews to support your points.\
"""

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
                    "items": {"type": "string"},
                },
                "referenced_review_ids": {
                    "type": "array",
                    "items": {"type": "string"},
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
    books: list[dict],    # each: {"book_id": str, "description": str}
    reviews: list[dict],  # each: {"review_id": str, "book_title": str, "review": str}
) -> dict:
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
        f"User query: {query}\n\n"
        f"Relevant books:\n{books_text}\n\n"
        f"Relevant reviews (may reference books not in the list above):\n{reviews_text}"
    )

    response = await client.chat.completions.create(
        model=DEFAULT_CHATBOT_MODEL,
        messages=[
            {"role": "system", "content": _CHATBOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        stream=False,
        response_format=_CHATBOT_RESPONSE_SCHEMA,
    )
    return json.loads(response.choices[0].message.content)
