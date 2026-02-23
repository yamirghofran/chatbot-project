---
icon: lucide/message-square
---

# Chatbots & RAG

BookDB includes LLM-powered chatbot capabilities for natural language book discovery. The system uses Retrieval-Augmented Generation (RAG) to ground responses in actual book data.

## Overview

The chatbot pipeline:

1. **Query Rewriting**: Transform user query into book description and review format
2. **Embedding**: Generate embeddings for rewritten queries
3. **Retrieval**: Find similar books and reviews from Qdrant
4. **Generation**: LLM generates response grounded in retrieved content

## Components

### Query Rewriter

Location: `bookdb/models/chatbot_llm.py`

The query rewriter transforms natural language queries into formats that match the embedding space:

- **Description Rewrite**: Creates a synthetic book description (title, author, shelves, description)
- **Review Rewrite**: Creates a synthetic review matching user intent

```python
from bookdb.models.chatbot_llm import create_groq_client_sync, rewrite_query_sync

client = create_groq_client_sync()
description, review = rewrite_query_sync(client, "I want a sci-fi book with time travel")

# description: "TITLE: The Time Machine\nAUTHOR: H.G. Wells\nSHELVES: sci-fi, time-travel\nDESCRIPTION: ..."
# review: "I loved how this book explored the consequences of..."
```

### Response Generator

Generates user-friendly responses grounded in retrieved books and reviews:

```python
from bookdb.models.chatbot_llm import generate_response_sync

response = generate_response_sync(
    client=client,
    query="I want a sci-fi book with time travel",
    books=[
        {"book_id": 123, "description": "TITLE: The Time Machine\n..."}
    ],
    reviews=[
        {"review_id": 456, "book_title": "The Time Machine", "review": "..."}
    ]
)

# Returns:
# {
#     "response": "I'd recommend 'The Time Machine' by H.G. Wells...",
#     "referenced_book_ids": [123],
#     "referenced_review_ids": [456]
# }
```

### RAG Book Search

Location: `chatbots/backend/rag_books.py`

Searches for books using vector similarity:

```python
from chatbots.backend.rag_books import search_books

results = search_books("A mystery novel set in Victorian London", top_k=10)
```

### RAG Review Search

Location: `chatbot/backend/rag_reviews.py`

Searches for similar reviews:

```python
from chatbot.backend.rag_reviews import search_reviews

reviews = search_reviews("I loved the plot twist at the end", top_k=10)
```

## API Integration

The chatbot is integrated into the `/books/search` endpoint:

- When keyword search returns no results, the AI pipeline is triggered
- Query is rewritten, embedded, and used for vector search
- LLM generates narrative response with book recommendations

**Response Format:**

```json
{
  "directHit": null,
  "keywordResults": [],
  "aiNarrative": "Based on your interest in time travel sci-fi, I'd recommend...",
  "aiBooks": [
    {"id": 123, "title": "The Time Machine", ...},
    ...
  ]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Groq API key (required) |
| `DEFAULT_QUERY_REWRITER_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Model for query rewriting |
| `DEFAULT_CHATBOT_MODEL` | `moonshotai/kimi-k2-instruct-0905` | Model for response generation |
| `CHATBOT_TOP_K` | 20 | Number of books to retrieve |
| `CHATBOT_MAX_REVIEWS` | 30 | Max reviews to include in context |
| `CHATBOT_MAX_BOOKS` | 6 | Max books in AI response |
| `CHATBOT_TEMPERATURE` | 0.7 | LLM temperature |
| `MAX_DESCRIPTION_TOKENS` | 1024 | Max tokens for description rewrite |
| `MAX_REVIEW_TOKENS` | 150 | Max tokens for review rewrite |
| `MAX_CHATBOT_TOKENS` | 1024 | Max tokens for response |

## Testing

Try the chatbot locally:

```bash
uv run python scripts/try_chatbot_llm.py
```

## Prompt Engineering

### Query Rewrite Prompts

**Book Description:**
```
You're a prompt re-writer. Your job is to rewrite user's queries 
to look like a book description. Make up the title, author, shelves, 
and description that would fit best the user's query.
```

**Book Review:**
```
You're a prompt re-writer. Your job is to rewrite user's queries 
to look like a book review. Make up a human review that would fit 
best the user's query, make it sound natural and engaging...
```

### Response Generation Prompt

```
You are a helpful book recommendation assistant. You are given a list 
of books and reader reviews retrieved from a database that are relevant 
to the user's query. Generate a friendly, conversational, yet brief 
response that recommends or discusses the most relevant books...

Rules:
1) Talk about the most relevant books first.
2) `referenced_book_ids` must be in the exact order mentioned.
3) Include each referenced book ID at most once.
```
