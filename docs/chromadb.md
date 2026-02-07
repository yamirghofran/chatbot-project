# ChromaDB Integration Documentation

This document provides comprehensive documentation for the ChromaDB vector database integration in the BookDB project.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
  - [Client Management](#client-management)
  - [Collections](#collections)
  - [Schemas and Metadata](#schemas-and-metadata)
  - [CRUD Operations](#crud-operations)
  - [Embeddings](#embeddings)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

ChromaDB is an AI-native open-source vector database used in this project to store and query embeddings for books and user preferences. This enables semantic search and personalized recommendations based on vector similarity.

### Key Features

- **Vector Similarity Search**: Find similar books based on semantic content
- **Metadata Filtering**: Combine vector search with traditional filtering (genre, author, year)
- **Dual Collections**: Separate collections for books and users
- **Flexible Deployment**: Support for both embedded and server modes
- **Type-Safe**: Pydantic schemas for metadata validation

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     BookDB Application                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │   Client     │  │  Collections  │  │   Embeddings    │  │
│  │  Management  │  │   Manager     │  │    Service      │  │
│  └──────┬───────┘  └───────┬───────┘  └────────┬────────┘  │
│         │                  │                     │            │
│         └──────────────────┼─────────────────────┘            │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────┐   │
│  │              ChromaDB Client (Singleton)              │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   ChromaDB      │
                    │   (Docker)      │
                    │  Port: 8000     │
                    └─────────────────┘
```

### Data Model

```
ChromaDB Instance
├── books (Collection)
│   ├── Embeddings: Vector representations of book content
│   ├── Documents: Book descriptions/summaries
│   └── Metadata: title, author, genre, year, isbn, etc.
│
└── users (Collection)
    ├── Embeddings: Vector representations of user preferences
    ├── Documents: User preference summaries
    └── Metadata: user_id, num_books_read, favorite_genres, etc.
```

## Setup

### 1. Start ChromaDB Docker Container

```bash
# Start ChromaDB
make chroma-up

# Or manually
docker compose up -d chromadb
```

### 2. Verify ChromaDB is Running

```bash
# Check logs
make chroma-logs

# Or manually
docker compose logs chromadb
```

You should see output indicating ChromaDB is running on `http://localhost:8000`.

### 3. Initialize Collections

```python
from bookdb.vector_db import initialize_all_collections

# Initialize both books and users collections
manager = initialize_all_collections()
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ChromaDB Connection Mode
CHROMA_MODE=server          # 'server' or 'embedded'
CHROMA_HOST=localhost       # Server host
CHROMA_PORT=8000           # Server port

# Embedding Model (optional)
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default model
```

### Configuration Options

- **Server Mode** (default): Connects to ChromaDB Docker container
- **Embedded Mode**: Runs ChromaDB in-process (for development/testing)

```python
from bookdb.vector_db import ChromaDBConfig, get_chroma_client

# Server mode (default)
config = ChromaDBConfig(mode="server", host="localhost", port=8000)
client = get_chroma_client(config)

# Embedded mode
config = ChromaDBConfig(mode="embedded", persist_directory="./chroma_data")
client = get_chroma_client(config)
```

## Usage Guide

### Client Management

The ChromaDB client uses a singleton pattern to ensure only one connection is maintained.

```python
from bookdb.vector_db import get_chroma_client, reset_client, get_client_info

# Get client (creates connection on first call)
client = get_chroma_client()

# Subsequent calls return the same instance
client2 = get_chroma_client()
assert client is client2

# Get connection info
info = get_client_info()
print(f"Connected: {info['connected']}")
print(f"Mode: {info['mode']}")

# Reset client (useful for testing)
reset_client()
```

### Collections

#### Initialize Collections

```python
from bookdb.vector_db import CollectionManager, CollectionNames

# Create manager
manager = CollectionManager()

# Initialize all collections
manager.initialize_collections()

# Get specific collection
books_collection = manager.get_collection(CollectionNames.BOOKS)
users_collection = manager.get_collection(CollectionNames.USERS)

# Check collection status
exists = manager.collection_exists(CollectionNames.BOOKS)
count = manager.get_collection_count(CollectionNames.BOOKS)
print(f"Books collection has {count} items")
```

#### Convenience Functions

```python
from bookdb.vector_db import (
    initialize_all_collections,
    get_books_collection,
    get_users_collection,
)

# Quick initialization
manager = initialize_all_collections()

# Direct collection access
books = get_books_collection()
users = get_users_collection()
```

### Schemas and Metadata

#### Book Metadata

```python
from bookdb.vector_db import BookMetadata, validate_book_metadata

# Create book metadata
metadata = BookMetadata(
    title="The Great Gatsby",
    author="F. Scott Fitzgerald",
    genre="Fiction",
    publication_year=1925,
    isbn="978-0743273565",
    language="en",
    page_count=180,
    average_rating=4.5,
)

# Validate from dictionary
metadata_dict = {
    "title": "1984",
    "author": "George Orwell",
    "genre": "Dystopian",
    "publication_year": 1949,
}
metadata = validate_book_metadata(metadata_dict)

# Convert to dict for ChromaDB
metadata_for_chroma = metadata.model_dump()
```

#### User Metadata

```python
from bookdb.vector_db import UserMetadata, validate_user_metadata

# Create user metadata
metadata = UserMetadata(
    user_id=12345,
    num_books_read=42,
    favorite_genres="Fiction,Science Fiction,Mystery",
    average_rating_given=4.2,
    reading_level="advanced",
)

# Access metadata
print(f"User has read {metadata.num_books_read} books")
print(f"Favorites: {metadata.favorite_genres}")
```

### CRUD Operations

The `BaseVectorCRUD` class provides all standard database operations.

#### Adding Items

```python
from bookdb.vector_db import BaseVectorCRUD, get_books_collection, BookMetadata

# Get collection and create CRUD instance
collection = get_books_collection()
crud = BaseVectorCRUD(collection)

# Add single item
crud.add(
    id="book_123",
    document="A classic novel about the American Dream...",
    metadata=BookMetadata(
        title="The Great Gatsby",
        author="F. Scott Fitzgerald",
        genre="Fiction",
    ).model_dump(),
)

# Add multiple items in batch
crud.add_batch(
    ids=["book_1", "book_2", "book_3"],
    documents=[
        "Doc 1 content...",
        "Doc 2 content...",
        "Doc 3 content...",
    ],
    metadatas=[
        BookMetadata(title="Book 1", author="Author 1").model_dump(),
        BookMetadata(title="Book 2", author="Author 2").model_dump(),
        BookMetadata(title="Book 3", author="Author 3").model_dump(),
    ],
)
```

#### Retrieving Items

```python
# Get single item
item = crud.get("book_123")
if item:
    print(f"Title: {item['metadata']['title']}")
    print(f"Author: {item['metadata']['author']}")
    print(f"Document: {item['document']}")

# Get multiple items
items = crud.get_batch(["book_1", "book_2", "book_3"])
for item in items:
    print(item['metadata']['title'])

# Get all items with pagination
all_items = crud.get_all(limit=100, offset=0)
print(f"Retrieved {len(all_items)} items")

# Check if item exists
if crud.exists("book_123"):
    print("Book exists!")

# Count total items
total = crud.count()
print(f"Total books: {total}")
```

#### Updating Items

```python
# Update document and metadata
crud.update(
    id="book_123",
    document="Updated description...",
    metadata=BookMetadata(
        title="The Great Gatsby (Updated Edition)",
        author="F. Scott Fitzgerald",
        genre="Classic Fiction",
    ).model_dump(),
)

# Partial update (only metadata)
crud.update(
    id="book_123",
    metadata={"average_rating": 4.8},
)
```

#### Deleting Items

```python
# Delete single item
crud.delete("book_123")

# Delete multiple items
crud.delete_batch(["book_1", "book_2", "book_3"])
```

### Embeddings

The embedding service generates vector representations from text. *(Note: Implementation pending)*

```python
from bookdb.vector_db import EmbeddingService

# Initialize service (TODO: Implement)
service = EmbeddingService()

# Generate book embedding (TODO: Implement)
embedding = service.generate_book_embedding(
    title="The Great Gatsby",
    description="A novel about the American Dream...",
    author="F. Scott Fitzgerald",
    genre="Fiction",
)

# Generate user embedding (TODO: Implement)
user_embedding = service.generate_user_embedding(
    preferences="I enjoy science fiction and mystery novels",
    favorite_genres="Sci-Fi,Mystery,Thriller",
    reading_history=["Book 1", "Book 2", "Book 3"],
)

# Batch generation (TODO: Implement)
texts = ["Book 1 description", "Book 2 description", "Book 3 description"]
embeddings = service.generate_embeddings_batch(texts)
```

## Best Practices

### 1. Use Singleton Client

Always use `get_chroma_client()` to ensure you're using the same connection instance.

```python
# ✅ Good
from bookdb.vector_db import get_chroma_client
client = get_chroma_client()

# ❌ Avoid creating multiple instances
```

### 2. Validate Metadata

Use Pydantic schemas to validate metadata before adding to ChromaDB.

```python
# ✅ Good - validated
metadata = BookMetadata(title="Book", author="Author")
crud.add(id="book_1", document="...", metadata=metadata.model_dump())

# ❌ Bad - unvalidated dict
crud.add(id="book_1", document="...", metadata={"title": "Book"})
```

### 3. Use Batch Operations

For multiple items, use batch operations for better performance.

```python
# ✅ Good - single batch operation
crud.add_batch(ids=ids, documents=docs, metadatas=metas)

# ❌ Avoid - multiple single operations
for id, doc, meta in zip(ids, docs, metas):
    crud.add(id, doc, meta)
```

### 4. Handle Errors Gracefully

Always handle potential errors when working with the database.

```python
from bookdb.vector_db import BaseVectorCRUD

try:
    crud.add(id="book_123", document="...", metadata=meta)
except ValueError as e:
    print(f"Item already exists: {e}")
except Exception as e:
    print(f"Failed to add item: {e}")
```

### 5. Keep IDs Synchronized

Use the same IDs in ChromaDB as your PostgreSQL primary keys for easy cross-referencing.

```python
# ✅ Good - same ID as PostgreSQL
book_id = book.id  # From PostgreSQL
crud.add(id=f"book_{book_id}", document="...", metadata=meta)

# Later, easy to correlate
chroma_item = crud.get(f"book_{book_id}")
pg_book = db.query(Book).filter(Book.id == book_id).first()
```

### 6. Use Metadata for Filtering

Store searchable attributes in metadata for efficient filtering.

```python
metadata = BookMetadata(
    title="Book Title",
    author="Author Name",
    genre="Science Fiction",  # Filterable
    publication_year=2020,     # Filterable
    language="en",             # Filterable
)
```

### 7. Reset Client in Tests

Always reset the client in test fixtures to ensure clean state.

```python
import pytest
from bookdb.vector_db import reset_client

@pytest.fixture(autouse=True)
def reset_chroma_client():
    """Reset client before each test."""
    reset_client()
    yield
    reset_client()
```

## Troubleshooting

### ChromaDB Container Not Starting

**Problem**: `docker compose up -d chromadb` fails

**Solutions**:
1. Check if port 8000 is already in use:
   ```bash
   lsof -i :8000
   ```

2. Check Docker logs:
   ```bash
   docker compose logs chromadb
   ```

3. Reset and restart:
   ```bash
   make chroma-reset
   ```

### Connection Refused

**Problem**: `ConnectionError: Failed to connect to ChromaDB`

**Solutions**:
1. Verify ChromaDB is running:
   ```bash
   docker compose ps chromadb
   ```

2. Check environment variables:
   ```bash
   cat .env | grep CHROMA
   ```

3. Test connection manually:
   ```bash
   curl http://localhost:8000/api/v1
   ```

### Duplicate ID Error

**Problem**: `ValueError: Item with ID 'book_123' already exists`

**Solutions**:
1. Check if item exists before adding:
   ```python
   if not crud.exists("book_123"):
       crud.add(id="book_123", document="...", metadata=meta)
   ```

2. Use update instead of add:
   ```python
   crud.update(id="book_123", document="...", metadata=meta)
   ```

3. Delete and re-add:
   ```python
   crud.delete("book_123")
   crud.add(id="book_123", document="...", metadata=meta)
   ```

### Item Not Found

**Problem**: `crud.get("book_123")` returns `None`

**Solutions**:
1. Verify the ID is correct
2. Check if the collection was initialized:
   ```python
   manager.initialize_collections()
   ```

3. List all items to verify:
   ```python
   items = crud.get_all()
   print([item['id'] for item in items])
   ```

### Metadata Validation Error

**Problem**: `ValueError: Language code must be 2 characters`

**Solution**: Use proper metadata schemas:
```python
# ✅ Correct
metadata = BookMetadata(
    title="Book",
    author="Author",
    language="en",  # 2 characters
)

# ❌ Wrong
metadata = {"language": "eng"}  # 3 characters
```

### Collection Not Found

**Problem**: `ValueError: Collection 'books' does not exist`

**Solution**: Initialize collections first:
```python
from bookdb.vector_db import initialize_all_collections

# Initialize before using
manager = initialize_all_collections()

# Then get collections
books = manager.get_collection(CollectionNames.BOOKS)
```

### Performance Issues

**Problem**: Slow operations with large datasets

**Solutions**:
1. Use batch operations instead of loops
2. Add pagination when retrieving large result sets:
   ```python
   # Get in chunks
   page_size = 100
   offset = 0
   while True:
       items = crud.get_all(limit=page_size, offset=offset)
       if not items:
           break
       process(items)
       offset += page_size
   ```

3. Consider indexing strategies in ChromaDB configuration

### Clean Slate / Reset

**Problem**: Need to start fresh with empty collections

**Solution**:
```bash
# Reset ChromaDB completely
make chroma-reset

# Or reset specific collection in Python
manager.reset_collection(CollectionNames.BOOKS)
```

## Additional Resources

- [ChromaDB Official Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Project README](../README.md)
- [Project Structure](../AGENTS.md)

## Support

For issues or questions:
1. Check this documentation first
2. Review the test files for usage examples
3. Check ChromaDB logs: `make chroma-logs`
4. Open an issue in the project repository
