# BookDB Chatbots

RAG (Retrieval-Augmented Generation) implementations for book discovery.

## Structure

```
chatbots/
└── backend/
    └── rag_books.py   # RAG book search using Qdrant vector DB
```

## RAG Book Search

The `backend/rag_books.py` module provides semantic book search using:

- **Vector Database:** Qdrant for similarity search
- **Embeddings:** Finetuned EmbeddingGemma model served via API endpoint
- **Metadata:** Book information from parquet files

### Usage

```python
from chatbots.backend.rag_books import search_books

results = search_books(
    query_text="A magical school adventure with friendship and mystery",
    top_k=10
)
```

### Configuration

The module requires:
- `QDRANT_URL` - Qdrant server URL
- `EMBEDDING_ENDPOINT` - Embedding API endpoint (default: `https://bookdb-models.up.railway.app/embed`)

### Testing

```bash
# From project root
python scripts/rag_books_test.py
```

### Requirements

- Running Qdrant instance with book embeddings
- Access to embedding endpoint
- Book data at `data/3_goodreads_books_with_metrics.parquet`

## Related Documentation

- [Embeddings Documentation](../docs/embeddings.md)
- [Qdrant Setup](../docs/qdrant.md)
- [Chatbots & RAG](../docs/chatbots.md)
