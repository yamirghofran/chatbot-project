---
icon: lucide/search
---

# Embedding Models

BookDB uses fine-tuned embedding models for semantic book search. The embeddings power vector similarity search in Qdrant.

## Overview

The embedding pipeline:

1. **Fine-tuning**: Adapt EmbeddingGemma to book descriptions and reviews
2. **Inference**: Generate embeddings for all books
3. **Ingestion**: Store embeddings in Qdrant
4. **Serving**: Real-time embedding service for queries

## Model Fine-Tuning

Location: `scripts/finetune_embeddinggemma.py`

Fine-tunes EmbeddingGemma on book description and review pairs.

### Training Data

- Book descriptions (title, author, description)
- User reviews (review text)
- Positive pairs: book-review matches from dataset

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `model_name` | Base model (default: google/embeddinggemma) |
| `max_seq_length` | Max sequence length |
| `batch_size` | Training batch size |
| `epochs` | Number of training epochs |
| `learning_rate` | Learning rate |
| `lora_r` | LoRA rank |
| `lora_alpha` | LoRA alpha |

### Running Fine-Tuning

```bash
# Using shell script
bash scripts/run_embeddinggemma_finetuning.sh

# Or directly
uv run python scripts/finetune_embeddinggemma.py \
    --model_name google/embeddinggemma \
    --output_dir models/embeddinggemma_finetuned \
    --epochs 3 \
    --batch_size 32
```

### Output Artifacts

- `models/embeddinggemma_finetuned/lora/` - LoRA adapter weights
- `models/embeddinggemma_finetuned/merged_16bit/` - Merged model for serving
- `models/embeddinggemma_finetuned/deployment_manifest.json` - Training metadata

## Inference

Location: `bookdb/models/embedding_inference.py`

### Loading Models

```python
from bookdb.models.embedding_inference import (
    load_embedding_model,
    encode_texts,
    detect_inference_device,
)

# Auto-detect best device (CUDA > MPS > CPU)
device = detect_inference_device()

# Load model (handles both LoRA and merged)
model, manifest = load_embedding_model("models/embeddinggemma_finetuned")
```

### Generating Embeddings

```python
# Batch encoding
texts = ["Book description 1", "Book description 2"]
embeddings = encode_texts(
    model,
    texts,
    normalize_embeddings=True,
    batch_size=32,
)
# Shape: (2, 768)
```

### Cosine Similarity

```python
from bookdb.models.embedding_inference import cosine_similarity

query_embedding = embeddings[0]
similarities = cosine_similarity(query_embedding, embeddings)
```

## Embedding Generation Script

Location: `scripts/generate_book_embeddings.py`

Generates embeddings for all books in the dataset:

```bash
uv run python scripts/generate_book_embeddings.py \
    --model_path models/embeddinggemma_finetuned \
    --input data/3_goodreads_books_with_metrics.parquet \
    --output data/book_embeddings.parquet \
    --batch_size 64
```

### Shell Script

```bash
bash scripts/run_generate_embeddings.sh
```

## Qdrant Ingestion

Location: `scripts/ingest_books_embeddings_to_qdrant.py`

Uploads embeddings to Qdrant:

```bash
uv run python scripts/ingest_books_embeddings_to_qdrant.py \
    --embeddings data/book_embeddings.parquet \
    --collection books \
    --batch_size 100
```

## Embedding Service

Location: `scripts/serve_embeddinggemma.py`

Serves the embedding model via HTTP API:

```bash
uv run python scripts/serve_embeddinggemma.py \
    --model_path models/embeddinggemma_finetuned \
    --host 0.0.0.0 \
    --port 8000
```

### API Endpoints

**POST /embed**

```json
{
  "texts": ["Book description 1", "Book description 2"],
  "model": "finetuned",
  "normalize_embeddings": true,
  "batch_size": 32
}
```

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

**GET /health**

Returns model status and device info.

## API Configuration

The FastAPI backend connects to the embedding service:

| Variable | Description |
|----------|-------------|
| `EMBEDDING_SERVICE_URL` | Service URL (e.g., http://localhost:8000) |
| `EMBEDDING_SERVICE_MODEL` | Model name (default: "finetuned") |
| `EMBEDDING_SERVICE_TIMEOUT_SECONDS` | Request timeout |
| `EMBEDDING_SERVICE_API_KEY` | Optional API key |

## Upload to HuggingFace

Location: `scripts/upload_model_to_huggingface.py`

Upload fine-tuned model to HuggingFace Hub:

```bash
uv run python scripts/upload_model_to_huggingface.py \
    --model_path models/embeddinggemma_finetuned \
    --repo_id your-username/bookdb-embeddinggemma
```

## Testing Inference

Quick inference test:

```bash
uv run python scripts/infer_embeddinggemma.py \
    --model_path models/embeddinggemma_finetuned \
    --text "A mystery novel set in Victorian London"
```
