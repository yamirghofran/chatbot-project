# BookDB Scripts

Utility scripts for data processing, model training, and system maintenance.

## Data Import

| Script | Description |
|--------|-------------|
| `import_goodreads_to_postgres.py` | Import Goodreads dataset into PostgreSQL |
| `import_tags_from_shelves.py` | Extract and import tags from user shelves |

## Embeddings & Vector DB

| Script | Description |
|--------|-------------|
| `generate_book_embeddings.py` | Generate vector embeddings for books |
| `ingest_books_embeddings_to_qdrant.py` | Upload embeddings to Qdrant |
| `check_qdrant_client.py` | Test Qdrant client connection |

## Model Training & Serving

| Script | Description |
|--------|-------------|
| `finetune_embeddinggemma.py` | Fine-tune EmbeddingGemma with LoRA |
| `serve_embeddinggemma.py` | Serve embedding model via FastAPI |
| `infer_embeddinggemma.py` | Run inference with embedding model |
| `train_bpr.py` | Train BPR collaborative filtering |
| `train_sar.py` | Train SAR recommendation model |
| `upload_model_to_huggingface.py` | Upload models to HuggingFace Hub |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `run_embeddinggemma_finetuning.sh` | Wrapper for embedding model training |
| `run_generate_embeddings.sh` | Wrapper for embedding generation |

## RAG & Chatbot

| Script | Description |
|--------|-------------|
| `rag_books_test.py` | Test RAG book search functionality |
| `try_chatbot_llm.py` | Test LLM chatbot integration |

## Utilities

| Script | Description |
|--------|-------------|
| `update_user_names.py` | Bulk update user display names |

## Usage

Most scripts are run from the project root:

```bash
# Example: Generate embeddings
python scripts/generate_book_embeddings.py

# Example: Fine-tune with custom args
python scripts/finetune_embeddinggemma.py --epochs 3 --batch-size 32

# Example: Import data
python scripts/import_goodreads_to_postgres.py
```

See individual script files for argument options and configuration requirements.
