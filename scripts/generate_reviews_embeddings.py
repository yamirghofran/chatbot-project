from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path for direct script execution.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bookdb.models.embedding_inference import (  # noqa: E402
    detect_inference_device,
    encode_texts,
    load_embedding_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate review embeddings parquet using the base EmbeddingGemma model."
    )
    parser.add_argument(
        "--model-path",
        default="unsloth/embeddinggemma-300m",
        help="HF model id or local model directory.",
    )
    parser.add_argument(
        "--reviews-path",
        default="data/3_goodreads_reviews_dedup_clean.parquet",
        help="Input parquet with review columns and review text.",
    )
    parser.add_argument(
        "--output-path",
        default="data/reviews_base_embeddings.parquet",
        help="Output parquet file path.",
    )
    parser.add_argument("--review-id-column", default="review_id")
    parser.add_argument("--user-id-column", default="user_id")
    parser.add_argument("--book-id-column", default="book_id")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--review-text-column", default="review_text")
    parser.add_argument("--embedding-column", default="embedding")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--parquet-read-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Token truncation length applied at model level when supported. "
            "Set 0 to keep model default."
        ),
    )
    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--oom-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On OOM, retry by reducing encode batch size (32->16->8->...).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to process. 0 means all rows.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (cpu/cuda/mps). Auto-detects when omitted.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
        help="Log progress every N processed batches. Set 0 to disable batch-based logs.",
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=15.0,
        help="Log progress at least every N seconds. Set 0 to disable time-based logs.",
    )
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, reviews_path: Path, output_path: Path) -> None:
    if not reviews_path.exists():
        raise FileNotFoundError(f"Input reviews parquet not found: {reviews_path}")
    if output_path.name in {"", "."} or output_path.is_dir():
        raise ValueError("Output path must be a parquet file path, not a directory.")
    if int(args.encode_batch_size) <= 0:
        raise ValueError("--encode-batch-size must be > 0")
    if int(args.parquet_read_batch_size) <= 0:
        raise ValueError("--parquet-read-batch-size must be > 0")
    if int(args.max_seq_length) < 0:
        raise ValueError("--max-seq-length must be >= 0")
    if int(args.max_rows) < 0:
        raise ValueError("--max-rows must be >= 0")
    if not str(args.embedding_column).strip():
        raise ValueError("--embedding-column cannot be empty")
    if int(args.log_every_batches) < 0:
        raise ValueError("--log-every-batches must be >= 0")
    if float(args.log_every_seconds) < 0:
        raise ValueError("--log-every-seconds must be >= 0")

    required_columns = [
        str(args.review_id_column).strip(),
        str(args.user_id_column).strip(),
        str(args.book_id_column).strip(),
        str(args.rating_column).strip(),
        str(args.review_text_column).strip(),
    ]
    if any(not col for col in required_columns):
        raise ValueError("Column arguments cannot be empty.")

    schema_names = set(pq.read_schema(str(reviews_path)).names)
    missing = [col for col in required_columns if col not in schema_names]
    if missing:
        raise ValueError(
            f"Missing required columns in input parquet: {missing}. "
            f"Available columns: {sorted(schema_names)}"
        )


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message


def _clear_torch_cache() -> None:
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    reviews_path = Path(args.reviews_path).expanduser()
    output_path = Path(args.output_path).expanduser()
    _validate_args(args, reviews_path=reviews_path, output_path=output_path)

    review_id_column = str(args.review_id_column).strip()
    user_id_column = str(args.user_id_column).strip()
    book_id_column = str(args.book_id_column).strip()
    rating_column = str(args.rating_column).strip()
    review_text_column = str(args.review_text_column).strip()
    embedding_column = str(args.embedding_column).strip()
    input_columns = [
        review_id_column,
        user_id_column,
        book_id_column,
        rating_column,
        review_text_column,
    ]

    resolved_device = detect_inference_device(args.device)
    model = load_embedding_model(
        model_path=str(args.model_path),
        manifest=None,
        device=resolved_device,
    )
    if int(args.max_seq_length) > 0 and hasattr(model, "max_seq_length"):
        model.max_seq_length = int(args.max_seq_length)

    active_encode_batch_size = int(args.encode_batch_size)
    logger.info("Starting review embedding generation")
    logger.info("Model path: %s", args.model_path)
    logger.info("Runtime device: %s", resolved_device)
    if hasattr(model, "max_seq_length"):
        logger.info("Model max_seq_length: %s", model.max_seq_length)
    logger.info("Initial encode batch size: %d", active_encode_batch_size)
    logger.info("Input parquet: %s", reviews_path)
    logger.info("Output parquet: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    input_reader = pq.ParquetFile(str(reviews_path))
    output_writer: pq.ParquetWriter | None = None

    rows_written = 0
    rows_skipped_empty_text = 0
    embedding_dim: int | None = None
    batches_processed = 0
    started_at = time.perf_counter()
    last_log_at = started_at
    max_rows = int(args.max_rows)

    try:
        for input_record_batch in input_reader.iter_batches(
            columns=input_columns,
            batch_size=int(args.parquet_read_batch_size),
        ):
            if max_rows > 0 and rows_written >= max_rows:
                break

            batches_processed += 1
            batch_frame = pl.from_arrow(input_record_batch).with_columns(
                pl.col(review_text_column)
                .cast(pl.String, strict=False)
                .fill_null("")
                .str.strip_chars()
                .alias(review_text_column),
            )
            batch_frame = batch_frame.filter(pl.col(review_text_column).str.len_chars() > 0)

            if max_rows > 0:
                remaining_rows_allowed = max_rows - rows_written
                if remaining_rows_allowed <= 0:
                    break
                if batch_frame.height > remaining_rows_allowed:
                    batch_frame = batch_frame.head(remaining_rows_allowed)

            kept_rows = int(batch_frame.height)
            rows_skipped_empty_text += int(input_record_batch.num_rows) - kept_rows
            if kept_rows == 0:
                continue

            batch_texts = batch_frame.get_column(review_text_column).to_list()
            current_batch_size = active_encode_batch_size
            while True:
                try:
                    batch_embeddings = encode_texts(
                        model=model,
                        texts=batch_texts,
                        normalize_embeddings=bool(args.normalize_embeddings),
                        batch_size=current_batch_size,
                    )
                    active_encode_batch_size = current_batch_size
                    break
                except RuntimeError as exc:
                    if not bool(args.oom_retry) or not _is_oom_error(exc):
                        raise
                    if current_batch_size <= 1:
                        raise RuntimeError(
                            "Out of memory even at encode batch size 1. "
                            "Try a smaller --max-seq-length (for example 256) or --device cpu."
                        ) from exc
                    next_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        "OOM while encoding %d rows with encode_batch_size=%d. Retrying with %d.",
                        kept_rows,
                        current_batch_size,
                        next_batch_size,
                    )
                    _clear_torch_cache()
                    current_batch_size = next_batch_size
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)

            if embedding_dim is None:
                embedding_dim = int(batch_embeddings.shape[1])

            output_batch = pa.table(
                {
                    "review_id": pa.array(
                        batch_frame.get_column(review_id_column).cast(pl.String).fill_null("").to_list(),
                        type=pa.string(),
                    ),
                    "user_id": pa.array(
                        batch_frame.get_column(user_id_column).cast(pl.Int64, strict=False).to_list(),
                        type=pa.int64(),
                    ),
                    "book_id": pa.array(
                        batch_frame.get_column(book_id_column).cast(pl.Int64, strict=False).to_list(),
                        type=pa.int64(),
                    ),
                    "rating": pa.array(
                        batch_frame.get_column(rating_column).cast(pl.Int64, strict=False).to_list(),
                        type=pa.int64(),
                    ),
                    embedding_column: pa.array(
                        batch_embeddings.tolist(),
                        type=pa.list_(pa.float32(), embedding_dim),
                    ),
                }
            )

            if output_writer is None:
                output_writer = pq.ParquetWriter(
                    str(temp_output_path),
                    output_batch.schema,
                    compression=str(args.compression),
                )
            output_writer.write_table(output_batch)
            rows_written += kept_rows

            now = time.perf_counter()
            should_log_batch = int(args.log_every_batches) > 0 and (
                batches_processed % int(args.log_every_batches) == 0
            )
            should_log_time = float(args.log_every_seconds) > 0 and (
                (now - last_log_at) >= float(args.log_every_seconds)
            )
            if should_log_batch or should_log_time:
                elapsed = now - started_at
                rows_per_sec = rows_written / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Progress: batches=%d rows_written=%d rows_skipped=%d elapsed=%.1fs rows_per_sec=%.1f",
                    batches_processed,
                    rows_written,
                    rows_skipped_empty_text,
                    elapsed,
                    rows_per_sec,
                )
                last_log_at = now
    finally:
        if output_writer is not None:
            output_writer.close()

    if rows_written == 0:
        raise RuntimeError("No rows were encoded. Check input data and filters.")

    if output_path.exists():
        output_path.unlink()
    temp_output_path.replace(output_path)

    elapsed_seconds = time.perf_counter() - started_at
    rows_per_sec = rows_written / elapsed_seconds if elapsed_seconds > 0 else 0.0
    logger.info(
        "Completed: rows_written=%d rows_skipped=%d batches=%d elapsed=%.1fs rows_per_sec=%.1f",
        rows_written,
        rows_skipped_empty_text,
        batches_processed,
        elapsed_seconds,
        rows_per_sec,
    )
    summary = {
        "model_path": str(args.model_path),
        "device": resolved_device,
        "input_path": str(reviews_path),
        "output_path": str(output_path),
        "embedding_column": embedding_column,
        "max_seq_length": (
            int(getattr(model, "max_seq_length")) if hasattr(model, "max_seq_length") else None
        ),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "final_encode_batch_size": active_encode_batch_size,
        "rows_written": rows_written,
        "rows_skipped_empty_text": rows_skipped_empty_text,
        "embedding_dim": embedding_dim,
        "batches_processed": batches_processed,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
