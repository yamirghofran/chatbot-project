from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Usage:
# 1. Base (default, if nothing is provided):
#    `uv run python scripts/generate_finetuned_book_embeddings.py`
# 2. Finetuned books:
#    `uv run python scripts/generate_finetuned_book_embeddings.py --model-variant finetuned-books --artifact-root models/finetuned_embeddinggemma_books`

# Add project root to path for direct script execution.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bookdb.models.embedding_inference import (
    detect_inference_device,
    encode_texts,
    load_embedding_model,
    resolve_artifact_model_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate book embeddings parquet from base or finetuned EmbeddingGemma models."
        )
    )
    parser.add_argument(
        "--model-variant",
        choices=("base", "finetuned-books"),
        default="base",
        help=(
            "Model variant to use. `base` loads `unsloth/embeddinggemma-300m`; "
            "`finetuned-books` loads `<artifact-root>/merged_16bit`."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="models/finetuned_embeddinggemma_books",
        help=(
            "Artifact location. Accepts artifact root, merged_16bit/lora model dir, "
            "deployment_manifest.json path, file:// URI, or alias paths."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Explicit model directory or HF model id. If set, overrides "
            "--model-variant defaults."
        ),
    )
    parser.add_argument(
        "--texts-path",
        default="data/books_embedding_texts.parquet",
        help="Input parquet with columns book_id and book_embedding_text.",
    )
    parser.add_argument(
        "--output-path",
        default="data/books_finetuned_embeddings.parquet",
        help="Output parquet file path.",
    )
    parser.add_argument("--embedding-column", default="embedding")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--parquet-read-batch-size", type=int, default=4096)
    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def _validate_args(args: argparse.Namespace, texts_path: Path, output_path: Path) -> None:
    if not texts_path.exists():
        raise FileNotFoundError(f"Input texts parquet not found: {texts_path}")
    if output_path.name in {"", "."} or output_path.is_dir():
        raise ValueError("Output path must be a parquet file path, not a directory.")
    if int(args.encode_batch_size) <= 0:
        raise ValueError("--encode-batch-size must be > 0")
    if int(args.parquet_read_batch_size) <= 0:
        raise ValueError("--parquet-read-batch-size must be > 0")
    if int(args.max_rows) < 0:
        raise ValueError("--max-rows must be >= 0")
    if not str(args.embedding_column).strip():
        raise ValueError("--embedding-column cannot be empty")
    if int(args.log_every_batches) < 0:
        raise ValueError("--log-every-batches must be >= 0")
    if float(args.log_every_seconds) < 0:
        raise ValueError("--log-every-seconds must be >= 0")


def _resolve_model_selection(
    args: argparse.Namespace,
) -> tuple[str, Optional[dict[str, object]], str]:
    # Explicit model path takes precedence regardless of selected variant.
    if args.model_path is not None and str(args.model_path).strip():
        model_path, manifest = resolve_artifact_model_path(
            artifact_root=args.artifact_root,
            model_path=args.model_path,
        )
        return str(model_path), manifest, str(args.model_variant)

    variant = str(args.model_variant).strip().lower()
    if variant == "base":
        return "unsloth/embeddinggemma-300m", None, "base"

    if variant != "finetuned-books":
        raise ValueError(f"Unsupported --model-variant: {args.model_variant}")

    model_path, manifest = resolve_artifact_model_path(
        artifact_root=args.artifact_root,
        model_path="merged_16bit",
    )
    resolved_path = Path(model_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Expected finetuned model at "
            f"{resolved_path}. Provide --artifact-root with merged_16bit or use --model-path."
        )
    return str(resolved_path), manifest, "finetuned-books"


def main() -> None:
    args = parse_args()

    texts_path = Path(args.texts_path).expanduser()
    output_path = Path(args.output_path).expanduser()
    _validate_args(args, texts_path=texts_path, output_path=output_path)

    resolved_device = detect_inference_device(args.device)
    model_path, manifest, resolved_model_variant = _resolve_model_selection(args)
    model = load_embedding_model(
        model_path=model_path,
        manifest=manifest,
        device=resolved_device,
    )
    logger.info("Starting embedding generation")
    logger.info("Model variant: %s", resolved_model_variant)
    logger.info("Resolved model path: %s", model_path)
    logger.info("Runtime device: %s", resolved_device)
    logger.info("Input parquet: %s", texts_path)
    logger.info("Output parquet: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    input_reader = pq.ParquetFile(str(texts_path))
    output_writer: pq.ParquetWriter | None = None

    rows_written = 0
    rows_skipped_empty_text = 0
    embedding_dim: int | None = None
    batches_processed = 0
    started_at = time.perf_counter()
    last_log_at = started_at
    max_rows = int(args.max_rows)
    embedding_column = str(args.embedding_column).strip()

    try:
        for input_record_batch in input_reader.iter_batches(
            columns=["book_id", "book_embedding_text"],
            batch_size=int(args.parquet_read_batch_size),
        ):
            if max_rows > 0 and rows_written >= max_rows:
                break

            batches_processed += 1
            batch_frame = pl.from_arrow(input_record_batch).with_columns(
                pl.col("book_embedding_text")
                .fill_null("")
                .str.strip_chars()
                .alias("book_embedding_text")
            )
            batch_frame = batch_frame.filter(
                pl.col("book_embedding_text").str.len_chars() > 0
            )

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

            batch_book_ids = [str(x) if x is not None else "" for x in batch_frame["book_id"].to_list()]
            batch_texts = [str(x) for x in batch_frame["book_embedding_text"].to_list()]
            batch_embeddings = encode_texts(
                model=model,
                texts=batch_texts,
                normalize_embeddings=bool(args.normalize_embeddings),
                batch_size=int(args.encode_batch_size),
            )
            batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32, order="C")
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)

            if embedding_dim is None:
                embedding_dim = int(batch_embeddings.shape[1])

            output_batch = pa.table(
                {
                    "book_id": pa.array(batch_book_ids, type=pa.string()),
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
        "model_variant": resolved_model_variant,
        "artifact_root_input": str(args.artifact_root),
        "model_path": str(model_path),
        "device": resolved_device,
        "input_path": str(texts_path),
        "output_path": str(output_path),
        "embedding_column": embedding_column,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "rows_written": rows_written,
        "rows_skipped_empty_text": rows_skipped_empty_text,
        "embedding_dim": embedding_dim,
        "batches_processed": batches_processed,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
