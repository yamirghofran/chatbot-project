from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
from dotenv import load_dotenv
from qdrant_client.models import PointStruct

# Add project root to path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bookdb.vector_db import CollectionManager, CollectionNames, QdrantConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest book embeddings from Parquet into Qdrant with streaming reads "
            "and batched parallel uploads."
        )
    )
    parser.add_argument(
        "--input-path",
        default="data/books_texts_embeddings.parquet",
        help="Input parquet path with id/text/embedding columns.",
    )
    parser.add_argument(
        "--collection-name",
        choices=[name.value for name in CollectionNames],
        default=CollectionNames.BOOKS.value,
        help="Target Qdrant collection.",
    )
    parser.add_argument(
        "--id-column",
        default="book_id",
        help="Column used as Qdrant point id.",
    )
    parser.add_argument(
        "--text-column",
        default="book_embedding_text",
        help="Column used as payload.document.",
    )
    parser.add_argument(
        "--embedding-column",
        default="embedding",
        help="Column used as vector embedding.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=0,
        help=(
            "Expected vector size. 0 auto-detects from parquet and updates runtime "
            "Qdrant config for this script run."
        ),
    )
    parser.add_argument(
        "--parquet-read-batch-size",
        type=int,
        default=4096,
        help="Rows to read per parquet batch.",
    )
    parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=256,
        help="Points per network batch sent by qdrant-client upload_points.",
    )
    parser.add_argument(
        "--qdrant-parallel",
        type=int,
        default=4,
        help="Parallel upload workers used by qdrant-client upload_points.",
    )
    parser.add_argument(
        "--qdrant-upload-method",
        choices=["auto", "http", "grpc"],
        default="auto",
        help="Transport for qdrant-client upload_points.",
    )
    parser.add_argument(
        "--upload-api",
        choices=["upload_points", "upsert"],
        default="upload_points",
        help=(
            "Qdrant write API. `upload_points` is high-throughput; `upsert` avoids "
            "multiprocessing worker path and can be more stable on some setups."
        ),
    )
    parser.add_argument(
        "--allow-multiprocess-upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow qdrant-client multiprocessing uploads on macOS when "
            "--qdrant-parallel > 1."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per failed upload batch.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Wait for each upload operation to be fully applied before moving on. "
            "Disable for higher throughput."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many valid rows before ingestion begins.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum valid rows to ingest after offset. 0 ingests all rows.",
    )
    parser.add_argument(
        "--reset-collection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete and recreate the target collection before ingestion.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=20,
        help="Emit progress log every N parquet batches (0 disables).",
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=15.0,
        help="Emit progress log at least every N seconds (0 disables).",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    if int(args.parquet_read_batch_size) <= 0:
        raise ValueError("--parquet-read-batch-size must be > 0")
    if int(args.qdrant_batch_size) <= 0:
        raise ValueError("--qdrant-batch-size must be > 0")
    if int(args.qdrant_parallel) <= 0:
        raise ValueError("--qdrant-parallel must be > 0")
    if int(args.max_retries) < 0:
        raise ValueError("--max-retries must be >= 0")
    if int(args.offset) < 0:
        raise ValueError("--offset must be >= 0")
    if int(args.max_rows) < 0:
        raise ValueError("--max-rows must be >= 0")
    if int(args.vector_size) < 0:
        raise ValueError("--vector-size must be >= 0")

    if not str(args.id_column).strip():
        raise ValueError("--id-column cannot be empty")
    if not str(args.text_column).strip():
        raise ValueError("--text-column cannot be empty")
    if not str(args.embedding_column).strip():
        raise ValueError("--embedding-column cannot be empty")

    if int(args.log_every_batches) < 0:
        raise ValueError("--log-every-batches must be >= 0")
    if float(args.log_every_seconds) < 0:
        raise ValueError("--log-every-seconds must be >= 0")


def _detect_embedding_size(
    parquet_file: pq.ParquetFile,
    embedding_column: str,
    probe_batch_size: int = 2048,
) -> int:
    for batch in parquet_file.iter_batches(
        columns=[embedding_column],
        batch_size=probe_batch_size,
    ):
        values = batch.column(0).to_pylist()
        for embedding in values:
            if embedding is None:
                continue
            return int(len(embedding))

    raise ValueError(
        f"Unable to detect vector size from '{embedding_column}' because no non-null embeddings were found."
    )


def _coerce_point_id(value: Any) -> int | str:
    if isinstance(value, bool):
        raise ValueError("Boolean ids are not supported")

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Empty id value")

    try:
        return int(text)
    except ValueError:
        return text


def _build_points(
    batch_frame: pl.DataFrame,
    id_column: str,
    text_column: str,
    embedding_column: str,
) -> tuple[list[PointStruct], int]:
    points: list[PointStruct] = []
    id_conversion_errors = 0

    ids = batch_frame.get_column(id_column).to_list()
    texts = batch_frame.get_column(text_column).to_list()
    embeddings = batch_frame.get_column(embedding_column).to_list()

    for row_id, text, embedding in zip(ids, texts, embeddings):
        try:
            point_id = _coerce_point_id(row_id)
        except ValueError:
            id_conversion_errors += 1
            continue

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document": text or "",
                },
            )
        )

    return points, id_conversion_errors


def _log_progress(
    *,
    parquet_batches: int,
    source_rows_scanned: int,
    ingested_rows: int,
    skipped_invalid_rows: int,
    skipped_offset_rows: int,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    ingest_rps = ingested_rows / elapsed if elapsed > 0 else 0.0
    scan_rps = source_rows_scanned / elapsed if elapsed > 0 else 0.0
    logger.info(
        (
            "Progress: batches=%d source_rows=%d ingested=%d skipped_invalid=%d "
            "skipped_offset=%d elapsed=%.1fs ingest_rows_per_sec=%.1f scan_rows_per_sec=%.1f"
        ),
        parquet_batches,
        source_rows_scanned,
        ingested_rows,
        skipped_invalid_rows,
        skipped_offset_rows,
        elapsed,
        ingest_rps,
        scan_rps,
    )


def _chunk_points(points: list[PointStruct], chunk_size: int) -> list[list[PointStruct]]:
    if chunk_size <= 0:
        return [points]
    return [points[i : i + chunk_size] for i in range(0, len(points), chunk_size)]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    input_path = Path(args.input_path).expanduser()
    _validate_args(args, input_path=input_path)

    reader = pq.ParquetFile(str(input_path))

    detected_vector_size = _detect_embedding_size(
        parquet_file=reader,
        embedding_column=str(args.embedding_column),
    )
    expected_vector_size = (
        int(args.vector_size)
        if int(args.vector_size) > 0
        else int(detected_vector_size)
    )
    if int(args.vector_size) > 0 and int(args.vector_size) != int(detected_vector_size):
        raise ValueError(
            f"--vector-size={args.vector_size} does not match detected parquet "
            f"embedding size={detected_vector_size}."
        )

    config = QdrantConfig.from_env()
    config.vector_size = expected_vector_size
    config.validate()

    manager = CollectionManager(config=config, vector_size=expected_vector_size)
    manager.initialize_collections()

    collection_name = str(args.collection_name).strip()
    collection_enum = CollectionNames(collection_name)
    if bool(args.reset_collection):
        logger.info("Resetting collection '%s' before ingestion.", collection_name)
        manager.reset_collection(collection_enum)
    manager.get_collection(collection_enum)
    client = manager.client

    qdrant_method = None if str(args.qdrant_upload_method) == "auto" else str(
        args.qdrant_upload_method
    )
    effective_parallel = int(args.qdrant_parallel)
    if (
        str(args.upload_api) == "upload_points"
        and sys.platform == "darwin"
        and effective_parallel > 1
        and not bool(args.allow_multiprocess_upload)
    ):
        logger.warning(
            "macOS detected: forcing --qdrant-parallel from %d to 1 for stability. "
            "Use --allow-multiprocess-upload to override.",
            effective_parallel,
        )
        effective_parallel = 1

    logger.info("Starting Qdrant ingestion")
    logger.info("Input parquet: %s", input_path)
    logger.info("Target collection: %s", collection_name)
    logger.info("Detected vector size: %d", detected_vector_size)
    logger.info("Effective vector size: %d", expected_vector_size)
    logger.info(
        "Subset controls: offset=%d max_rows=%d",
        int(args.offset),
        int(args.max_rows),
    )
    logger.info(
        "Upload tuning: api=%s read_batch=%d qdrant_batch=%d parallel=%d method=%s wait=%s retries=%d",
        str(args.upload_api),
        int(args.parquet_read_batch_size),
        int(args.qdrant_batch_size),
        effective_parallel,
        str(args.qdrant_upload_method),
        bool(args.wait),
        int(args.max_retries),
    )

    source_rows_scanned = 0
    ingested_rows = 0
    skipped_invalid_rows = 0
    skipped_offset_rows = 0
    parquet_batches = 0
    remaining_offset = int(args.offset)
    max_rows = int(args.max_rows)
    started_at = time.perf_counter()
    last_log_at = started_at

    for record_batch in reader.iter_batches(
        columns=[args.id_column, args.text_column, args.embedding_column],
        batch_size=int(args.parquet_read_batch_size),
    ):
        if max_rows > 0 and ingested_rows >= max_rows:
            break

        parquet_batches += 1
        source_rows_scanned += int(record_batch.num_rows)

        batch_frame = pl.from_arrow(record_batch).with_columns(
            pl.col(args.text_column).cast(pl.String).fill_null("").alias(args.text_column)
        )
        valid_frame = batch_frame.filter(
            pl.col(args.id_column).is_not_null()
            & pl.col(args.embedding_column).is_not_null()
            & (pl.col(args.embedding_column).list.len() == expected_vector_size)
        )
        skipped_invalid_rows += int(batch_frame.height) - int(valid_frame.height)

        if valid_frame.is_empty():
            continue

        if remaining_offset > 0:
            skip_now = min(remaining_offset, int(valid_frame.height))
            valid_frame = valid_frame.slice(skip_now)
            remaining_offset -= skip_now
            skipped_offset_rows += skip_now
            if valid_frame.is_empty():
                continue

        if max_rows > 0:
            remaining_rows = max_rows - ingested_rows
            if remaining_rows <= 0:
                break
            if int(valid_frame.height) > remaining_rows:
                valid_frame = valid_frame.head(remaining_rows)

        points, id_conversion_errors = _build_points(
            batch_frame=valid_frame,
            id_column=args.id_column,
            text_column=args.text_column,
            embedding_column=args.embedding_column,
        )
        skipped_invalid_rows += id_conversion_errors
        if not points:
            continue

        if str(args.upload_api) == "upload_points":
            client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=int(args.qdrant_batch_size),
                parallel=effective_parallel,
                method=qdrant_method,
                max_retries=int(args.max_retries),
                wait=bool(args.wait),
            )
        else:
            for point_chunk in _chunk_points(points, int(args.qdrant_batch_size)):
                client.upsert(
                    collection_name=collection_name,
                    points=point_chunk,
                    wait=bool(args.wait),
                )
        ingested_rows += len(points)

        now = time.perf_counter()
        should_log_batch = parquet_batches == 1 or (
            int(args.log_every_batches) > 0
            and (
            parquet_batches % int(args.log_every_batches) == 0
            )
        )
        should_log_time = float(args.log_every_seconds) > 0 and (
            (now - last_log_at) >= float(args.log_every_seconds)
        )
        if should_log_batch or should_log_time:
            _log_progress(
                parquet_batches=parquet_batches,
                source_rows_scanned=source_rows_scanned,
                ingested_rows=ingested_rows,
                skipped_invalid_rows=skipped_invalid_rows,
                skipped_offset_rows=skipped_offset_rows,
                started_at=started_at,
            )
            last_log_at = now

    elapsed = time.perf_counter() - started_at
    ingest_rps = ingested_rows / elapsed if elapsed > 0 else 0.0

    collection_count = None
    if bool(args.wait):
        collection_count = manager.get_collection_count(collection_enum)

    summary = {
        "input_path": str(input_path),
        "collection_name": collection_name,
        "vector_size_detected": detected_vector_size,
        "vector_size_effective": expected_vector_size,
        "source_rows_scanned": source_rows_scanned,
        "ingested_rows": ingested_rows,
        "skipped_invalid_rows": skipped_invalid_rows,
        "skipped_offset_rows": skipped_offset_rows,
        "parquet_batches_processed": parquet_batches,
        "offset": int(args.offset),
        "max_rows": int(args.max_rows),
        "elapsed_seconds": round(elapsed, 3),
        "ingest_rows_per_sec": round(ingest_rps, 3),
        "collection_count_if_wait_true": collection_count,
    }
    logger.info("Completed ingestion")
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
