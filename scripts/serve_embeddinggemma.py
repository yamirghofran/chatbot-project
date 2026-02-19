from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import numpy as np
from pathlib import Path

# Add project root to path for direct script execution.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bookdb.models.embedding_inference import (
    cosine_similarity,
    detect_inference_device,
    encode_texts,
    load_embedding_model,
    resolve_artifact_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve finetuned EmbeddingGemma artifacts as a REST API.",
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
        help="Explicit model directory. If set, it overrides artifact auto-resolution.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (cpu/cuda/mps). Auto-detects when omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable autoreload in development.",
    )
    return parser.parse_args()


def _import_fastapi_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field

        return uvicorn, FastAPI, HTTPException, BaseModel, Field
    except Exception as exc:
        raise RuntimeError(
            "FastAPI runtime dependencies are missing. Install with:\n"
            "  uv add fastapi uvicorn"
        ) from exc


def main() -> None:
    args = parse_args()
    uvicorn, FastAPI, HTTPException, BaseModel, Field = _import_fastapi_runtime()
    resolved_device = detect_inference_device(args.device)

    model_path, manifest = resolve_artifact_model_path(
        artifact_root=args.artifact_root,
        model_path=args.model_path,
    )
    model = load_embedding_model(
        model_path=model_path,
        manifest=manifest,
        device=resolved_device,
    )

    class EmbedRequest(BaseModel):
        texts: list[str] = Field(..., min_length=1)
        normalize_embeddings: bool = True
        batch_size: Optional[int] = None

    class SimilarityRequest(BaseModel):
        query: str
        candidates: list[str] = Field(..., min_length=1)
        top_k: Optional[int] = None
        normalize_embeddings: bool = True
        batch_size: Optional[int] = None

    app = FastAPI(title="EmbeddingGemma Inference API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": str(model_path),
            "device": resolved_device,
            "default_batch_size": int(args.batch_size),
        }

    @app.post("/embed")
    def embed(request: EmbedRequest) -> dict[str, Any]:
        batch_size = max(1, int(request.batch_size or args.batch_size))
        embeddings = encode_texts(
            model=model,
            texts=request.texts,
            normalize_embeddings=request.normalize_embeddings,
            batch_size=batch_size,
        )
        return {
            "model_path": str(model_path),
            "shape": [int(x) for x in embeddings.shape],
            "embeddings": embeddings.tolist(),
        }

    @app.post("/similarity")
    def similarity(request: SimilarityRequest) -> dict[str, Any]:
        if request.top_k is not None and int(request.top_k) <= 0:
            raise HTTPException(status_code=400, detail="top_k must be > 0")

        batch_size = max(1, int(request.batch_size or args.batch_size))
        texts = [request.query] + request.candidates
        embeddings = encode_texts(
            model=model,
            texts=texts,
            normalize_embeddings=request.normalize_embeddings,
            batch_size=batch_size,
        )

        query_embedding = np.asarray(embeddings[0], dtype=np.float32)
        candidate_embeddings = np.asarray(embeddings[1:], dtype=np.float32)
        scores = cosine_similarity(query_embedding, candidate_embeddings)

        sorted_indices = np.argsort(-scores)
        if request.top_k is not None:
            sorted_indices = sorted_indices[: int(request.top_k)]

        results = []
        for rank, index in enumerate(sorted_indices, start=1):
            idx = int(index)
            results.append(
                {
                    "rank": rank,
                    "text": request.candidates[idx],
                    "score": float(scores[idx]),
                }
            )

        return {
            "query": request.query,
            "model_path": str(model_path),
            "count": len(results),
            "results": results,
        }

    uvicorn.run(app, host=str(args.host), port=int(args.port), reload=bool(args.reload))


if __name__ == "__main__":
    main()
