from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from pathlib import Path

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
        "--base-model-path",
        default="unsloth/embeddinggemma-300m",
        help=(
            "Base model directory or HF model id. Defaults to "
            "`unsloth/embeddinggemma-300m`."
        ),
    )
    parser.add_argument(
        "--default-model",
        choices=("finetuned", "base"),
        default="finetuned",
        help="Default model used by requests that omit `model`.",
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


def _import_fastapi_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import uvicorn
        from fastapi import Body, FastAPI, HTTPException

        return uvicorn, FastAPI, HTTPException, Body
    except Exception as exc:
        raise RuntimeError(
            "FastAPI runtime dependencies are missing. Install with:\n"
            "  uv add fastapi uvicorn"
        ) from exc


def _resolve_base_model_path(
    explicit_base_model_path: Optional[str],
) -> str:
    if explicit_base_model_path and str(explicit_base_model_path).strip():
        return str(explicit_base_model_path).strip()

    return "unsloth/embeddinggemma-300m"


def main() -> None:
    args = parse_args()
    uvicorn, FastAPI, HTTPException, Body = _import_fastapi_runtime()
    resolved_device = detect_inference_device(args.device)

    finetuned_model_path, manifest = resolve_artifact_model_path(
        artifact_root=args.artifact_root,
        model_path=args.model_path,
    )
    finetuned_model = load_embedding_model(
        model_path=finetuned_model_path,
        manifest=manifest,
        device=resolved_device,
    )
    base_model_path = _resolve_base_model_path(args.base_model_path)
    base_model = load_embedding_model(
        model_path=base_model_path,
        manifest=None,
        device=resolved_device,
    )

    models: dict[str, dict[str, Any]] = {
        "finetuned": {
            "model": finetuned_model,
            "model_path": str(finetuned_model_path),
        },
        "base": {
            "model": base_model,
            "model_path": str(base_model_path),
        },
    }

    app = FastAPI(title="EmbeddingGemma Inference API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "models": {
                "finetuned": str(finetuned_model_path),
                "base": str(base_model_path),
            },
            "default_model": str(args.default_model),
            "device": resolved_device,
            "default_batch_size": int(args.batch_size),
        }

    def _resolve_runtime_model(model_key: str) -> tuple[str, dict[str, Any]]:
        selected = str(model_key).strip().lower()
        if selected not in models:
            raise HTTPException(status_code=400, detail="model must be `finetuned` or `base`")
        return selected, models[selected]

    @app.post("/embed")
    def embed(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        texts_raw = request.get("texts")
        if not isinstance(texts_raw, list) or len(texts_raw) == 0:
            raise HTTPException(status_code=400, detail="texts must be a non-empty list of strings")
        texts = [str(x) for x in texts_raw]

        model_key = request.get("model", args.default_model)
        selected_key, selected_model = _resolve_runtime_model(str(model_key))

        normalize_embeddings = bool(request.get("normalize_embeddings", True))
        batch_size = max(1, int(request.get("batch_size", args.batch_size)))
        embeddings = encode_texts(
            model=selected_model["model"],
            texts=texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )
        return {
            "model": selected_key,
            "model_path": str(selected_model["model_path"]),
            "shape": [int(x) for x in embeddings.shape],
            "embeddings": embeddings.tolist(),
        }

    uvicorn.run(app, host=str(args.host), port=int(args.port), reload=bool(args.reload))


if __name__ == "__main__":
    main()
