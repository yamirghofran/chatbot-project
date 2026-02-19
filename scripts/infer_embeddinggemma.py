from __future__ import annotations

import argparse
import json
import sys
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


def _load_texts_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("texts"), list):
        return [str(x) for x in payload["texts"]]

    raise ValueError("Input JSON must be a list of strings or {'texts': [...]} format.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with finetuned EmbeddingGemma artifacts.",
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
        "--text",
        action="append",
        default=[],
        help="Text to embed. Provide multiple --text values for batching.",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="JSON file with either ['text1', ...] or {'texts': ['text1', ...]}.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (cpu/cuda/mps). Auto-detects when omitted.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for JSON output. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_device = detect_inference_device(args.device)

    texts = list(args.text)
    if args.input_json:
        texts.extend(_load_texts_from_json(Path(args.input_json)))

    if not texts:
        raise ValueError("Provide at least one input via --text or --input-json.")

    model_path, manifest = resolve_artifact_model_path(
        artifact_root=args.artifact_root,
        model_path=args.model_path,
    )
    model = load_embedding_model(
        model_path=model_path,
        manifest=manifest,
        device=resolved_device,
    )
    embeddings = encode_texts(
        model=model,
        texts=texts,
        normalize_embeddings=bool(args.normalize_embeddings),
        batch_size=max(1, int(args.batch_size)),
    )

    payload = {
        "model_path": str(model_path),
        "device": resolved_device,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "shape": [int(x) for x in embeddings.shape],
        "texts": texts,
        "embeddings": embeddings.tolist(),
    }

    output = json.dumps(payload, ensure_ascii=True)
    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
