from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def _resolve_existing_path(path_str: str, artifact_root: Path) -> Optional[Path]:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    rooted_candidate = artifact_root / path_str
    if rooted_candidate.exists():
        return rooted_candidate

    parent_rooted_candidate = artifact_root.parent / path_str
    if parent_rooted_candidate.exists():
        return parent_rooted_candidate

    return None


def load_deployment_manifest(artifact_root: Path) -> Optional[dict[str, Any]]:
    manifest_path = artifact_root / "deployment_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_artifact_model_path(
    artifact_root: str | Path,
    model_path: Optional[str | Path] = None,
) -> tuple[Path, Optional[dict[str, Any]]]:
    root = Path(artifact_root)
    manifest = load_deployment_manifest(root)

    if model_path is not None:
        explicit_model_path = Path(model_path)
        return explicit_model_path, manifest

    if manifest:
        for key in ("merged_model_path", "lora_model_path"):
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                resolved = _resolve_existing_path(value, root)
                if resolved is not None:
                    return resolved, manifest

    merged = root / "merged_16bit"
    if merged.exists():
        return merged, manifest

    lora = root / "lora"
    if lora.exists():
        return lora, manifest

    return root, manifest


def load_embedding_model(
    model_path: str | Path,
    manifest: Optional[dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Any:
    resolved_model_path = Path(model_path)
    adapter_config = resolved_model_path / "adapter_config.json"

    if adapter_config.exists():
        try:
            from unsloth import FastSentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "LoRA artifacts require Unsloth at inference time. "
                "Install Unsloth, or serve `merged_16bit` artifacts instead."
            ) from exc

        kwargs: dict[str, Any] = {}
        if manifest is not None:
            max_seq_length = manifest.get("max_seq_length")
            if isinstance(max_seq_length, int) and max_seq_length > 0:
                kwargs["max_seq_length"] = int(max_seq_length)

        try:
            return FastSentenceTransformer.from_pretrained(
                model_name=str(resolved_model_path),
                **kwargs,
            )
        except TypeError:
            return FastSentenceTransformer.from_pretrained(
                model_name=str(resolved_model_path),
            )

    if device:
        return SentenceTransformer(str(resolved_model_path), device=device)
    return SentenceTransformer(str(resolved_model_path))


def encode_texts(
    model: Any,
    texts: list[str],
    normalize_embeddings: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32, order="C")


def cosine_similarity(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)

    query_norm = float(np.linalg.norm(query))
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denom = np.maximum(candidate_norms * query_norm, 1e-12)
    return (candidates @ query) / denom
