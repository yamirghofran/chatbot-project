from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import urllib.parse

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class _TokenizerRegexWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "incorrect regex pattern" in message and "fix_mistral_regex=True" in message:
            return False
        return True


@contextmanager
def _suppress_tokenizer_regex_warning():
    tokenizer_logger = logging.getLogger("transformers.tokenization_utils_base")
    warning_filter = _TokenizerRegexWarningFilter()
    tokenizer_logger.addFilter(warning_filter)
    try:
        yield
    finally:
        tokenizer_logger.removeFilter(warning_filter)


def detect_inference_device(device: Optional[str] = None) -> str:
    requested = (device or "").strip().lower()
    if requested and requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        logger.warning("PyTorch is unavailable; defaulting inference device to cpu")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _coerce_local_path(path_like: str | Path) -> Path:
    raw = str(path_like).strip()
    if raw.startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        raw = urllib.parse.unquote(parsed.path)
    raw = os.path.expandvars(raw)
    return Path(raw).expanduser()


def _iter_path_candidates(path_like: str | Path) -> list[Path]:
    base = _coerce_local_path(path_like)
    raw = str(base)
    candidates: list[Path] = [base]

    if raw.endswith("_merged_16bit"):
        candidates.append(Path(f"{raw[: -len('_merged_16bit')]}/merged_16bit"))
    if raw.endswith("_lora"):
        candidates.append(Path(f"{raw[: -len('_lora')]}/lora"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _resolve_input_path(path_like: str | Path) -> Path:
    candidates = _iter_path_candidates(path_like)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_existing_path(path_str: str, artifact_root: Path) -> Optional[Path]:
    for candidate in _iter_path_candidates(path_str):
        if candidate.exists():
            return candidate

        rooted_candidate = artifact_root / candidate
        if rooted_candidate.exists():
            return rooted_candidate

        parent_rooted_candidate = artifact_root.parent / candidate
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
    root = _resolve_input_path(artifact_root)
    if root.is_file() and root.name == "deployment_manifest.json":
        with root.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        root = root.parent
    else:
        manifest = load_deployment_manifest(root)

    if model_path is not None:
        explicit_model_path = _resolve_existing_path(str(model_path), root)
        if explicit_model_path is None:
            explicit_model_path = _coerce_local_path(model_path)
        return explicit_model_path, manifest

    if manifest:
        for key in ("merged_model_path", "lora_model_path"):
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                resolved = _resolve_existing_path(value, root)
                if resolved is not None:
                    return resolved, manifest

    if root.is_dir() and (root / "modules.json").exists():
        if (root / "model.safetensors").exists() or (root / "adapter_model.safetensors").exists():
            return root, manifest

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
    resolved_device = detect_inference_device(device)
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

        unsloth_kwargs: dict[str, Any] = {
            "model_name": str(resolved_model_path),
            "device": resolved_device,
            "fix_mistral_regex": True,
            **kwargs,
        }
        with _suppress_tokenizer_regex_warning():
            try:
                return FastSentenceTransformer.from_pretrained(**unsloth_kwargs)
            except TypeError:
                pass

            unsloth_kwargs.pop("fix_mistral_regex", None)
            try:
                return FastSentenceTransformer.from_pretrained(**unsloth_kwargs)
            except TypeError:
                pass

            unsloth_kwargs.pop("device", None)
            return FastSentenceTransformer.from_pretrained(**unsloth_kwargs)

    sentence_transformer_kwargs: dict[str, Any] = {
        "device": resolved_device,
        "tokenizer_kwargs": {"fix_mistral_regex": True},
    }
    with _suppress_tokenizer_regex_warning():
        try:
            return SentenceTransformer(
                str(resolved_model_path),
                **sentence_transformer_kwargs,
            )
        except TypeError:
            sentence_transformer_kwargs.pop("tokenizer_kwargs", None)
            return SentenceTransformer(
                str(resolved_model_path),
                **sentence_transformer_kwargs,
            )


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
