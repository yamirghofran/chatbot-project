"""Configuration for Qdrant client."""

import os
from dataclasses import dataclass
from typing import Literal, Optional


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection.

    Attributes:
        mode: Connection mode - 'local' or 'server'
        host: Host for server mode (default: localhost)
        port: Port for server mode (default: 6333)
        api_key: Optional API key for server mode
        https: Whether to use HTTPS in server mode
        path: Directory path for local mode persistence
        timeout: Client timeout in seconds
    """

    mode: Literal["local", "server"] = "server"
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False
    path: str = "./qdrant_data"
    timeout: float = 10.0
    vector_size: int = 768
    books_on_disk: bool = True
    books_int8_quantization: bool = True
    books_quantile: float = 0.99
    books_quantization_always_ram: bool = True
    books_hnsw_on_disk: bool = True
    books_hnsw_m: Optional[int] = 16

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Load configuration from environment variables.

        Environment variables:
            QDRANT_MODE: Connection mode ('local' or 'server')
            QDRANT_HOST: Server host
            QDRANT_PORT: Server port
            QDRANT_API_KEY: Optional API key
            QDRANT_HTTPS: Whether to use HTTPS
            QDRANT_PATH: Directory for local mode
            QDRANT_TIMEOUT: Client timeout in seconds
            QDRANT_VECTOR_SIZE: Vector dimension used by collections
            QDRANT_BOOKS_ON_DISK: Store book vectors on disk
            QDRANT_BOOKS_INT8_QUANTIZATION: Enable Int8 scalar quantization for books
            QDRANT_BOOKS_QUANTILE: Scalar quantization quantile for books
            QDRANT_BOOKS_QUANT_ALWAYS_RAM: Keep quantized vectors in RAM
            QDRANT_BOOKS_HNSW_ON_DISK: Store books HNSW graph on disk
            QDRANT_BOOKS_HNSW_M: HNSW M parameter for books

        Returns:
            QdrantConfig instance
        """
        mode = os.getenv("QDRANT_MODE", "server")

        if mode not in ("local", "server"):
            raise ValueError(
                f"Invalid QDRANT_MODE: {mode}. Must be 'local' or 'server'"
            )

        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        books_hnsw_m_raw = os.getenv("QDRANT_BOOKS_HNSW_M", "16").strip()
        books_hnsw_m = int(books_hnsw_m_raw) if books_hnsw_m_raw else None

        return cls(
            mode=mode,  # type: ignore[arg-type]
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=api_key,
            https=_parse_bool(os.getenv("QDRANT_HTTPS", "false")),
            path=os.getenv("QDRANT_PATH", "./qdrant_data"),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "10.0")),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "768")),
            books_on_disk=_parse_bool(os.getenv("QDRANT_BOOKS_ON_DISK", "true")),
            books_int8_quantization=_parse_bool(
                os.getenv("QDRANT_BOOKS_INT8_QUANTIZATION", "true")
            ),
            books_quantile=float(os.getenv("QDRANT_BOOKS_QUANTILE", "0.99")),
            books_quantization_always_ram=_parse_bool(
                os.getenv("QDRANT_BOOKS_QUANT_ALWAYS_RAM", "true")
            ),
            books_hnsw_on_disk=_parse_bool(
                os.getenv("QDRANT_BOOKS_HNSW_ON_DISK", "true")
            ),
            books_hnsw_m=books_hnsw_m,
        )

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.mode not in ("local", "server"):
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.mode == "server":
            if not self.host:
                raise ValueError("Host is required for server mode")
            if not (1 <= self.port <= 65535):
                raise ValueError(f"Invalid port: {self.port}")

        if self.mode == "local":
            if not self.path:
                raise ValueError("Path is required for local mode")

        if self.timeout <= 0:
            raise ValueError(f"Timeout must be > 0. Got: {self.timeout}")

        if self.vector_size <= 0:
            raise ValueError(f"Vector size must be > 0. Got: {self.vector_size}")

        if not (0.0 < self.books_quantile <= 1.0):
            raise ValueError(
                f"QDRANT_BOOKS_QUANTILE must be in (0, 1]. Got: {self.books_quantile}"
            )

        if self.books_hnsw_m is not None and self.books_hnsw_m <= 0:
            raise ValueError(
                f"QDRANT_BOOKS_HNSW_M must be > 0 when set. Got: {self.books_hnsw_m}"
            )
