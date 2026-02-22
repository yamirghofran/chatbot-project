"""Configuration for Qdrant client."""

import os
from dataclasses import dataclass
from typing import Literal, Optional


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_qdrant_url(raw_url: str, https_hint: bool) -> Optional[str]:
    url = raw_url.strip()
    if not url:
        return None

    if url.startswith("http://") or url.startswith("https://"):
        return url

    local_hosts = ("localhost", "127.0.0.1", "[::1]")
    scheme = "http" if url.startswith(local_hosts) else ("https" if https_hint else "https")
    return f"{scheme}://{url}"


def _from_profile_or_global(
    key_suffix: str,
    profile: Optional[str],
    default: str,
) -> str:
    profile_key = ""
    if profile:
        profile_key = f"QDRANT_{profile.upper()}_{key_suffix}"
    global_key = f"QDRANT_{key_suffix}"

    if profile_key:
        profiled_value = os.getenv(profile_key)
        if profiled_value is not None:
            return profiled_value

    return os.getenv(global_key, default)


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
    url: Optional[str] = None
    api_key: Optional[str] = None
    https: bool = False
    path: str = "./qdrant_data"
    timeout: float = 10.0
    environment: Optional[str] = None
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
        raw_environment = os.getenv("QDRANT_ENV", "").strip().lower()
        environment = raw_environment if raw_environment in {"dev", "prod"} else None

        mode = _from_profile_or_global("MODE", environment, "server").strip()

        if mode not in ("local", "server"):
            raise ValueError(
                f"Invalid QDRANT_MODE: {mode}. Must be 'local' or 'server'"
            )

        https = _parse_bool(_from_profile_or_global("HTTPS", environment, "false"))
        api_key = _from_profile_or_global("API_KEY", environment, "").strip() or None
        url = _normalize_qdrant_url(
            _from_profile_or_global("URL", environment, ""),
            https_hint=https,
        )
        books_hnsw_m_raw = os.getenv("QDRANT_BOOKS_HNSW_M", "16").strip()
        books_hnsw_m = int(books_hnsw_m_raw) if books_hnsw_m_raw else None

        return cls(
            mode=mode,  # type: ignore[arg-type]
            host=_from_profile_or_global("HOST", environment, "localhost"),
            port=int(_from_profile_or_global("PORT", environment, "6333")),
            url=url,
            api_key=api_key,
            https=https,
            path=_from_profile_or_global("PATH", environment, "./qdrant_data"),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "10.0")),
            environment=environment,
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
            if self.url is not None:
                if not self.url.strip():
                    raise ValueError("QDRANT URL cannot be blank when provided")
            else:
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
