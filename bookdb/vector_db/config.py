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

        Returns:
            QdrantConfig instance
        """
        mode = os.getenv("QDRANT_MODE", "server")

        if mode not in ("local", "server"):
            raise ValueError(
                f"Invalid QDRANT_MODE: {mode}. Must be 'local' or 'server'"
            )

        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        return cls(
            mode=mode,  # type: ignore[arg-type]
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=api_key,
            https=_parse_bool(os.getenv("QDRANT_HTTPS", "false")),
            path=os.getenv("QDRANT_PATH", "./qdrant_data"),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "10.0")),
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
