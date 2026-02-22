"""Qdrant client management with singleton pattern."""

from typing import Optional

from qdrant_client import QdrantClient

from .config import QdrantConfig


# Global client instance for singleton pattern
_client_instance: Optional[QdrantClient] = None
_client_config: Optional[QdrantConfig] = None


def get_qdrant_client(config: Optional[QdrantConfig] = None) -> QdrantClient:
    """Get Qdrant client instance (singleton pattern).

    This function implements a singleton pattern to ensure only one client
    instance is created and reused throughout the application lifecycle.

    Args:
        config: Qdrant configuration. If None, loads from environment.

    Returns:
        Qdrant client instance

    Raises:
        ConnectionError: If unable to connect to Qdrant
        ValueError: If configuration is invalid
    """
    global _client_instance, _client_config

    if _client_instance is not None:
        return _client_instance

    if config is None:
        config = QdrantConfig.from_env()

    config.validate()

    try:
        if config.mode == "local":
            _client_instance = _create_local_client(config)
        else:
            _client_instance = _create_server_client(config)

        _validate_connection(_client_instance)
        _client_config = config

        return _client_instance

    except Exception as e:
        _client_instance = None
        _client_config = None
        raise ConnectionError(
            f"Failed to connect to Qdrant in {config.mode} mode: {str(e)}"
        ) from e


def _create_local_client(config: QdrantConfig) -> QdrantClient:
    """Create local Qdrant client."""
    return QdrantClient(path=config.path, timeout=config.timeout)


def _create_server_client(config: QdrantConfig) -> QdrantClient:
    """Create server-mode Qdrant client."""
    if config.url:
        return QdrantClient(
            url=config.url,
            # Important: qdrant-client defaults to port=6333 even with URL set.
            # For hosted endpoints (e.g., Railway) we must let the URL decide.
            port=None,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    return QdrantClient(
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        https=config.https,
        timeout=config.timeout,
    )


def _validate_connection(client: QdrantClient) -> None:
    """Validate Qdrant connection."""
    try:
        # Collection listing is a lightweight probe to verify connectivity.
        client.get_collections()
    except Exception as e:
        raise ConnectionError(f"Qdrant connection validation failed: {str(e)}") from e


def reset_client() -> None:
    """Reset the global client instance."""
    global _client_instance, _client_config
    _client_instance = None
    _client_config = None


def get_client_info() -> dict:
    """Get information about the current client instance."""
    global _client_instance, _client_config

    if _client_instance is None:
        return {
            "connected": False,
            "mode": None,
            "config": None,
        }

    return {
        "connected": True,
        "mode": _client_config.mode if _client_config else "unknown",
        "config": _client_config.__dict__ if _client_config else None,
    }
