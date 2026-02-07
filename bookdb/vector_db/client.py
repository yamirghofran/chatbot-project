"""ChromaDB client management with singleton pattern."""

from typing import Optional
import chromadb
from chromadb import HttpClient, PersistentClient, ClientAPI

from .config import ChromaDBConfig


# Global client instance for singleton pattern
_client_instance: Optional[ClientAPI] = None


def get_chroma_client(config: Optional[ChromaDBConfig] = None) -> ClientAPI:
    """Get ChromaDB client instance (singleton pattern).
    
    This function implements a singleton pattern to ensure only one client
    instance is created and reused throughout the application lifecycle.
    
    Args:
        config: ChromaDB configuration. If None, loads from environment.
    
    Returns:
        ChromaDB client instance
        
    Raises:
        ConnectionError: If unable to connect to ChromaDB
        ValueError: If configuration is invalid
    
    Example:
        >>> client = get_chroma_client()
        >>> collections = client.list_collections()
    """
    global _client_instance
    
    # Return existing instance if available
    if _client_instance is not None:
        return _client_instance
    
    # Load configuration
    if config is None:
        config = ChromaDBConfig.from_env()
    
    # Validate configuration
    config.validate()
    
    # Create client based on mode
    try:
        if config.mode == "embedded":
            _client_instance = _create_embedded_client(config)
        else:
            _client_instance = _create_server_client(config)
        
        # Validate connection
        _validate_connection(_client_instance)
        
        return _client_instance
        
    except Exception as e:
        _client_instance = None
        raise ConnectionError(
            f"Failed to connect to ChromaDB in {config.mode} mode: {str(e)}"
        ) from e


def _create_embedded_client(config: ChromaDBConfig) -> PersistentClient:
    """Create embedded ChromaDB client.
    
    Args:
        config: ChromaDB configuration
    
    Returns:
        Embedded ChromaDB client
    """
    return chromadb.PersistentClient(path=config.persist_directory)


def _create_server_client(config: ChromaDBConfig) -> HttpClient:
    """Create server-mode ChromaDB client.
    
    Args:
        config: ChromaDB configuration
    
    Returns:
        HTTP ChromaDB client
    """
    return chromadb.HttpClient(host=config.host, port=config.port)


def _validate_connection(client: ClientAPI) -> None:
    """Validate ChromaDB connection.
    
    Args:
        client: ChromaDB client to validate
    
    Raises:
        ConnectionError: If connection validation fails
    """
    try:
        # Try to get heartbeat or list collections to verify connection
        client.heartbeat()
    except Exception as e:
        raise ConnectionError(f"ChromaDB connection validation failed: {str(e)}") from e


def reset_client() -> None:
    """Reset the global client instance.
    
    This is primarily useful for testing to ensure a fresh client
    instance is created.
    """
    global _client_instance
    _client_instance = None


def get_client_info() -> dict:
    """Get information about the current client instance.
    
    Returns:
        Dictionary with client information including:
        - connected: Whether a client instance exists
        - mode: Client mode (embedded/server) if connected
        - config: Configuration details if connected
    """
    global _client_instance
    
    if _client_instance is None:
        return {
            "connected": False,
            "mode": None,
            "config": None,
        }
    
    # Try to determine client type by checking class name
    client_class_name = type(_client_instance).__name__
    
    if "Persistent" in client_class_name:
        mode = "embedded"
    elif "Http" in client_class_name:
        mode = "server"
    else:
        mode = "unknown"
    
    return {
        "connected": True,
        "mode": mode,
        "config": ChromaDBConfig.from_env().__dict__,
    }
