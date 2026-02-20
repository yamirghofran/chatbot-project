"""Configuration for ChromaDB client."""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB connection.
    
    Attributes:
        mode: Connection mode - 'embedded' or 'server'
        host: Host for server mode (default: localhost)
        port: Port for server mode (default: 8000)
        persist_directory: Directory for embedded mode persistence (default: ./chroma_data)
    """
    
    mode: Literal["embedded", "server"] = "server"
    host: str = "localhost"
    port: int = 8000
    persist_directory: str = "./chroma_data"
    
    @classmethod
    def from_env(cls) -> "ChromaDBConfig":
        """Load configuration from environment variables.
        
        Environment variables:
            CHROMA_MODE: Connection mode ('embedded' or 'server')
            CHROMA_HOST: Server host
            CHROMA_PORT: Server port
            CHROMA_PERSIST_DIR: Directory for embedded mode
        
        Returns:
            ChromaDBConfig instance
        """
        mode = os.getenv("CHROMA_MODE", "server")
        
        if mode not in ("embedded", "server"):
            raise ValueError(
                f"Invalid CHROMA_MODE: {mode}. Must be 'embedded' or 'server'"
            )
        
        return cls(
            mode=mode,  # type: ignore
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
        )
    
    def validate(self) -> None:
        """Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.mode not in ("embedded", "server"):
            raise ValueError(f"Invalid mode: {self.mode}")
        
        if self.mode == "server":
            if not self.host:
                raise ValueError("Host is required for server mode")
            if not (1 <= self.port <= 65535):
                raise ValueError(f"Invalid port: {self.port}")
        
        if self.mode == "embedded":
            if not self.persist_directory:
                raise ValueError("Persist directory is required for embedded mode")
