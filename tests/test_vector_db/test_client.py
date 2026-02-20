"""Tests for ChromaDB client module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from bookdb.vector_db.client import (
    get_chroma_client,
    reset_client,
    get_client_info,
    _create_embedded_client,
    _create_server_client,
    _validate_connection,
)
from bookdb.vector_db.config import ChromaDBConfig


@pytest.fixture(autouse=True)
def reset_client_before_test():
    """Reset client before each test to ensure clean state."""
    reset_client()
    yield
    reset_client()


@pytest.fixture
def embedded_config():
    """Fixture for embedded mode configuration."""
    return ChromaDBConfig(
        mode="embedded",
        persist_directory="./test_chroma_data",
    )


@pytest.fixture
def server_config():
    """Fixture for server mode configuration."""
    return ChromaDBConfig(
        mode="server",
        host="localhost",
        port=8000,
    )


class TestChromaDBConfig:
    """Tests for ChromaDBConfig class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ChromaDBConfig()
        assert config.mode == "server"
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.persist_directory == "./chroma_data"
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            "CHROMA_MODE": "embedded",
            "CHROMA_HOST": "testhost",
            "CHROMA_PORT": "9000",
            "CHROMA_PERSIST_DIR": "/tmp/chroma",
        }):
            config = ChromaDBConfig.from_env()
            assert config.mode == "embedded"
            assert config.host == "testhost"
            assert config.port == 9000
            assert config.persist_directory == "/tmp/chroma"
    
    def test_config_from_env_invalid_mode(self):
        """Test error handling for invalid mode in environment."""
        with patch.dict(os.environ, {"CHROMA_MODE": "invalid"}):
            with pytest.raises(ValueError, match="Invalid CHROMA_MODE"):
                ChromaDBConfig.from_env()
    
    def test_config_validate_server_mode(self):
        """Test validation for server mode."""
        config = ChromaDBConfig(mode="server", host="localhost", port=8000)
        config.validate()  # Should not raise
    
    def test_config_validate_server_mode_no_host(self):
        """Test validation error for server mode without host."""
        config = ChromaDBConfig(mode="server", host="", port=8000)
        with pytest.raises(ValueError, match="Host is required"):
            config.validate()
    
    def test_config_validate_server_mode_invalid_port(self):
        """Test validation error for invalid port."""
        config = ChromaDBConfig(mode="server", host="localhost", port=70000)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()
    
    def test_config_validate_embedded_mode(self):
        """Test validation for embedded mode."""
        config = ChromaDBConfig(mode="embedded", persist_directory="./data")
        config.validate()  # Should not raise
    
    def test_config_validate_embedded_mode_no_directory(self):
        """Test validation error for embedded mode without directory."""
        config = ChromaDBConfig(mode="embedded", persist_directory="")
        with pytest.raises(ValueError, match="Persist directory is required"):
            config.validate()


class TestGetChromaClient:
    """Tests for get_chroma_client function."""
    
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_get_server_client(self, mock_http_client, server_config):
        """Test creating server mode client."""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_http_client.return_value = mock_client
        
        client = get_chroma_client(server_config)
        
        assert client is mock_client
        mock_http_client.assert_called_once_with(host="localhost", port=8000)
        mock_client.heartbeat.assert_called_once()
    
    @patch("bookdb.vector_db.client.chromadb.PersistentClient")
    def test_get_embedded_client(self, mock_persistent_client, embedded_config):
        """Test creating embedded mode client."""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_persistent_client.return_value = mock_client
        
        client = get_chroma_client(embedded_config)
        
        assert client is mock_client
        mock_persistent_client.assert_called_once_with(path="./test_chroma_data")
        mock_client.heartbeat.assert_called_once()
    
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_singleton_pattern(self, mock_http_client, server_config):
        """Test that client follows singleton pattern."""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_http_client.return_value = mock_client
        
        client1 = get_chroma_client(server_config)
        client2 = get_chroma_client(server_config)
        
        assert client1 is client2
        # HttpClient should only be called once due to singleton
        mock_http_client.assert_called_once()
    
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_connection_failure(self, mock_http_client, server_config):
        """Test error handling for connection failure."""
        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("Connection refused")
        mock_http_client.return_value = mock_client
        
        with pytest.raises(ConnectionError, match="Failed to connect"):
            get_chroma_client(server_config)
    
    @patch("bookdb.vector_db.client.ChromaDBConfig.from_env")
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_get_client_from_env(self, mock_http_client, mock_from_env):
        """Test getting client with config from environment."""
        mock_config = ChromaDBConfig(mode="server", host="localhost", port=8000)
        mock_from_env.return_value = mock_config
        
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_http_client.return_value = mock_client
        
        client = get_chroma_client()
        
        assert client is mock_client
        mock_from_env.assert_called_once()


class TestResetClient:
    """Tests for reset_client function."""
    
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_reset_client(self, mock_http_client):
        """Test resetting client instance."""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_http_client.return_value = mock_client
        
        config = ChromaDBConfig(mode="server", host="localhost", port=8000)
        
        # Create first client
        client1 = get_chroma_client(config)
        assert client1 is mock_client
        
        # Reset and create new client
        reset_client()
        client2 = get_chroma_client(config)
        
        # Should create a new instance
        assert mock_http_client.call_count == 2


class TestGetClientInfo:
    """Tests for get_client_info function."""
    
    def test_get_client_info_no_connection(self):
        """Test getting info when no client is connected."""
        info = get_client_info()
        
        assert info["connected"] is False
        assert info["mode"] is None
        assert info["config"] is None
    
    @patch("bookdb.vector_db.client.chromadb.HttpClient")
    def test_get_client_info_server_mode(self, mock_http_client):
        """Test getting info for server mode client."""
        # Create a mock with a class name that contains "Http"
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        mock_client.__class__.__name__ = "HttpClient"
        mock_http_client.return_value = mock_client
        
        with patch.dict(os.environ, {
            "CHROMA_MODE": "server",
            "CHROMA_HOST": "testhost",
            "CHROMA_PORT": "9000",
        }):
            config = ChromaDBConfig.from_env()
            get_chroma_client(config)
            
            info = get_client_info()
            
            assert info["connected"] is True
            assert info["mode"] == "server"
            assert info["config"]["mode"] == "server"
            assert info["config"]["host"] == "testhost"
            assert info["config"]["port"] == 9000


class TestValidateConnection:
    """Tests for _validate_connection function."""
    
    def test_validate_connection_success(self):
        """Test successful connection validation."""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = None
        
        _validate_connection(mock_client)  # Should not raise
        mock_client.heartbeat.assert_called_once()
    
    def test_validate_connection_failure(self):
        """Test connection validation failure."""
        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("Heartbeat failed")
        
        with pytest.raises(ConnectionError, match="connection validation failed"):
            _validate_connection(mock_client)
