"""Tests for Qdrant client module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from bookdb.vector_db.client import (
    _create_local_client,
    _create_server_client,
    _validate_connection,
    get_client_info,
    get_qdrant_client,
    reset_client,
)
from bookdb.vector_db.config import QdrantConfig


@pytest.fixture(autouse=True)
def reset_client_before_test():
    """Reset client before each test to ensure clean state."""
    reset_client()
    yield
    reset_client()


@pytest.fixture
def local_config():
    """Fixture for local mode configuration."""
    return QdrantConfig(
        mode="local",
        path="./test_qdrant_data",
    )


@pytest.fixture
def server_config():
    """Fixture for server mode configuration."""
    return QdrantConfig(
        mode="server",
        host="localhost",
        port=6333,
        api_key=None,
        https=False,
        timeout=10.0,
    )


class TestQdrantConfig:
    """Tests for QdrantConfig class."""

    def test_config_defaults(self):
        config = QdrantConfig()
        assert config.mode == "server"
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.path == "./qdrant_data"
        assert config.timeout == 10.0

    def test_config_from_env(self):
        with patch.dict(os.environ, {
            "QDRANT_MODE": "local",
            "QDRANT_HOST": "testhost",
            "QDRANT_PORT": "7000",
            "QDRANT_API_KEY": "secret",
            "QDRANT_HTTPS": "true",
            "QDRANT_PATH": "/tmp/qdrant",
            "QDRANT_TIMEOUT": "5.5",
        }):
            config = QdrantConfig.from_env()
            assert config.mode == "local"
            assert config.host == "testhost"
            assert config.port == 7000
            assert config.api_key == "secret"
            assert config.https is True
            assert config.path == "/tmp/qdrant"
            assert config.timeout == 5.5

    def test_config_from_env_invalid_mode(self):
        with patch.dict(os.environ, {"QDRANT_MODE": "invalid"}):
            with pytest.raises(ValueError, match="Invalid QDRANT_MODE"):
                QdrantConfig.from_env()

    def test_config_validate_server_mode(self):
        config = QdrantConfig(mode="server", host="localhost", port=6333)
        config.validate()

    def test_config_validate_server_mode_no_host(self):
        config = QdrantConfig(mode="server", host="", port=6333)
        with pytest.raises(ValueError, match="Host is required"):
            config.validate()

    def test_config_validate_server_mode_invalid_port(self):
        config = QdrantConfig(mode="server", host="localhost", port=70000)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_config_validate_local_mode(self):
        config = QdrantConfig(mode="local", path="./data")
        config.validate()

    def test_config_validate_local_mode_no_path(self):
        config = QdrantConfig(mode="local", path="")
        with pytest.raises(ValueError, match="Path is required"):
            config.validate()

    def test_config_validate_timeout(self):
        config = QdrantConfig(timeout=0)
        with pytest.raises(ValueError, match="Timeout must be > 0"):
            config.validate()


class TestGetQdrantClient:
    """Tests for get_qdrant_client function."""

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_get_server_client(self, mock_qdrant_client, server_config):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        client = get_qdrant_client(server_config)

        assert client is mock_client
        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333,
            api_key=None,
            https=False,
            timeout=10.0,
        )
        mock_client.get_collections.assert_called_once()

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_get_local_client(self, mock_qdrant_client, local_config):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        client = get_qdrant_client(local_config)

        assert client is mock_client
        mock_qdrant_client.assert_called_once_with(
            path="./test_qdrant_data",
            timeout=10.0,
        )
        mock_client.get_collections.assert_called_once()

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_singleton_pattern(self, mock_qdrant_client, server_config):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        client1 = get_qdrant_client(server_config)
        client2 = get_qdrant_client(server_config)

        assert client1 is client2
        mock_qdrant_client.assert_called_once()

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_connection_failure(self, mock_qdrant_client, server_config):
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection refused")
        mock_qdrant_client.return_value = mock_client

        with pytest.raises(ConnectionError, match="Failed to connect"):
            get_qdrant_client(server_config)

    @patch("bookdb.vector_db.client.QdrantConfig.from_env")
    @patch("bookdb.vector_db.client.QdrantClient")
    def test_get_client_from_env(self, mock_qdrant_client, mock_from_env):
        mock_config = QdrantConfig(mode="server", host="localhost", port=6333)
        mock_from_env.return_value = mock_config

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        client = get_qdrant_client()

        assert client is mock_client
        mock_from_env.assert_called_once()

class TestResetClient:
    """Tests for reset_client function."""

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_reset_client(self, mock_qdrant_client):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        config = QdrantConfig(mode="server", host="localhost", port=6333)

        client1 = get_qdrant_client(config)
        assert client1 is mock_client

        reset_client()
        client2 = get_qdrant_client(config)

        assert mock_qdrant_client.call_count == 2
        assert client2 is mock_client


class TestGetClientInfo:
    """Tests for get_client_info function."""

    def test_get_client_info_no_connection(self):
        info = get_client_info()

        assert info["connected"] is False
        assert info["mode"] is None
        assert info["config"] is None

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_get_client_info_server_mode(self, mock_qdrant_client):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client.return_value = mock_client

        config = QdrantConfig(mode="server", host="testhost", port=7001)
        get_qdrant_client(config)

        info = get_client_info()
        assert info["connected"] is True
        assert info["mode"] == "server"
        assert info["config"]["host"] == "testhost"
        assert info["config"]["port"] == 7001


class TestValidateConnection:
    """Tests for _validate_connection function."""

    def test_validate_connection_success(self):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()

        _validate_connection(mock_client)
        mock_client.get_collections.assert_called_once()

    def test_validate_connection_failure(self):
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Probe failed")

        with pytest.raises(ConnectionError, match="connection validation failed"):
            _validate_connection(mock_client)


class TestCreateClientHelpers:
    """Tests for low-level create_*_client helpers."""

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_create_local_client(self, mock_qdrant_client):
        config = QdrantConfig(mode="local", path="./foo", timeout=2.5)
        _create_local_client(config)
        mock_qdrant_client.assert_called_once_with(path="./foo", timeout=2.5)

    @patch("bookdb.vector_db.client.QdrantClient")
    def test_create_server_client(self, mock_qdrant_client):
        config = QdrantConfig(
            mode="server",
            host="localhost",
            port=6333,
            api_key="k",
            https=True,
            timeout=3.0,
        )
        _create_server_client(config)
        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333,
            api_key="k",
            https=True,
            timeout=3.0,
        )
