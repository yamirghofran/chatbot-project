"""Tests for the MCP adapter."""

from unittest.mock import MagicMock, patch

from apps.api.core.mcp_adapter import MCPAdapter


def test_unavailable_when_no_url():
    adapter = MCPAdapter(base_url=None)
    assert not adapter.is_available()
    assert adapter.recommend(user_id=1, limit=5) == []


def test_recommend_success():
    adapter = MCPAdapter(base_url="http://mcp.local")

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [
        {"book_id": 1, "title": "Book A"},
        {"book_id": 2, "title": "Book B"},
    ]

    with patch("apps.api.core.mcp_adapter.requests.post", return_value=mock_response) as mock_post:
        result = adapter.recommend(user_id=42, limit=5)

    assert len(result) == 2
    assert result[0]["book_id"] == 1
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["user_id"] == 42
    assert call_kwargs[1]["json"]["limit"] == 5


def test_recommend_handles_timeout():
    adapter = MCPAdapter(base_url="http://mcp.local", timeout=1.0)

    with patch("apps.api.core.mcp_adapter.requests.post", side_effect=Exception("Connection timed out")):
        result = adapter.recommend(user_id=1)

    assert result == []


def test_recommend_handles_http_error():
    adapter = MCPAdapter(base_url="http://mcp.local")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("500 Server Error")

    with patch("apps.api.core.mcp_adapter.requests.post", return_value=mock_response):
        result = adapter.recommend(user_id=1)

    assert result == []


def test_recommend_handles_dict_response():
    """MCP server might wrap results in a dict."""
    adapter = MCPAdapter(base_url="http://mcp.local")

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "recommendations": [{"book_id": 10}],
    }

    with patch("apps.api.core.mcp_adapter.requests.post", return_value=mock_response):
        result = adapter.recommend(user_id=1)

    assert len(result) == 1
    assert result[0]["book_id"] == 10


def test_health_returns_false_when_no_url():
    adapter = MCPAdapter(base_url=None)
    assert not adapter.health()


def test_health_returns_true_on_success():
    adapter = MCPAdapter(base_url="http://mcp.local")

    mock_response = MagicMock()
    mock_response.ok = True

    with patch("apps.api.core.mcp_adapter.requests.get", return_value=mock_response):
        assert adapter.health()


def test_health_returns_false_on_error():
    adapter = MCPAdapter(base_url="http://mcp.local")

    with patch("apps.api.core.mcp_adapter.requests.get", side_effect=Exception("timeout")):
        assert not adapter.health()
