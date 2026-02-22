from qdrant_client import QdrantClient


def check_qdrant_connection():
    try:
        client = QdrantClient(url="", api_key="", port=None)
        collections = client.get_collections()
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


connected, message = check_qdrant_connection()
print(f"Connection status: {message}")
