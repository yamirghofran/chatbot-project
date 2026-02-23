---
icon: lucide/server
---

# REST API Documentation

BookDB provides a FastAPI REST API for web and mobile clients. The API handles authentication, book discovery, user interactions, and recommendations.

## Running the API

```bash
# Development with auto-reload
uv run uvicorn apps.api.main:app --reload

# Production (uses PORT env var or defaults to 8000)
python -m apps.api.main

# With custom host/port
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables (loaded from `apps/api/.env`):

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `JWT_SECRET` | Secret for JWT signing (min 32 bytes) | Yes |
| `JWT_EXPIRE_MINUTES` | Token expiration time | Yes |
| `CORS_ORIGINS` | Comma-separated allowed origins | Yes |
| `QDRANT_URL` | Qdrant server URL | No |
| `QDRANT_PORT` | Qdrant port | No |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | No |
| `EMBEDDING_SERVICE_URL` | Embedding service endpoint | No |
| `BPR_PARQUET_URL` | BPR recommendations parquet path/URL | No |
| `BOOK_METRICS_PARQUET_URL` | Book metrics parquet path/URL | No |

## API Routers

### `/auth` - Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Create new user account |
| `/auth/login` | POST | Authenticate and get JWT |
| `/auth/me` | GET | Get current user info |

**Register Request:**
```json
{
  "email": "user@example.com",
  "username": "bookworm",
  "name": "Alice Reader",
  "password": "securepassword"
}
```

**Login Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### `/books` - Book Operations

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/books/search` | GET | No | Search books by title/author |
| `/books/{id}` | GET | No | Get book details |
| `/books/{id}/reviews` | GET | No | List book reviews |
| `/books/{id}/reviews` | POST | Yes | Create a review |
| `/books/{id}/related` | GET | No | Get related books |

**Search Query Parameters:**
- `q` (required): Search query
- `limit`: Max results (default 20, max 100)

**Search Response:**
```json
{
  "directHit": { ... },      // Best match or null
  "keywordResults": [...],   // Other matches
  "aiNarrative": "...",      // LLM response (if no keyword matches)
  "aiBooks": [...]           // AI-recommended books
}
```

### `/discovery` - Recommendations

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/discovery/recommendations` | GET | Optional | Personalized recommendations |
| `/discovery/staff-picks` | GET | No | Curated high-rated books |
| `/discovery/activity` | GET | Optional | Activity feed (placeholder) |

**Recommendations Strategy:**

1. **BPR Model**: Pre-computed recommendations for users with Goodreads IDs
2. **Interaction Vectors**: Real-time similarity search based on user's ratings, shell, and lists
3. **Cold Start**: Popular books by rating count

### `/me` - User Profile

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/me/shell` | GET | Get user's shell (reading queue) |
| `/me/shell/{book_id}` | POST | Add book to shell |
| `/me/shell/{book_id}` | DELETE | Remove from shell |
| `/me/ratings/{book_id}` | GET | Get user's rating for book |
| `/me/ratings` | POST | Upsert rating (1-5) |
| `/me/ratings/{book_id}` | DELETE | Remove rating |
| `/me/lists` | GET | Get user's custom lists |
| `/me/lists` | POST | Create new list |
| `/me/favorites` | GET | Get favorites (max 3) |
| `/me/favorites/{book_id}` | POST | Add to favorites |
| `/me/favorites/{book_id}` | DELETE | Remove from favorites |

### `/users` - User Discovery

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/{id}` | GET | Get public user profile |

### `/lists` - Book Lists

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/lists/{id}` | GET | Get list details |
| `/lists/{id}/books` | POST | Add book to list |
| `/lists/{id}/books/{book_id}` | DELETE | Remove book from list |

### `/reviews` - Review Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reviews/{id}` | GET | Get review details |
| `/reviews/{id}` | PUT | Update own review |
| `/reviews/{id}` | DELETE | Delete own review |
| `/reviews/{id}/likes` | POST | Like a review |
| `/reviews/{id}/likes` | DELETE | Unlike a review |
| `/reviews/{id}/comments` | GET | Get review comments |
| `/reviews/{id}/comments` | POST | Add comment |

## Authentication Flow

1. Client registers or logs in via `/auth/*` endpoints
2. Server returns JWT access token
3. Client includes token in `Authorization: Bearer <token>` header
4. Protected endpoints validate token and inject `current_user`

## Startup Behavior

On startup, the API:

1. Connects to PostgreSQL (fails if unavailable)
2. Connects to Qdrant (graceful degradation if unavailable)
3. Loads BPR recommendations parquet (fallback to cold-start if missing)
4. Loads book metrics parquet (fallback to Postgres aggregations)

## Error Handling

Standard HTTP status codes:

- `200` - Success
- `201` - Created
- `204` - No Content (successful deletion)
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

Error response format:
```json
{
  "detail": "Error message"
}
```
