# BookDB Tests

Test suites for the BookDB project using pytest.

## Structure

```
tests/
├── config.py              # Shared test configuration
├── test_datasets.py       # Dataset validation tests
├── test_api/              # API endpoint tests
│   ├── test_config.py     # API configuration tests
│   ├── test_discovery.py  # Route discovery tests
│   └── test_serialize.py  # Serialization tests
├── test_db/               # PostgreSQL database tests
│   ├── conftest.py        # Fixtures for DB tests
│   ├── test_author_crud.py
│   ├── test_book_crud.py
│   ├── test_user_crud.py
│   ├── test_review_crud.py
│   ├── test_list_shell_rating_crud.py
│   └── test_validation_helpers.py
└── test_vector_db/        # Qdrant vector DB tests
    ├── test_client.py     # Client connection tests
    ├── test_collections.py
    ├── test_book_crud.py
    └── test_crud_base.py
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test directory
pytest tests/test_db/

# Run with coverage
pytest tests/ --cov=bookdb

# Run specific test file
pytest tests/test_db/test_book_crud.py -v

# Run with verbose output
pytest tests/ -v
```

## Requirements

- PostgreSQL database (for `test_db/`)
- Qdrant instance (for `test_vector_db/`)
- Test data fixtures

## Test Categories

| Directory | Purpose |
|-----------|---------|
| `test_api/` | FastAPI endpoint validation |
| `test_db/` | PostgreSQL CRUD operations |
| `test_vector_db/` | Qdrant vector operations |

## CI/CD

Tests run automatically on pull requests via GitHub Actions (`.github/workflows/`).
