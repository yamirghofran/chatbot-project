# Current Schema and Dataset Mapping

This page documents the current relational schema and exactly how parquet datasets are imported.

## Schema Overview

### 1. `users`

- `id BIGSERIAL PRIMARY KEY` (internal id)
- `goodreads_id BIGINT UNIQUE NULL` (from datasets)
- `name TEXT NOT NULL`
- `username TEXT UNIQUE NOT NULL`
- `email TEXT UNIQUE NOT NULL`
- `password_hash TEXT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 2. `authors`

- `id BIGSERIAL PRIMARY KEY` (internal id)
- `goodreads_id BIGINT UNIQUE NULL` (from datasets)
- `name TEXT NOT NULL`
- `description TEXT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 3. `books`

- `id BIGSERIAL PRIMARY KEY` (internal id)
- `goodreads_id BIGINT UNIQUE NOT NULL`
- `title TEXT NOT NULL`
- `description TEXT NULL`
- `image_url TEXT NULL`
- `format TEXT NULL`
- `publisher TEXT NULL`
- `publication_year INT NULL`
- `isbn13 TEXT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 4. `book_authors`

- `book_id BIGINT REFERENCES books(id) ON DELETE CASCADE`
- `author_id BIGINT REFERENCES authors(id) ON DELETE RESTRICT`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `PRIMARY KEY (book_id, author_id)`

### 5. `lists`

- `id BIGSERIAL PRIMARY KEY`
- `user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`
- `title TEXT NOT NULL`
- `description TEXT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `UNIQUE (user_id, title)`

### 6. `list_books`

- `list_id BIGINT REFERENCES lists(id) ON DELETE CASCADE`
- `book_id BIGINT REFERENCES books(id) ON DELETE CASCADE`
- `added_at TIMESTAMPTZ NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `PRIMARY KEY (list_id, book_id)`

### 7. `shells`

- `id BIGSERIAL PRIMARY KEY`
- `user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE`
- `name TEXT NOT NULL DEFAULT 'My Shell'`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 8. `shell_books`

- `shell_id BIGINT REFERENCES shells(id) ON DELETE CASCADE`
- `book_id BIGINT REFERENCES books(id) ON DELETE CASCADE`
- `added_at TIMESTAMPTZ NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `PRIMARY KEY (shell_id, book_id)`

### 9. `book_ratings`

- `user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`
- `book_id BIGINT REFERENCES books(id) ON DELETE CASCADE`
- `rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5)`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `PRIMARY KEY (user_id, book_id)`

### 10. `reviews`

- `id BIGSERIAL PRIMARY KEY`
- `goodreads_id TEXT UNIQUE NULL`
- `user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`
- `book_id BIGINT REFERENCES books(id) ON DELETE CASCADE`
- `review_text TEXT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 11. `review_comments`

- `id BIGSERIAL PRIMARY KEY`
- `review_id BIGINT REFERENCES reviews(id) ON DELETE CASCADE`
- `user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`
- `comment_text TEXT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

### 12. `review_likes`

- `review_id BIGINT REFERENCES reviews(id) ON DELETE CASCADE`
- `user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`
- `created_at TIMESTAMPTZ NOT NULL`
- `PRIMARY KEY (review_id, user_id)`

## Dataset Import Mapping

Importer script: `scripts/import_goodreads_to_postgres.py`

### Authors dataset

Source:

- `data/raw_goodreads_book_authors.parquet`

Mapping:

- `author_id` -> `authors.goodreads_id`
- `name` -> `authors.name`

### Books dataset

Source:

- `data/3_goodreads_books_with_metrics.parquet`

Mapping:

- `book_id` -> `books.goodreads_id`
- `title` -> `books.title`
- `description` -> `books.description`
- `image_url` -> `books.image_url`
- `format` -> `books.format`
- `publisher` -> `books.publisher`
- `publication_year` -> `books.publication_year`
- `isbn13` -> `books.isbn13`

`authors` field is used to build `book_authors`:

- `books.authors[*].author_id` matches `authors.goodreads_id`
- relation stored with internal PKs: `book_authors.book_id` + `book_authors.author_id`

### Reviews dataset

Source:

- `data/3_goodreads_reviews_dedup_clean.parquet`

Mapping:

- `review_id` -> `reviews.goodreads_id`
- `review_text` -> `reviews.review_text`
- `book_id` (Goodreads) -> `books.goodreads_id` -> internal `reviews.book_id`
- `user_id` (Goodreads) -> `users.goodreads_id` -> internal `reviews.user_id`
- `ts_updated` -> `reviews.created_at` / `reviews.updated_at` (UTC)

### Interactions dataset

Source:

- `data/3_goodreads_interactions_reduced.parquet`

Behavior:

- no persisted `interactions` table
- `is_read = 1`:
  - ensure list titled `read` exists per user
  - insert into `list_books`
- `rating > 0`:
  - upsert into `book_ratings`

ID resolution:

- dataset `user_id` resolves through `users.goodreads_id`
- dataset `book_id` resolves through `books.goodreads_id`

## Important Operational Rule

`reviews` and `interactions` rows are skipped when referenced `book_id` does not exist in imported `books` (by Goodreads ID).
