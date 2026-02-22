"""add_performance_indexes

Revision ID: b7d4f91a2c10
Revises: a1b2c3d4e5f6
Create Date: 2026-02-21 21:05:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b7d4f91a2c10"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Required for trigram GIN indexes used by ILIKE/LIKE '%...%' search paths.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_books_title_trgm "
            "ON books USING gin (lower(title) gin_trgm_ops)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_authors_name_trgm "
            "ON authors USING gin (lower(name) gin_trgm_ops)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_book_authors_author_book "
            "ON book_authors (author_id, book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_reviews_book_created "
            "ON reviews (book_id, created_at DESC, id DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_reviews_user_created "
            "ON reviews (user_id, created_at DESC, id DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_reviews_user_book "
            "ON reviews (user_id, book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_review_comments_review_created "
            "ON review_comments (review_id, created_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_review_comments_user "
            "ON review_comments (user_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_review_likes_user_review "
            "ON review_likes (user_id, review_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_book_ratings_book "
            "ON book_ratings (book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_book_ratings_user_updated "
            "ON book_ratings (user_id, updated_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_book_ratings_user_rating "
            "ON book_ratings (user_id, rating DESC, updated_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_list_books_list_added "
            "ON list_books (list_id, added_at DESC, book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_list_books_book "
            "ON list_books (book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_shell_books_shell_added "
            "ON shell_books (shell_id, added_at DESC, book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_shell_books_book "
            "ON shell_books (book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_book_tags_tag_book "
            "ON book_tags (tag_id, book_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY idx_users_email_lower "
            "ON users (lower(email))"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_users_email_lower")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_book_tags_tag_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_shell_books_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_shell_books_shell_added")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_list_books_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_list_books_list_added")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_book_ratings_user_rating")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_book_ratings_user_updated")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_book_ratings_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_review_likes_user_review")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_review_comments_user")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_review_comments_review_created")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_reviews_user_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_reviews_user_created")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_reviews_book_created")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_book_authors_author_book")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_authors_name_trgm")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_books_title_trgm")
