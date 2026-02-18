"""truncate all data

Revision ID: b3f7a8c2d901
Revises: 1edca5c1893a
Create Date: 2026-02-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b3f7a8c2d901'
down_revision: Union[str, None] = '1edca5c1893a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "TRUNCATE TABLE book_authors, list_books, reviews, ratings, lists, books, authors, users RESTART IDENTITY CASCADE"
    )


def downgrade() -> None:
    # Data cannot be restored — intentional one-way migration
    pass
