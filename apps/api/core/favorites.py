from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from bookdb.db.crud import BookListCRUD
from bookdb.db.models import BookList

FAVORITES_LIST_TITLE = "__favorites__"


def is_favorites_list(list_obj: BookList) -> bool:
    return list_obj.title == FAVORITES_LIST_TITLE


def get_or_create_favorites_list(db: Session, user_id: int) -> tuple[BookList, bool]:
    existing = db.scalar(
        select(BookList).where(
            BookList.user_id == user_id,
            BookList.title == FAVORITES_LIST_TITLE,
        )
    )
    if existing is not None:
        return existing, False
    return BookListCRUD.create(db, user_id=user_id, title=FAVORITES_LIST_TITLE, description=None), True
