from __future__ import annotations

from bookdb.db.crud import BookListCRUD
from bookdb.db.models import BookList
from sqlalchemy.orm import Session

FAVORITES_LIST_TITLE = "__favorites__"


def is_favorites_list(list_obj: BookList) -> bool:
    return list_obj.title == FAVORITES_LIST_TITLE


def get_or_create_favorites_list(db: Session, user_id: int) -> tuple[BookList, bool]:
    for book_list in BookListCRUD.get_by_user(db, user_id):
        if is_favorites_list(book_list):
            return book_list, False
    return BookListCRUD.create(db, user_id=user_id, title=FAVORITES_LIST_TITLE, description=None), True
