from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookCRUD, BookListCRUD
from bookdb.db.models import Book, BookAuthor, BookList, BookTag, ListBook

from ..core.deps import get_current_user, get_db
from ..core.serialize import serialize_list
from ..schemas.list import CreateListRequest, UpdateListRequest
from bookdb.db.models import User

router = APIRouter(prefix="/lists", tags=["lists"])


def _load_list(db: Session, list_id: int) -> BookList:
    lst = db.scalar(
        select(BookList)
        .where(BookList.id == list_id)
        .options(
            selectinload(BookList.user),
            selectinload(BookList.books)
            .selectinload(ListBook.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(BookList.books)
            .selectinload(ListBook.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
    )
    if lst is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="List not found")
    return lst


@router.get("/{list_id}")
def get_list(list_id: int, db: Session = Depends(get_db)):
    return serialize_list(_load_list(db, list_id))


@router.put("/{list_id}")
def update_list(
    list_id: int,
    body: UpdateListRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lst = _load_list(db, list_id)
    if lst.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your list")
    if body.name is not None:
        lst.title = body.name
    if body.description is not None:
        lst.description = body.description
    db.commit()
    db.refresh(lst)
    return serialize_list(_load_list(db, list_id))


@router.delete("/{list_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_list(
    list_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lst = BookListCRUD.get_by_id(db, list_id)
    if lst is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="List not found")
    if lst.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your list")
    db.delete(lst)
    db.commit()


@router.post("/{list_id}/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_book_to_list(
    list_id: int,
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lst = BookListCRUD.get_by_id(db, list_id)
    if lst is None or lst.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="List not found")
    ok = BookListCRUD.add_book(db, list_id, book_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    db.commit()


@router.delete("/{list_id}/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_book_from_list(
    list_id: int,
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lst = BookListCRUD.get_by_id(db, list_id)
    if lst is None or lst.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="List not found")
    BookListCRUD.remove_book(db, list_id, book_id)
    db.commit()


@router.put("/{list_id}/reorder", status_code=status.HTTP_204_NO_CONTENT)
def reorder_list(
    list_id: int,
    book_ids: list[int],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lst = BookListCRUD.get_by_id(db, list_id)
    if lst is None or lst.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="List not found")
    # Reorder is stored implicitly via added_at; for MVP we just accept the call.
    db.commit()
