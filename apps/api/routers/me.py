from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookListCRUD, BookRatingCRUD, ShellCRUD
from bookdb.db.models import (
    Book,
    BookAuthor,
    BookList,
    BookRating,
    BookTag,
    ListBook,
    ShellBook,
    User,
)

from ..core.deps import get_current_user, get_db
from ..core.favorites import get_or_create_favorites_list, is_favorites_list
from ..core.serialize import serialize_book, serialize_list
from ..schemas.list import CreateListRequest

router = APIRouter(prefix="/me", tags=["me"])


@router.get("/shell")
def get_shell(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    shell = ShellCRUD.get_by_user(db, current_user.id)
    if shell is None:
        return []
    shell_books = db.scalars(
        select(ShellBook)
        .where(ShellBook.shell_id == shell.id)
        .options(
            selectinload(ShellBook.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(ShellBook.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
        .order_by(ShellBook.added_at.desc())
    ).all()
    return [serialize_book(sb.book) for sb in shell_books]


@router.post("/shell/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_to_shell(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    shell = ShellCRUD.get_or_create_for_user(db, current_user.id)
    ok = ShellCRUD.add_book(db, shell.id, book_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    db.commit()


@router.delete("/shell/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_shell(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    shell = ShellCRUD.get_by_user(db, current_user.id)
    if shell:
        ShellCRUD.remove_book(db, shell.id, book_id)
        db.commit()


@router.get("/ratings/{book_id}")
def get_my_rating(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rating = db.scalar(
        select(BookRating).where(
            BookRating.user_id == current_user.id,
            BookRating.book_id == book_id,
        )
    )
    return {"rating": rating.rating if rating is not None else None}


@router.post("/ratings", status_code=status.HTTP_204_NO_CONTENT)
def upsert_rating(
    book_id: int,
    rating: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        BookRatingCRUD.upsert(db, current_user.id, book_id, rating)
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.delete("/ratings/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    BookRatingCRUD.delete(db, current_user.id, book_id)
    db.commit()


@router.get("/lists")
def get_my_lists(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    lists = db.scalars(
        select(BookList)
        .where(BookList.user_id == current_user.id)
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
    ).all()
    return [serialize_list(lst) for lst in lists if not is_favorites_list(lst)]


@router.post("/lists", status_code=status.HTTP_201_CREATED)
def create_list(
    body: CreateListRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        lst = BookListCRUD.create(db, current_user.id, body.name, body.description)
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {"id": str(lst.id), "name": lst.title}


@router.get("/favorites")
def get_my_favorites(
    limit: int = 3,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    favorites, created = get_or_create_favorites_list(db, current_user.id)
    if created:
        db.commit()
    rows = db.scalars(
        select(ListBook)
        .where(ListBook.list_id == favorites.id)
        .options(
            selectinload(ListBook.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(ListBook.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
        .order_by(ListBook.added_at.desc())
        .limit(max(1, min(limit, 20)))
    ).all()
    return [serialize_book(row.book) for row in rows if row.book is not None]


@router.post("/favorites/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_to_favorites(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    favorites, _ = get_or_create_favorites_list(db, current_user.id)
    existing = db.scalar(
        select(ListBook).where(ListBook.list_id == favorites.id, ListBook.book_id == book_id)
    )
    if existing is not None:
        return
    count = db.scalar(
        select(func.count()).where(ListBook.list_id == favorites.id)
    ) or 0
    if count >= 3:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Favorites can contain at most 3 books")
    ok = BookListCRUD.add_book(db, favorites.id, book_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    db.commit()


@router.delete("/favorites/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_favorites(
    book_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    favorites, _ = get_or_create_favorites_list(db, current_user.id)
    BookListCRUD.remove_book(db, favorites.id, book_id)
    db.commit()
