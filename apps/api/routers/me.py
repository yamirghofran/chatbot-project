from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookCRUD, BookListCRUD, BookRatingCRUD, ReviewCRUD, ShellCRUD
from bookdb.db.models import (
    Book,
    BookAuthor,
    BookList,
    BookRating,
    BookTag,
    ListBook,
    Review,
    ShellBook,
    User,
)

from ..core.deps import get_current_user, get_db
from ..core.favorites import get_or_create_favorites_list, is_favorites_list
from ..core.serialize import serialize_book, serialize_list
from ..schemas.list import CreateListRequest


def _strip_goodreads_isbn(raw: str) -> str | None:
    """Strip Goodreads CSV ISBN formatting: '=\"9780593734223\"' → '9780593734223'."""
    if not raw:
        return None
    cleaned = raw.strip().lstrip('=').strip('"')
    return cleaned or None

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


@router.post("/import/goodreads")
async def import_goodreads(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Import a Goodreads library CSV export for the authenticated user.

    Matches rows to existing books via Goodreads Book Id then ISBN-13 fallback.
    Imports ratings (1–5) and non-empty reviews; skips books not in the catalog.
    """
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")  # strip BOM if present
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid CSV file")

    matched = skipped = ratings_imported = reviews_imported = 0

    for row in rows:
        # --- locate book ---
        book: Book | None = None

        raw_gid = (row.get("Book Id") or "").strip()
        if raw_gid:
            try:
                book = BookCRUD.get_by_goodreads_id(db, int(raw_gid))
            except (ValueError, TypeError):
                pass

        if book is None:
            isbn13 = _strip_goodreads_isbn(row.get("ISBN13") or "")
            if isbn13:
                book = db.scalar(select(Book).where(Book.isbn13 == isbn13))

        if book is None:
            skipped += 1
            continue

        matched += 1

        # --- rating ---
        try:
            my_rating = int((row.get("My Rating") or "0").strip())
        except (ValueError, TypeError):
            my_rating = 0

        if 1 <= my_rating <= 5:
            try:
                BookRatingCRUD.upsert(db, current_user.id, book.id, my_rating)
                ratings_imported += 1
            except ValueError:
                pass

        # --- review ---
        review_text = (row.get("My Review") or "").strip()
        if review_text:
            existing = ReviewCRUD.get_by_user_and_book(db, current_user.id, book.id)
            if not existing:
                try:
                    ReviewCRUD.create(db, current_user.id, book.id, review_text)
                    reviews_imported += 1
                except ValueError:
                    pass

    db.commit()

    return {
        "matched": matched,
        "skipped": skipped,
        "ratings_imported": ratings_imported,
        "reviews_imported": reviews_imported,
    }
