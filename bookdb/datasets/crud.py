"""
Simple CRUD helpers for database models.
Nothing fancy, just readable SQLAlchemy ORM code.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from bookdb.db.models import (
    Book,
    Author,
    User,
    BookList,
    book_authors,
)

class AuthorCRUD:
    """Crud operations for Authors"""
    @staticmethod
    def get_by_id(session: Session, author_id: int) -> Author | None:
        return session.get(Author, author_id)

    @staticmethod
    def get_by_name(session: Session, name: str) -> Author | None:
        stmt = select(Author).where(Author.name == name)
        return session.scalar(stmt)

    @staticmethod
    def create(session: Session, name: str, **kwargs) -> Author:
        author = Author(name=name, **kwargs)
        session.add(author)
        session.flush()
        return author

    @staticmethod
    def get_or_create(session: Session, name: str) -> Author:
        author = AuthorCRUD.get_by_name(session, name)
        if author:
            return author
        return AuthorCRUD.create(session, name)

    @staticmethod
    def bulk_get_or_create(session: Session, names: list[str]) -> dict[str, Author]:
        names = list(set(names))
        if not names:
            return {}

        stmt = select(Author).where(Author.name.in_(names))
        existing = {a.name: a for a in session.scalars(stmt).all()}

        for name in names:
            if name not in existing:
                existing[name] = AuthorCRUD.create(session, name)

        return existing


# Books
class BookCRUD:
    """Crud operations for Books"""
    @staticmethod
    def get_by_id(session: Session, book_id: int) -> Book | None:
        return session.get(Book, book_id)

    @staticmethod
    def get_by_title(session: Session, title: str) -> list[Book]:
        stmt = select(Book).where(Book.title == title)
        return session.scalars(stmt).all()

    @staticmethod
    def search_by_title(session: Session, query: str, limit: int = 100) -> list[Book]:
        stmt = select(Book).where(Book.title.ilike(f"%{query}%")).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def create(session: Session, **kwargs) -> Book:
        book = Book(**kwargs)
        session.add(book)
        session.flush()
        return book

    @staticmethod
    def create_with_authors(
        session: Session,
        author_names: list[str],
        **book_kwargs,
    ) -> Book:
        book = BookCRUD.create(session, **book_kwargs)

        if author_names:
            authors = AuthorCRUD.bulk_get_or_create(session, author_names)
            book.authors = [authors[name] for name in author_names]

        session.flush()
        return book

    @staticmethod
    def get_by_author(session: Session, author_id: int, limit: int = 100) -> list[Book]:
        stmt = (
            select(Book)
            .join(book_authors)
            .where(book_authors.c.author_id == author_id)
            .limit(limit)
        )
        return session.scalars(stmt).all()


class UserCRUD:
    """Crud operations for Users"""
    @staticmethod
    def get_by_id(session: Session, user_id: int) -> User | None:
        return session.get(User, user_id)

    @staticmethod
    def get_by_email(session: Session, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return session.scalar(stmt)

    @staticmethod
    def create(session: Session, email: str, name: str) -> User:
        user = User(email=email, name=name)
        session.add(user)
        session.flush()
        return user

    @staticmethod
    def get_or_create(session: Session, email: str, name: str) -> User:
        user = UserCRUD.get_by_email(session, email)
        if user:
            return user
        return UserCRUD.create(session, email, name)


class BookListCRUD:
    """Crud operations for User's BookLists"""
    @staticmethod
    def get_by_id(session: Session, list_id: int) -> BookList | None:
        return session.get(BookList, list_id)

    @staticmethod
    def get_by_user(session: Session, user_id: int) -> list[BookList]:
        stmt = select(BookList).where(BookList.user_id == user_id)
        return session.scalars(stmt).all()

    @staticmethod
    def create(session: Session, user_id: int, name: str) -> BookList:
        book_list = BookList(user_id=user_id, name=name)
        session.add(book_list)
        session.flush()
        return book_list

    @staticmethod
    def add_book(session: Session, list_id: int, book_id: int) -> bool:
        book_list = BookListCRUD.get_by_id(session, list_id)
        book = BookCRUD.get_by_id(session, book_id)

        if not book_list or not book:
            return False

        if book not in book_list.books:
            book_list.books.append(book)
            session.flush()

        return True

    @staticmethod
    def remove_book(session: Session, list_id: int, book_id: int) -> bool:
        book_list = BookListCRUD.get_by_id(session, list_id)
        book = BookCRUD.get_by_id(session, book_id)

        if not book_list or not book:
            return False

        if book in book_list.books:
            book_list.books.remove(book)
            session.flush()
            return True

        return False