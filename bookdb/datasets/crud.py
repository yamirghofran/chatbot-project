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
    Review,
    Rating,
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
    def get_by_external_id(session: Session, external_id: str) -> Author | None:
        stmt = select(Author).where(Author.external_id == external_id)
        return session.scalar(stmt)

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

        to_create = [name for name in names if name not in existing]
        if to_create:
            new_authors = [Author(name=name) for name in to_create]
            session.add_all(new_authors)
            session.flush()
            for author in new_authors:
                existing[author.name] = author

        return existing

    @staticmethod
    def bulk_get_by_external_ids(session: Session, external_ids: list[str]) -> dict[str, Author]:
        external_ids = list(set(external_ids))
        if not external_ids:
            return {}
        stmt = select(Author).where(Author.external_id.in_(external_ids))
        return {a.external_id: a for a in session.scalars(stmt).all()}


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
    def get_by_book_id(session: Session, book_id: str) -> Book | None:
        stmt = select(Book).where(Book.book_id == book_id)
        return session.scalar(stmt)

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


class ReviewCRUD:
    """CRUD operations for Reviews."""

    @staticmethod
    def get_by_id(session: Session, review_id: int) -> Review | None:
        return session.get(Review, review_id)

    @staticmethod
    def get_by_user(session: Session, user_id: int, limit: int = 100) -> list[Review]:
        stmt = select(Review).where(Review.user_id == user_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_book(session: Session, book_id: int, limit: int = 100) -> list[Review]:
        stmt = select(Review).where(Review.book_id == book_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_user_and_book(
        session: Session, user_id: int, book_id: int
    ) -> list[Review]:
        stmt = select(Review).where(
            Review.user_id == user_id, Review.book_id == book_id
        )
        return session.scalars(stmt).all()

    @staticmethod
    def create(session: Session, user_id: int, book_id: int, text: str) -> Review:
        review = Review(user_id=user_id, book_id=book_id, text=text)
        session.add(review)
        session.flush()
        return review


class RatingCRUD:
    """CRUD operations for Ratings."""

    @staticmethod
    def get(session: Session, user_id: int, book_id: int) -> Rating | None:
        return session.get(Rating, (user_id, book_id))

    @staticmethod
    def get_by_user(session: Session, user_id: int, limit: int = 100) -> list[Rating]:
        stmt = select(Rating).where(Rating.user_id == user_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_book(session: Session, book_id: int, limit: int = 100) -> list[Rating]:
        stmt = select(Rating).where(Rating.book_id == book_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def upsert(session: Session, user_id: int, book_id: int, rating: int) -> Rating:
        """Create or update a rating. One user can only rate a book once."""
        existing = session.get(Rating, (user_id, book_id))
        if existing:
            existing.rating = rating
            session.flush()
            return existing
        new_rating = Rating(user_id=user_id, book_id=book_id, rating=rating)
        session.add(new_rating)
        session.flush()
        return new_rating

    @staticmethod
    def delete(session: Session, user_id: int, book_id: int) -> bool:
        existing = session.get(Rating, (user_id, book_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True