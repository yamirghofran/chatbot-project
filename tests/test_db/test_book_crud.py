"""Tests for BookCRUD."""

import pytest

from bookdb.db.crud import BookCRUD, BookAuthorCRUD, AuthorCRUD
from tests.test_db.conftest import make_author, make_book


class TestBookCRUDCreate:
    def test_create_minimal(self, session):
        book = make_book(session)
        assert book.id is not None
        assert book.title == "1984"
        assert book.goodreads_id == 1001

    def test_create_with_all_fields(self, session):
        book = BookCRUD.create(
            session,
            goodreads_id=2002,
            title="Brave New World",
            description="A dystopian novel",
            publication_year=1932,
            publisher="Chatto & Windus",
            isbn13="9780060850524",
        )
        assert book.title == "Brave New World"
        assert book.publication_year == 1932

    def test_create_goodreads_id_as_string(self, session):
        book = BookCRUD.create(session, goodreads_id="3003", title="Dune")
        assert book.goodreads_id == 3003

    def test_create_missing_goodreads_id_raises(self, session):
        with pytest.raises(ValueError, match="goodreads_id is required"):
            BookCRUD.create(session, goodreads_id=None, title="Title")

    def test_create_empty_title_raises(self, session):
        with pytest.raises(ValueError, match="title"):
            BookCRUD.create(session, goodreads_id=4004, title="")

    def test_create_whitespace_title_raises(self, session):
        with pytest.raises(ValueError, match="title"):
            BookCRUD.create(session, goodreads_id=4004, title="   ")

    def test_create_duplicate_goodreads_id_raises(self, session):
        make_book(session, goodreads_id=5005, title="Book A")
        with pytest.raises(ValueError, match="goodreads_id"):
            make_book(session, goodreads_id=5005, title="Book B")

    def test_create_invalid_year_raises(self, session):
        with pytest.raises(ValueError, match="1000 and 9999"):
            BookCRUD.create(session, goodreads_id=6006, title="Old Book", publication_year=500)

    def test_create_year_too_high_raises(self, session):
        with pytest.raises(ValueError, match="1000 and 9999"):
            BookCRUD.create(session, goodreads_id=6007, title="Future Book", publication_year=99999)

    def test_create_year_none_allowed(self, session):
        book = BookCRUD.create(session, goodreads_id=7007, title="Timeless", publication_year=None)
        assert book.publication_year is None


class TestBookCRUDRead:
    def test_get_by_id_found(self, session):
        book = make_book(session)
        assert BookCRUD.get_by_id(session, book.id).id == book.id

    def test_get_by_id_not_found(self, session):
        assert BookCRUD.get_by_id(session, 99999) is None

    def test_get_by_goodreads_id_found(self, session):
        make_book(session, goodreads_id=8008)
        result = BookCRUD.get_by_goodreads_id(session, 8008)
        assert result is not None
        assert result.goodreads_id == 8008

    def test_get_by_goodreads_id_string(self, session):
        make_book(session, goodreads_id=8009)
        assert BookCRUD.get_by_goodreads_id(session, "8009") is not None

    def test_get_by_goodreads_id_not_found(self, session):
        assert BookCRUD.get_by_goodreads_id(session, 99999) is None

    def test_get_by_title_exact(self, session):
        make_book(session, title="Unique Title")
        results = BookCRUD.get_by_title(session, "Unique Title")
        assert len(results) == 1

    def test_get_by_title_not_found(self, session):
        assert BookCRUD.get_by_title(session, "Nonexistent") == []

    def test_search_by_title_partial(self, session):
        make_book(session, goodreads_id=9001, title="The Great Gatsby")
        make_book(session, goodreads_id=9002, title="Great Expectations")
        results = BookCRUD.search_by_title(session, "Great")
        assert len(results) == 2

    def test_search_by_title_case_insensitive(self, session):
        make_book(session, goodreads_id=9003, title="Animal Farm")
        results = BookCRUD.search_by_title(session, "animal")
        assert len(results) == 1

    def test_search_by_title_limit(self, session):
        for i in range(5):
            make_book(session, goodreads_id=9100 + i, title=f"Book {i}")
        results = BookCRUD.search_by_title(session, "Book", limit=3)
        assert len(results) == 3


class TestBookCRUDUpdate:
    def test_update_title(self, session):
        book = make_book(session, title="Old Title")
        BookCRUD.update(session, book.id, title="New Title")
        assert BookCRUD.get_by_id(session, book.id).title == "New Title"

    def test_update_empty_title_raises(self, session):
        book = make_book(session)
        with pytest.raises(ValueError, match="title"):
            BookCRUD.update(session, book.id, title="")

    def test_update_year(self, session):
        book = make_book(session)
        BookCRUD.update(session, book.id, publication_year=2020)
        assert BookCRUD.get_by_id(session, book.id).publication_year == 2020

    def test_update_year_invalid_raises(self, session):
        book = make_book(session)
        with pytest.raises(ValueError, match="1000 and 9999"):
            BookCRUD.update(session, book.id, publication_year=0)

    def test_update_goodreads_id_to_new_unique(self, session):
        book = make_book(session, goodreads_id=1111)
        BookCRUD.update(session, book.id, goodreads_id=2222)
        assert BookCRUD.get_by_id(session, book.id).goodreads_id == 2222

    def test_update_goodreads_id_to_taken_raises(self, session):
        b1 = make_book(session, goodreads_id=3333, title="B1")
        b2 = make_book(session, goodreads_id=4444, title="B2")
        with pytest.raises(ValueError, match="goodreads_id"):
            BookCRUD.update(session, b2.id, goodreads_id=3333)

    def test_update_same_goodreads_id_no_conflict(self, session):
        book = make_book(session, goodreads_id=5555)
        BookCRUD.update(session, book.id, goodreads_id=5555, title="Same ID")
        assert BookCRUD.get_by_id(session, book.id).title == "Same ID"

    def test_update_not_found_raises(self, session):
        with pytest.raises(ValueError, match="not found"):
            BookCRUD.update(session, 99999, title="X")


class TestBookCRUDDelete:
    def test_delete_found(self, session):
        book = make_book(session)
        assert BookCRUD.delete(session, book.id) is True
        assert BookCRUD.get_by_id(session, book.id) is None

    def test_delete_not_found(self, session):
        assert BookCRUD.delete(session, 99999) is False


class TestBookCRUDWithAuthors:
    def test_create_with_authors_links(self, session):
        book = BookCRUD.create_with_authors(
            session,
            author_names=["Orwell", "Huxley"],
            goodreads_id=6001,
            title="Anthology",
        )
        session.refresh(book)
        assert len(book.authors) == 2

    def test_create_with_authors_deduplicates_names(self, session):
        book = BookCRUD.create_with_authors(
            session,
            author_names=["Orwell", "Orwell"],
            goodreads_id=6002,
            title="Solo",
        )
        session.refresh(book)
        assert len(book.authors) == 1

    def test_create_with_authors_reuses_existing_author(self, session):
        existing = make_author(session, name="Tolkien")
        book = BookCRUD.create_with_authors(
            session,
            author_names=["Tolkien"],
            goodreads_id=6003,
            title="Middle Earth Tales",
        )
        session.refresh(book)
        assert book.authors[0].author.id == existing.id

    def test_get_by_author(self, session):
        author = make_author(session, name="Kafka")
        book = BookCRUD.create_with_authors(
            session, author_names=["Kafka"], goodreads_id=7001, title="The Trial"
        )
        results = BookCRUD.get_by_author(session, author.id)
        assert any(b.id == book.id for b in results)

    def test_get_by_author_empty(self, session):
        author = make_author(session)
        assert BookCRUD.get_by_author(session, author.id) == []


class TestBookAuthorCRUD:
    def test_link_creates_association(self, session):
        book = make_book(session)
        author = make_author(session)
        link = BookAuthorCRUD.link(session, book.id, author.id)
        assert link.book_id == book.id
        assert link.author_id == author.id

    def test_link_idempotent(self, session):
        book = make_book(session)
        author = make_author(session)
        l1 = BookAuthorCRUD.link(session, book.id, author.id)
        l2 = BookAuthorCRUD.link(session, book.id, author.id)
        assert l1.book_id == l2.book_id

    def test_unlink_removes_association(self, session):
        book = make_book(session)
        author = make_author(session)
        BookAuthorCRUD.link(session, book.id, author.id)
        assert BookAuthorCRUD.unlink(session, book.id, author.id) is True

    def test_unlink_not_found(self, session):
        assert BookAuthorCRUD.unlink(session, 99999, 99999) is False
