"""Tests for BookListCRUD, ShellCRUD, and BookRatingCRUD."""

import pytest

from bookdb.db.crud import BookListCRUD, BookRatingCRUD, ShellCRUD
from tests.test_db.conftest import (
    make_book,
    make_book_list,
    make_shell,
    make_user,
)


# ---------------------------------------------------------------------------
# BookListCRUD
# ---------------------------------------------------------------------------


class TestBookListCRUDCreate:
    def test_create_minimal(self, session):
        user = make_user(session)
        book_list = make_book_list(session, user)
        assert book_list.id is not None
        assert book_list.title == "Favourites"
        assert book_list.user_id == user.id

    def test_create_with_description(self, session):
        user = make_user(session)
        bl = BookListCRUD.create(session, user.id, title="SF", description="Sci-fi reads")
        assert bl.description == "Sci-fi reads"

    def test_create_empty_title_raises(self, session):
        user = make_user(session)
        with pytest.raises(ValueError, match="title"):
            BookListCRUD.create(session, user.id, title="")

    def test_create_whitespace_title_raises(self, session):
        user = make_user(session)
        with pytest.raises(ValueError, match="title"):
            BookListCRUD.create(session, user.id, title="   ")

    def test_create_duplicate_title_same_user_raises(self, session):
        user = make_user(session)
        BookListCRUD.create(session, user.id, title="Duplicated")
        with pytest.raises(ValueError, match="already has a list"):
            BookListCRUD.create(session, user.id, title="Duplicated")

    def test_create_same_title_different_users_allowed(self, session):
        u1 = make_user(session, email="a@x.com", username="u1")
        u2 = make_user(session, email="b@x.com", username="u2")
        BookListCRUD.create(session, u1.id, title="Same Title")
        bl = BookListCRUD.create(session, u2.id, title="Same Title")
        assert bl.id is not None


class TestBookListCRUDRead:
    def test_get_by_id_found(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        assert BookListCRUD.get_by_id(session, bl.id).id == bl.id

    def test_get_by_id_not_found(self, session):
        assert BookListCRUD.get_by_id(session, 99999) is None

    def test_get_by_user(self, session):
        user = make_user(session)
        BookListCRUD.create(session, user.id, title="List A")
        BookListCRUD.create(session, user.id, title="List B")
        results = BookListCRUD.get_by_user(session, user.id)
        assert len(results) == 2

    def test_get_by_user_empty(self, session):
        user = make_user(session)
        assert BookListCRUD.get_by_user(session, user.id) == []


class TestBookListAddRemoveBook:
    def test_add_book(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        book = make_book(session)
        assert BookListCRUD.add_book(session, bl.id, book.id) is True

    def test_add_book_idempotent(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        book = make_book(session)
        BookListCRUD.add_book(session, bl.id, book.id)
        assert BookListCRUD.add_book(session, bl.id, book.id) is True

    def test_add_book_invalid_list_returns_false(self, session):
        book = make_book(session)
        assert BookListCRUD.add_book(session, 99999, book.id) is False

    def test_add_book_invalid_book_returns_false(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        assert BookListCRUD.add_book(session, bl.id, 99999) is False

    def test_remove_book(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        book = make_book(session)
        BookListCRUD.add_book(session, bl.id, book.id)
        assert BookListCRUD.remove_book(session, bl.id, book.id) is True

    def test_remove_book_not_in_list(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        assert BookListCRUD.remove_book(session, bl.id, 99999) is False

    def test_add_multiple_books(self, session):
        user = make_user(session)
        bl = make_book_list(session, user)
        b1 = make_book(session, goodreads_id=1001)
        b2 = make_book(session, goodreads_id=1002, title="Book 2")
        b3 = make_book(session, goodreads_id=1003, title="Book 3")
        for book in [b1, b2, b3]:
            BookListCRUD.add_book(session, bl.id, book.id)
        session.refresh(bl)
        assert len(bl.books) == 3


# ---------------------------------------------------------------------------
# ShellCRUD
# ---------------------------------------------------------------------------


class TestShellCRUDCreate:
    def test_create(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        assert shell.id is not None
        assert shell.user_id == user.id
        assert shell.name == "My Shell"

    def test_create_custom_name(self, session):
        user = make_user(session)
        shell = ShellCRUD.create(session, user_id=user.id, name="Reading Shelf")
        assert shell.name == "Reading Shelf"


class TestShellCRUDRead:
    def test_get_by_id_found(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        assert ShellCRUD.get_by_id(session, shell.id).id == shell.id

    def test_get_by_id_not_found(self, session):
        assert ShellCRUD.get_by_id(session, 99999) is None

    def test_get_by_user_found(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        result = ShellCRUD.get_by_user(session, user.id)
        assert result is not None
        assert result.id == shell.id

    def test_get_by_user_not_found(self, session):
        user = make_user(session)
        assert ShellCRUD.get_by_user(session, user.id) is None


class TestShellCRUDGetOrCreate:
    def test_creates_when_absent(self, session):
        user = make_user(session)
        shell = ShellCRUD.get_or_create_for_user(session, user.id)
        assert shell.id is not None

    def test_returns_existing(self, session):
        user = make_user(session)
        s1 = make_shell(session, user)
        s2 = ShellCRUD.get_or_create_for_user(session, user.id)
        assert s1.id == s2.id


class TestShellAddRemoveBook:
    def test_add_book(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        book = make_book(session)
        assert ShellCRUD.add_book(session, shell.id, book.id) is True

    def test_add_book_idempotent(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        book = make_book(session)
        ShellCRUD.add_book(session, shell.id, book.id)
        assert ShellCRUD.add_book(session, shell.id, book.id) is True

    def test_add_book_invalid_shell_returns_false(self, session):
        book = make_book(session)
        assert ShellCRUD.add_book(session, 99999, book.id) is False

    def test_add_book_invalid_book_returns_false(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        assert ShellCRUD.add_book(session, shell.id, 99999) is False

    def test_remove_book(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        book = make_book(session)
        ShellCRUD.add_book(session, shell.id, book.id)
        assert ShellCRUD.remove_book(session, shell.id, book.id) is True

    def test_remove_book_not_in_shell(self, session):
        user = make_user(session)
        shell = make_shell(session, user)
        assert ShellCRUD.remove_book(session, shell.id, 99999) is False


# ---------------------------------------------------------------------------
# BookRatingCRUD
# ---------------------------------------------------------------------------


class TestBookRatingCRUD:
    def test_upsert_creates(self, session):
        user = make_user(session)
        book = make_book(session)
        rating = BookRatingCRUD.upsert(session, user.id, book.id, 4)
        assert rating.rating == 4
        assert rating.user_id == user.id
        assert rating.book_id == book.id

    def test_upsert_updates_existing(self, session):
        user = make_user(session)
        book = make_book(session)
        BookRatingCRUD.upsert(session, user.id, book.id, 3)
        updated = BookRatingCRUD.upsert(session, user.id, book.id, 5)
        assert updated.rating == 5

    def test_upsert_boundary_1(self, session):
        user = make_user(session)
        book = make_book(session)
        rating = BookRatingCRUD.upsert(session, user.id, book.id, 1)
        assert rating.rating == 1

    def test_upsert_boundary_5(self, session):
        user = make_user(session)
        book = make_book(session)
        rating = BookRatingCRUD.upsert(session, user.id, book.id, 5)
        assert rating.rating == 5

    def test_upsert_below_range_raises(self, session):
        user = make_user(session)
        book = make_book(session)
        with pytest.raises(ValueError, match="between 1 and 5"):
            BookRatingCRUD.upsert(session, user.id, book.id, 0)

    def test_upsert_above_range_raises(self, session):
        user = make_user(session)
        book = make_book(session)
        with pytest.raises(ValueError, match="between 1 and 5"):
            BookRatingCRUD.upsert(session, user.id, book.id, 6)

    def test_get_found(self, session):
        user = make_user(session)
        book = make_book(session)
        BookRatingCRUD.upsert(session, user.id, book.id, 3)
        rating = BookRatingCRUD.get(session, user.id, book.id)
        assert rating is not None
        assert rating.rating == 3

    def test_get_not_found(self, session):
        assert BookRatingCRUD.get(session, 99999, 99999) is None

    def test_get_by_user(self, session):
        user = make_user(session)
        b1 = make_book(session, goodreads_id=2001)
        b2 = make_book(session, goodreads_id=2002, title="B2")
        BookRatingCRUD.upsert(session, user.id, b1.id, 4)
        BookRatingCRUD.upsert(session, user.id, b2.id, 2)
        results = BookRatingCRUD.get_by_user(session, user.id)
        assert len(results) == 2

    def test_get_by_book(self, session):
        book = make_book(session)
        u1 = make_user(session, email="a@x.com", username="u1")
        u2 = make_user(session, email="b@x.com", username="u2")
        BookRatingCRUD.upsert(session, u1.id, book.id, 3)
        BookRatingCRUD.upsert(session, u2.id, book.id, 5)
        results = BookRatingCRUD.get_by_book(session, book.id)
        assert len(results) == 2

    def test_delete_found(self, session):
        user = make_user(session)
        book = make_book(session)
        BookRatingCRUD.upsert(session, user.id, book.id, 4)
        assert BookRatingCRUD.delete(session, user.id, book.id) is True
        assert BookRatingCRUD.get(session, user.id, book.id) is None

    def test_delete_not_found(self, session):
        assert BookRatingCRUD.delete(session, 99999, 99999) is False
