"""Tests for AuthorCRUD."""

import pytest

from bookdb.db.crud import AuthorCRUD
from tests.test_db.conftest import make_author


class TestAuthorCRUDCreate:
    def test_create_minimal(self, session):
        author = make_author(session)
        assert author.id is not None
        assert author.name == "George Orwell"

    def test_create_with_goodreads_id(self, session):
        author = make_author(session, goodreads_id=9999)
        assert author.goodreads_id == 9999

    def test_create_goodreads_id_as_string(self, session):
        author = make_author(session, goodreads_id="1234")
        assert author.goodreads_id == 1234

    def test_create_empty_name_raises(self, session):
        with pytest.raises(ValueError, match="name"):
            AuthorCRUD.create(session, name="")

    def test_create_whitespace_name_raises(self, session):
        with pytest.raises(ValueError, match="name"):
            AuthorCRUD.create(session, name="   ")

    def test_create_none_name_raises(self, session):
        with pytest.raises(ValueError, match="name"):
            AuthorCRUD.create(session, name=None)

    def test_create_duplicate_goodreads_id_raises(self, session):
        make_author(session, name="Author A", goodreads_id=42)
        with pytest.raises(ValueError, match="goodreads_id"):
            make_author(session, name="Author B", goodreads_id=42)

    def test_create_same_name_different_goodreads_allowed(self, session):
        a1 = make_author(session, goodreads_id=1)
        a2 = make_author(session, goodreads_id=2)
        assert a1.id != a2.id


class TestAuthorCRUDRead:
    def test_get_by_id_found(self, session):
        author = make_author(session)
        result = AuthorCRUD.get_by_id(session, author.id)
        assert result is not None
        assert result.id == author.id

    def test_get_by_id_not_found(self, session):
        assert AuthorCRUD.get_by_id(session, 99999) is None

    def test_get_by_name_found(self, session):
        make_author(session, name="Tolkien")
        result = AuthorCRUD.get_by_name(session, "Tolkien")
        assert result is not None
        assert result.name == "Tolkien"

    def test_get_by_name_not_found(self, session):
        assert AuthorCRUD.get_by_name(session, "Nobody") is None

    def test_get_by_goodreads_id_found(self, session):
        make_author(session, goodreads_id=77)
        result = AuthorCRUD.get_by_goodreads_id(session, 77)
        assert result is not None
        assert result.goodreads_id == 77

    def test_get_by_goodreads_id_string_input(self, session):
        make_author(session, goodreads_id=77)
        result = AuthorCRUD.get_by_goodreads_id(session, "77")
        assert result is not None

    def test_get_by_goodreads_id_not_found(self, session):
        assert AuthorCRUD.get_by_goodreads_id(session, 99999) is None


class TestAuthorCRUDUpdate:
    def test_update_name(self, session):
        author = make_author(session, name="Old Name")
        updated = AuthorCRUD.update(session, author.id, name="New Name")
        assert updated.name == "New Name"

    def test_update_description(self, session):
        author = make_author(session)
        AuthorCRUD.update(session, author.id, description="A famous author.")
        assert AuthorCRUD.get_by_id(session, author.id).description == "A famous author."

    def test_update_goodreads_id(self, session):
        author = make_author(session)
        AuthorCRUD.update(session, author.id, goodreads_id=555)
        assert AuthorCRUD.get_by_id(session, author.id).goodreads_id == 555

    def test_update_empty_name_raises(self, session):
        author = make_author(session)
        with pytest.raises(ValueError, match="name"):
            AuthorCRUD.update(session, author.id, name="")

    def test_update_duplicate_goodreads_id_raises(self, session):
        a1 = make_author(session, name="A1", goodreads_id=10)
        a2 = make_author(session, name="A2", goodreads_id=20)
        with pytest.raises(ValueError, match="goodreads_id"):
            AuthorCRUD.update(session, a2.id, goodreads_id=10)

    def test_update_same_goodreads_id_no_conflict(self, session):
        author = make_author(session, goodreads_id=10)
        updated = AuthorCRUD.update(session, author.id, goodreads_id=10, name="Updated")
        assert updated.goodreads_id == 10

    def test_update_not_found_raises(self, session):
        with pytest.raises(ValueError, match="not found"):
            AuthorCRUD.update(session, 99999, name="X")


class TestAuthorCRUDDelete:
    def test_delete_found(self, session):
        author = make_author(session)
        assert AuthorCRUD.delete(session, author.id) is True
        assert AuthorCRUD.get_by_id(session, author.id) is None

    def test_delete_not_found(self, session):
        assert AuthorCRUD.delete(session, 99999) is False


class TestAuthorCRUDGetOrCreate:
    def test_creates_when_absent(self, session):
        author = AuthorCRUD.get_or_create(session, "New Author")
        assert author.id is not None
        assert author.name == "New Author"

    def test_returns_existing(self, session):
        a1 = make_author(session, name="Existing")
        a2 = AuthorCRUD.get_or_create(session, "Existing")
        assert a1.id == a2.id


class TestAuthorCRUDBulk:
    def test_bulk_get_or_create_by_names_creates_all(self, session):
        result = AuthorCRUD.bulk_get_or_create_by_names(session, ["Alice", "Bob", "Carol"])
        assert set(result.keys()) == {"Alice", "Bob", "Carol"}
        assert all(a.id is not None for a in result.values())

    def test_bulk_get_or_create_by_names_reuses_existing(self, session):
        a = make_author(session, name="Existing")
        result = AuthorCRUD.bulk_get_or_create_by_names(session, ["Existing", "New"])
        assert result["Existing"].id == a.id

    def test_bulk_get_or_create_by_names_deduplicates_input(self, session):
        result = AuthorCRUD.bulk_get_or_create_by_names(session, ["Dup", "Dup", "Dup"])
        assert len(result) == 1
        assert "Dup" in result

    def test_bulk_get_or_create_by_names_empty(self, session):
        assert AuthorCRUD.bulk_get_or_create_by_names(session, []) == {}

    def test_bulk_get_by_goodreads_ids(self, session):
        make_author(session, name="A1", goodreads_id=1)
        make_author(session, name="A2", goodreads_id=2)
        result = AuthorCRUD.bulk_get_by_goodreads_ids(session, [1, 2, 999])
        assert set(result.keys()) == {1, 2}

    def test_bulk_get_by_goodreads_ids_empty(self, session):
        assert AuthorCRUD.bulk_get_by_goodreads_ids(session, []) == {}
