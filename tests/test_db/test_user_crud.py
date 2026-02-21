"""Tests for UserCRUD."""

import pytest

from bookdb.db.crud import UserCRUD
from tests.test_db.conftest import make_user


class TestUserCRUDCreate:
    def test_create_minimal(self, session):
        user = make_user(session)
        assert user.id is not None
        assert user.email == "alice@example.com"
        assert user.name == "Alice"
        assert user.username == "alice"

    def test_create_with_goodreads_id(self, session):
        user = make_user(session, goodreads_id=100)
        assert user.goodreads_id == 100

    def test_create_goodreads_id_as_string(self, session):
        user = make_user(session, goodreads_id="200")
        assert user.goodreads_id == 200

    def test_create_invalid_email_raises(self, session):
        with pytest.raises(ValueError, match="Invalid email"):
            make_user(session, email="not_an_email")

    def test_create_empty_email_raises(self, session):
        with pytest.raises(ValueError, match="email"):
            make_user(session, email="")

    def test_create_empty_name_raises(self, session):
        with pytest.raises(ValueError, match="name"):
            make_user(session, name="")

    def test_create_whitespace_name_raises(self, session):
        with pytest.raises(ValueError, match="name"):
            make_user(session, name="   ")

    def test_create_empty_username_raises(self, session):
        with pytest.raises(ValueError, match="username"):
            make_user(session, username="")

    def test_create_empty_password_hash_raises(self, session):
        with pytest.raises(ValueError, match="password_hash"):
            make_user(session, password_hash="")

    def test_create_duplicate_email_raises(self, session):
        make_user(session, email="dup@example.com", username="user1")
        with pytest.raises(ValueError, match="email"):
            make_user(session, email="dup@example.com", username="user2")

    def test_create_duplicate_username_raises(self, session):
        make_user(session, email="a@example.com", username="same_name")
        with pytest.raises(ValueError, match="username"):
            make_user(session, email="b@example.com", username="same_name")

    def test_create_duplicate_goodreads_id_raises(self, session):
        make_user(session, email="a@x.com", username="u1", goodreads_id=50)
        with pytest.raises(ValueError, match="goodreads_id"):
            make_user(session, email="b@x.com", username="u2", goodreads_id=50)


class TestUserCRUDRead:
    def test_get_by_id_found(self, session):
        user = make_user(session)
        assert UserCRUD.get_by_id(session, user.id).id == user.id

    def test_get_by_id_not_found(self, session):
        assert UserCRUD.get_by_id(session, 99999) is None

    def test_get_by_email_found(self, session):
        make_user(session, email="find@example.com")
        result = UserCRUD.get_by_email(session, "find@example.com")
        assert result is not None

    def test_get_by_email_not_found(self, session):
        assert UserCRUD.get_by_email(session, "nobody@example.com") is None

    def test_get_by_username_found(self, session):
        make_user(session, username="findme")
        result = UserCRUD.get_by_username(session, "findme")
        assert result is not None
        assert result.username == "findme"

    def test_get_by_username_not_found(self, session):
        assert UserCRUD.get_by_username(session, "ghost") is None

    def test_get_by_goodreads_id_found(self, session):
        make_user(session, goodreads_id=77)
        result = UserCRUD.get_by_goodreads_id(session, 77)
        assert result is not None
        assert result.goodreads_id == 77

    def test_get_by_goodreads_id_string(self, session):
        make_user(session, goodreads_id=88)
        assert UserCRUD.get_by_goodreads_id(session, "88") is not None

    def test_get_by_goodreads_id_not_found(self, session):
        assert UserCRUD.get_by_goodreads_id(session, 99999) is None


class TestUserCRUDUpdate:
    def test_update_name(self, session):
        user = make_user(session)
        UserCRUD.update(session, user.id, name="Bob")
        assert UserCRUD.get_by_id(session, user.id).name == "Bob"

    def test_update_email_valid(self, session):
        user = make_user(session)
        UserCRUD.update(session, user.id, email="new@example.com")
        assert UserCRUD.get_by_id(session, user.id).email == "new@example.com"

    def test_update_email_invalid_raises(self, session):
        user = make_user(session)
        with pytest.raises(ValueError, match="Invalid email"):
            UserCRUD.update(session, user.id, email="bad_email")

    def test_update_email_duplicate_raises(self, session):
        u1 = make_user(session, email="a@x.com", username="u1")
        u2 = make_user(session, email="b@x.com", username="u2")
        with pytest.raises(ValueError, match="email"):
            UserCRUD.update(session, u2.id, email="a@x.com")

    def test_update_same_email_no_conflict(self, session):
        user = make_user(session, email="same@x.com")
        UserCRUD.update(session, user.id, email="same@x.com", name="Updated")
        assert UserCRUD.get_by_id(session, user.id).name == "Updated"

    def test_update_username_duplicate_raises(self, session):
        u1 = make_user(session, email="a@x.com", username="taken")
        u2 = make_user(session, email="b@x.com", username="free")
        with pytest.raises(ValueError, match="username"):
            UserCRUD.update(session, u2.id, username="taken")

    def test_update_same_username_no_conflict(self, session):
        user = make_user(session, username="myname")
        UserCRUD.update(session, user.id, username="myname", name="Updated")
        assert UserCRUD.get_by_id(session, user.id).name == "Updated"

    def test_update_empty_name_raises(self, session):
        user = make_user(session)
        with pytest.raises(ValueError, match="name"):
            UserCRUD.update(session, user.id, name="")

    def test_update_empty_password_hash_raises(self, session):
        user = make_user(session)
        with pytest.raises(ValueError, match="password_hash"):
            UserCRUD.update(session, user.id, password_hash="")

    def test_update_not_found_raises(self, session):
        with pytest.raises(ValueError, match="not found"):
            UserCRUD.update(session, 99999, name="X")


class TestUserCRUDDelete:
    def test_delete_found(self, session):
        user = make_user(session)
        assert UserCRUD.delete(session, user.id) is True
        assert UserCRUD.get_by_id(session, user.id) is None

    def test_delete_not_found(self, session):
        assert UserCRUD.delete(session, 99999) is False


class TestUserCRUDGetOrCreate:
    def test_creates_when_absent(self, session):
        user = UserCRUD.get_or_create(
            session,
            email="new@x.com",
            name="New",
            username="newuser",
            password_hash="hash",
        )
        assert user.id is not None

    def test_returns_existing_by_email(self, session):
        u1 = make_user(session, email="existing@x.com")
        u2 = UserCRUD.get_or_create(
            session,
            email="existing@x.com",
            name="Different Name",
            username="different_username",
            password_hash="different_hash",
        )
        assert u1.id == u2.id


class TestUserCRUDImportStub:
    def test_creates_stub_user(self, session):
        user = UserCRUD.create_import_stub(session, goodreads_id=999)
        assert user.id is not None
        assert user.goodreads_id == 999
        assert user.email.endswith("@import.local")
        assert user.password_hash == "import_stub"

    def test_returns_existing_stub(self, session):
        u1 = UserCRUD.create_import_stub(session, goodreads_id=999)
        u2 = UserCRUD.create_import_stub(session, goodreads_id=999)
        assert u1.id == u2.id

    def test_no_goodreads_id_raises(self, session):
        with pytest.raises(ValueError, match="goodreads_id is required"):
            UserCRUD.create_import_stub(session, goodreads_id=None)


class TestUserCRUDBulk:
    def test_bulk_get_by_goodreads_ids(self, session):
        make_user(session, email="a@x.com", username="u1", goodreads_id=10)
        make_user(session, email="b@x.com", username="u2", goodreads_id=20)
        result = UserCRUD.bulk_get_by_goodreads_ids(session, [10, 20, 999])
        assert set(result.keys()) == {10, 20}

    def test_bulk_get_by_goodreads_ids_empty(self, session):
        assert UserCRUD.bulk_get_by_goodreads_ids(session, []) == {}

    def test_bulk_get_or_create_from_goodreads_ids_creates_stubs(self, session):
        result = UserCRUD.bulk_get_or_create_from_goodreads_ids(session, [301, 302])
        assert set(result.keys()) == {301, 302}
        assert all(u.password_hash == "import_stub" for u in result.values())

    def test_bulk_get_or_create_from_goodreads_ids_reuses_existing(self, session):
        existing = UserCRUD.create_import_stub(session, goodreads_id=401)
        result = UserCRUD.bulk_get_or_create_from_goodreads_ids(session, [401, 402])
        assert result[401].id == existing.id
