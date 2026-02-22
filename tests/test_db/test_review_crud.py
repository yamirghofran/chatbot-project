"""Tests for ReviewCRUD, ReviewCommentCRUD, and ReviewLikeCRUD."""

import pytest

from bookdb.db.crud import ReviewCRUD, ReviewCommentCRUD, ReviewLikeCRUD
from tests.test_db.conftest import make_author, make_book, make_review, make_user


# ---------------------------------------------------------------------------
# ReviewCRUD
# ---------------------------------------------------------------------------


class TestReviewCRUDCreate:
    def test_create_minimal(self, session):
        user = make_user(session)
        book = make_book(session)
        review = make_review(session, user, book)
        assert review.id is not None
        assert review.review_text == "Great read!"
        assert review.user_id == user.id
        assert review.book_id == book.id

    def test_create_with_goodreads_id(self, session):
        user = make_user(session)
        book = make_book(session)
        review = ReviewCRUD.create(
            session,
            user_id=user.id,
            book_id=book.id,
            review_text="Amazing!",
            goodreads_id="gr_review_1",
        )
        assert review.goodreads_id == "gr_review_1"

    def test_create_empty_text_raises(self, session):
        user = make_user(session)
        book = make_book(session)
        with pytest.raises(ValueError, match="review_text"):
            ReviewCRUD.create(session, user_id=user.id, book_id=book.id, review_text="")

    def test_create_whitespace_text_raises(self, session):
        user = make_user(session)
        book = make_book(session)
        with pytest.raises(ValueError, match="review_text"):
            ReviewCRUD.create(session, user_id=user.id, book_id=book.id, review_text="   ")

    def test_create_duplicate_goodreads_id_raises(self, session):
        user = make_user(session)
        book1 = make_book(session, goodreads_id=1001)
        book2 = make_book(session, goodreads_id=1002)
        ReviewCRUD.create(
            session, user_id=user.id, book_id=book1.id,
            review_text="First", goodreads_id="same_id"
        )
        with pytest.raises(ValueError, match="goodreads_id"):
            ReviewCRUD.create(
                session, user_id=user.id, book_id=book2.id,
                review_text="Second", goodreads_id="same_id"
            )


class TestReviewCRUDRead:
    def test_get_by_id_found(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        assert ReviewCRUD.get_by_id(session, review.id).id == review.id

    def test_get_by_id_not_found(self, session):
        assert ReviewCRUD.get_by_id(session, 99999) is None

    def test_get_by_goodreads_id(self, session):
        user, book = make_user(session), make_book(session)
        ReviewCRUD.create(
            session, user_id=user.id, book_id=book.id,
            review_text="Text", goodreads_id="gr123"
        )
        result = ReviewCRUD.get_by_goodreads_id(session, "gr123")
        assert result is not None
        assert result.goodreads_id == "gr123"

    def test_get_by_user(self, session):
        user, book = make_user(session), make_book(session)
        make_review(session, user, book)
        make_review(session, user, make_book(session, goodreads_id=2222, title="B2"), text="Also good")
        results = ReviewCRUD.get_by_user(session, user.id)
        assert len(results) == 2

    def test_get_by_user_limit(self, session):
        user = make_user(session)
        for i in range(5):
            book = make_book(session, goodreads_id=3000 + i, title=f"Book {i}")
            make_review(session, user, book, text=f"Review {i}")
        results = ReviewCRUD.get_by_user(session, user.id, limit=3)
        assert len(results) == 3

    def test_get_by_book(self, session):
        book = make_book(session)
        u1 = make_user(session, email="a@x.com", username="u1")
        u2 = make_user(session, email="b@x.com", username="u2")
        make_review(session, u1, book)
        make_review(session, u2, book, text="Also great")
        assert len(ReviewCRUD.get_by_book(session, book.id)) == 2

    def test_get_by_user_and_book(self, session):
        user, book = make_user(session), make_book(session)
        make_review(session, user, book)
        results = ReviewCRUD.get_by_user_and_book(session, user.id, book.id)
        assert len(results) == 1

    def test_get_by_user_and_book_no_match(self, session):
        user, book = make_user(session), make_book(session)
        assert ReviewCRUD.get_by_user_and_book(session, user.id, book.id) == []


class TestReviewCRUDUpdate:
    def test_update_text(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book, text="Old text")
        ReviewCRUD.update(session, review.id, review_text="New text")
        assert ReviewCRUD.get_by_id(session, review.id).review_text == "New text"

    def test_update_empty_text_raises(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        with pytest.raises(ValueError, match="review_text"):
            ReviewCRUD.update(session, review.id, review_text="")

    def test_update_goodreads_id(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        ReviewCRUD.update(session, review.id, goodreads_id="new_gr_id")
        assert ReviewCRUD.get_by_id(session, review.id).goodreads_id == "new_gr_id"

    def test_update_duplicate_goodreads_id_raises(self, session):
        user = make_user(session)
        b1 = make_book(session, goodreads_id=4001)
        b2 = make_book(session, goodreads_id=4002)
        ReviewCRUD.create(session, user_id=user.id, book_id=b1.id,
                          review_text="R1", goodreads_id="taken_id")
        r2 = ReviewCRUD.create(session, user_id=user.id, book_id=b2.id, review_text="R2")
        with pytest.raises(ValueError, match="goodreads_id"):
            ReviewCRUD.update(session, r2.id, goodreads_id="taken_id")

    def test_update_not_found_raises(self, session):
        with pytest.raises(ValueError, match="not found"):
            ReviewCRUD.update(session, 99999, review_text="X")


class TestReviewCRUDDelete:
    def test_delete_found(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        assert ReviewCRUD.delete(session, review.id) is True
        assert ReviewCRUD.get_by_id(session, review.id) is None

    def test_delete_not_found(self, session):
        assert ReviewCRUD.delete(session, 99999) is False

    def test_delete_cascades_comments(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        ReviewCommentCRUD.create(session, review.id, user.id, "A comment")
        ReviewCRUD.delete(session, review.id)
        # After cascade, comment should also be gone
        assert ReviewCRUD.get_by_id(session, review.id) is None

    def test_delete_cascades_likes(self, session):
        user, book = make_user(session), make_book(session)
        review = make_review(session, user, book)
        ReviewLikeCRUD.add(session, review.id, user.id)
        ReviewCRUD.delete(session, review.id)
        assert ReviewCRUD.get_by_id(session, review.id) is None


# ---------------------------------------------------------------------------
# ReviewCommentCRUD
# ---------------------------------------------------------------------------


class TestReviewCommentCRUD:
    def _setup(self, session):
        user = make_user(session)
        book = make_book(session)
        review = make_review(session, user, book)
        return user, book, review

    def test_create_comment(self, session):
        user, _, review = self._setup(session)
        comment = ReviewCommentCRUD.create(session, review.id, user.id, "Great review!")
        assert comment.id is not None
        assert comment.comment_text == "Great review!"

    def test_create_empty_text_raises(self, session):
        user, _, review = self._setup(session)
        with pytest.raises(ValueError, match="comment_text"):
            ReviewCommentCRUD.create(session, review.id, user.id, "")

    def test_create_whitespace_text_raises(self, session):
        user, _, review = self._setup(session)
        with pytest.raises(ValueError, match="comment_text"):
            ReviewCommentCRUD.create(session, review.id, user.id, "   ")

    def test_get_by_id_found(self, session):
        user, _, review = self._setup(session)
        comment = ReviewCommentCRUD.create(session, review.id, user.id, "Hello")
        assert ReviewCommentCRUD.get_by_id(session, comment.id).id == comment.id

    def test_get_by_id_not_found(self, session):
        assert ReviewCommentCRUD.get_by_id(session, 99999) is None

    def test_get_by_review(self, session):
        user, _, review = self._setup(session)
        ReviewCommentCRUD.create(session, review.id, user.id, "Comment 1")
        ReviewCommentCRUD.create(session, review.id, user.id, "Comment 2")
        results = ReviewCommentCRUD.get_by_review(session, review.id)
        assert len(results) == 2

    def test_get_by_review_limit(self, session):
        user, _, review = self._setup(session)
        for i in range(5):
            ReviewCommentCRUD.create(session, review.id, user.id, f"Comment {i}")
        assert len(ReviewCommentCRUD.get_by_review(session, review.id, limit=3)) == 3

    def test_delete_found(self, session):
        user, _, review = self._setup(session)
        comment = ReviewCommentCRUD.create(session, review.id, user.id, "Delete me")
        assert ReviewCommentCRUD.delete(session, comment.id) is True
        assert ReviewCommentCRUD.get_by_id(session, comment.id) is None

    def test_delete_not_found(self, session):
        assert ReviewCommentCRUD.delete(session, 99999) is False


# ---------------------------------------------------------------------------
# ReviewLikeCRUD
# ---------------------------------------------------------------------------


class TestReviewLikeCRUD:
    def _setup(self, session):
        user = make_user(session)
        book = make_book(session)
        review = make_review(session, user, book)
        return user, review

    def test_add_like(self, session):
        user, review = self._setup(session)
        like = ReviewLikeCRUD.add(session, review.id, user.id)
        assert like.review_id == review.id
        assert like.user_id == user.id

    def test_add_like_idempotent(self, session):
        user, review = self._setup(session)
        l1 = ReviewLikeCRUD.add(session, review.id, user.id)
        l2 = ReviewLikeCRUD.add(session, review.id, user.id)
        assert l1.review_id == l2.review_id

    def test_get_like_found(self, session):
        user, review = self._setup(session)
        ReviewLikeCRUD.add(session, review.id, user.id)
        like = ReviewLikeCRUD.get(session, review.id, user.id)
        assert like is not None

    def test_get_like_not_found(self, session):
        assert ReviewLikeCRUD.get(session, 99999, 99999) is None

    def test_get_by_review(self, session):
        user, review = self._setup(session)
        u2 = make_user(session, email="b@x.com", username="u2")
        ReviewLikeCRUD.add(session, review.id, user.id)
        ReviewLikeCRUD.add(session, review.id, u2.id)
        assert len(ReviewLikeCRUD.get_by_review(session, review.id)) == 2

    def test_remove_like(self, session):
        user, review = self._setup(session)
        ReviewLikeCRUD.add(session, review.id, user.id)
        assert ReviewLikeCRUD.remove(session, review.id, user.id) is True
        assert ReviewLikeCRUD.get(session, review.id, user.id) is None

    def test_remove_not_found(self, session):
        assert ReviewLikeCRUD.remove(session, 99999, 99999) is False
