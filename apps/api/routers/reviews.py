from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import ReviewCRUD, ReviewCommentCRUD, ReviewLikeCRUD
from bookdb.db.models import Review, ReviewComment, User

from ..core.deps import get_current_user, get_db
from ..core.serialize import relative_time, serialize_user
from ..schemas.review import CreateCommentRequest

router = APIRouter(prefix="/reviews", tags=["reviews"])


def _get_review_or_404(db: Session, review_id: int) -> Review:
    review = ReviewCRUD.get_by_id(db, review_id)
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    return review


@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = _get_review_or_404(db, review_id)
    if review.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your review")
    ReviewCRUD.delete(db, review_id)
    db.commit()


@router.post("/{review_id}/likes", status_code=status.HTTP_204_NO_CONTENT)
def like_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_review_or_404(db, review_id)
    ReviewLikeCRUD.add(db, review_id, current_user.id)
    db.commit()


@router.delete("/{review_id}/likes", status_code=status.HTTP_204_NO_CONTENT)
def unlike_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_review_or_404(db, review_id)
    ReviewLikeCRUD.remove(db, review_id, current_user.id)
    db.commit()


@router.post("/{review_id}/comments")
def add_comment(
    review_id: int,
    body: CreateCommentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_review_or_404(db, review_id)
    comment = ReviewCommentCRUD.create(db, review_id, current_user.id, body.text)
    db.commit()
    comment = db.scalar(
        select(ReviewComment)
        .where(ReviewComment.id == comment.id)
        .options(selectinload(ReviewComment.user))
    )
    return {
        "id": str(comment.id),
        "user": serialize_user(comment.user),
        "text": comment.comment_text,
        "likes": 0,
        "isLikedByMe": False,
        "timestamp": relative_time(comment.created_at),
    }


@router.delete("/{review_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(
    review_id: int,
    comment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    comment = ReviewCommentCRUD.get_by_id(db, comment_id)
    if comment is None or comment.review_id != review_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")
    if comment.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your comment")
    ReviewCommentCRUD.delete(db, comment_id)
    db.commit()
