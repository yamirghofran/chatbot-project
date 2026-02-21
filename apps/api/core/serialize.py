"""Helpers to serialize SQLAlchemy ORM models to API schema dicts."""

from __future__ import annotations

from datetime import datetime, timezone

from bookdb.db.models import Book, BookList, Review, User


def _is_goodreads_nophoto_url(url: str | None) -> bool:
    if not url:
        return False
    return "gr-assets.com/assets/nophoto" in url.lower()


def _openlibrary_cover_from_isbn(isbn13: str | None) -> str | None:
    if not isbn13:
        return None
    normalized = "".join(ch for ch in isbn13 if ch.isdigit() or ch.upper() == "X")
    if not normalized:
        return None
    return f"https://covers.openlibrary.org/b/isbn/{normalized}-L.jpg"


def resolve_cover_url(image_url: str | None, isbn13: str | None) -> str | None:
    if image_url and not _is_goodreads_nophoto_url(image_url):
        return image_url
    return _openlibrary_cover_from_isbn(isbn13) or image_url


def relative_time(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        m = max(1, seconds // 60)
        return f"{m}m ago"
    if seconds < 86400:
        h = seconds // 3600
        return f"{h}h ago"
    if seconds < 604800:
        d = seconds // 86400
        return f"{d}d ago"
    w = seconds // 604800
    return f"{w}w ago"


def serialize_user(user: User) -> dict:
    return {
        "id": str(user.id),
        "handle": user.username,
        "displayName": user.name,
        "avatarUrl": None,
        "followingCount": 0,
        "followersCount": 0,
    }


def serialize_book(book: Book, engagement: dict | None = None) -> dict:
    author_names = [ba.author.name for ba in book.authors if ba.author]
    tag_names = [bt.tag.name for bt in book.tags if bt.tag]
    data = {
        "id": str(book.id),
        "title": book.title,
        "author": ", ".join(author_names),
        "coverUrl": resolve_cover_url(book.image_url, book.isbn13),
        "description": book.description,
        "tags": tag_names,
        "publicationYear": book.publication_year,
        "isbn13": book.isbn13,
    }
    if engagement is not None:
        data.update({
            "averageRating": engagement.get("averageRating"),
            "ratingCount": int(engagement.get("ratingCount", 0) or 0),
            "commentCount": int(engagement.get("commentCount", 0) or 0),
            "shellCount": int(engagement.get("shellCount", 0) or 0),
        })
    return data


def serialize_list(book_list: BookList) -> dict:
    return {
        "id": str(book_list.id),
        "name": book_list.title,
        "description": book_list.description,
        "owner": serialize_user(book_list.user),
        "books": [serialize_book(lb.book) for lb in book_list.books if lb.book],
    }


def serialize_review(
    review: Review,
    current_user_id: int | None = None,
    *,
    likes_count: int = 0,
    is_liked: bool = False,
) -> dict:
    comments = [
        {
            "id": str(c.id),
            "user": serialize_user(c.user),
            "text": c.comment_text,
            "likes": 0,
            "isLikedByMe": False,
            "timestamp": relative_time(c.created_at),
        }
        for c in review.comments
    ]
    return {
        "id": str(review.id),
        "user": serialize_user(review.user),
        "text": review.review_text,
        "likes": likes_count,
        "isLikedByMe": is_liked,
        "timestamp": relative_time(review.created_at),
        "replies": comments,
    }
