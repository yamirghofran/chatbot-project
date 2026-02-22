from pydantic import BaseModel
from .user import UserOut


class CreateReviewRequest(BaseModel):
    text: str


class CreateCommentRequest(BaseModel):
    text: str


class ReplyOut(BaseModel):
    id: str
    user: UserOut
    text: str
    likes: int = 0
    timestamp: str


class ReviewOut(BaseModel):
    id: str
    user: UserOut
    text: str
    likes: int
    isLikedByMe: bool = False
    timestamp: str
    replies: list[ReplyOut] = []
