from pydantic import BaseModel
from .user import UserOut


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
