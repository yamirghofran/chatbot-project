from typing import Literal
from pydantic import BaseModel
from .user import UserOut
from .book import BookOut


class ActivityItemOut(BaseModel):
    id: str
    user: UserOut
    type: Literal["rating", "shell_add", "list_add"]
    book: BookOut
    rating: int | None = None
    listName: str | None = None
    timestamp: str
