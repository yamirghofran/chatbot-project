from pydantic import BaseModel
from .book import BookOut
from .user import UserOut


class ListOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    owner: UserOut
    books: list[BookOut] = []

    class Config:
        from_attributes = True


class ListDetail(ListOut):
    pass


class CreateListRequest(BaseModel):
    name: str
    description: str | None = None


class UpdateListRequest(BaseModel):
    name: str | None = None
    description: str | None = None
