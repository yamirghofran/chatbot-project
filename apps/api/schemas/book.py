from pydantic import BaseModel


class BookOut(BaseModel):
    id: str
    title: str
    author: str
    coverUrl: str | None
    description: str | None
    tags: list[str]

    class Config:
        from_attributes = True


class BookStatsOut(BaseModel):
    averageRating: float | None
    ratingCount: int
    shellCount: int


class BookDetail(BookOut):
    stats: BookStatsOut | None = None
    publicationYear: int | None = None
    isbn13: str | None = None
