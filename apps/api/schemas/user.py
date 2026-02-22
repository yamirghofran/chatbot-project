from pydantic import BaseModel


class UserOut(BaseModel):
    id: str
    handle: str
    displayName: str
    avatarUrl: str | None = None
    followingCount: int = 0
    followersCount: int = 0

    class Config:
        from_attributes = True


class UserProfileOut(UserOut):
    pass
