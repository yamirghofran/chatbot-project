from datetime import datetime, timedelta, timezone

import jwt

from .config import settings

ALGORITHM = "HS256"


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """Decode and validate a JWT token. Returns the user_id (int). Raises jwt.PyJWTError on failure."""
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    return int(payload["sub"])
