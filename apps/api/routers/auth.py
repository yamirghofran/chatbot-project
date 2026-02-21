import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bookdb.db.crud import UserCRUD
from bookdb.db.models import User

from ..core.auth import create_token
from ..core.deps import get_current_user, get_db
from ..core.serialize import serialize_user
from ..schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        password_hash = _hash_password(body.password)
        user = UserCRUD.create(
            db,
            email=body.email,
            name=body.name,
            username=body.username,
            password_hash=password_hash,
        )
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    token = create_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = UserCRUD.get_by_email(db, body.email)
    if user is None or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(user.id)
    return TokenResponse(access_token=token)


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return serialize_user(current_user)
