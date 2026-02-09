from sqlalchemy import select

from bookdb.db.session import SessionLocal
from bookdb.db.models import User

def seed():
    with SessionLocal() as db:
        stmt = select(User).where(User.email == "admin@example.com")
        if not db.scalar(stmt):
            db.add(User(email="admin@example.com", name="Admin"))
            db.commit()

if __name__ == "__main__":
    seed()