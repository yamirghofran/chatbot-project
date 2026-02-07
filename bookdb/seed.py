from bookdb.db.session import SessionLocal
from bookdb.db.models import User

def seed():
    db = SessionLocal()

    if not db.query(User).filter_by(email="admin@example.com").first():
        db.add(User(email="admin@example.com", name="Admin"))

    db.commit()
    db.close()

if __name__ == "__main__":
    seed()