import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = f"postgresql+psycopg://{os.environ['DATABASE_USER']}:{os.environ['DATABASE_PW']}@localhost:5433/{os.environ['DATABASE_NAME']}"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)