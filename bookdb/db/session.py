import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PW = os.getenv('DATABASE_PW')
DATABASE_NAME = os.getenv('DATABASE_NAME')

if not all([DATABASE_USER, DATABASE_PW, DATABASE_NAME]):
    raise ValueError("DATABASE_USER, DATABASE_PW, and DATABASE_NAME must be set in environment variables")

DATABASE_URL = f"postgresql+psycopg://{DATABASE_USER}:{DATABASE_PW}@localhost:5433/{DATABASE_NAME}"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)