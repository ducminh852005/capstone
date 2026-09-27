from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# We can replace the default URL with a real Postgres URL via .env
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./caulongstats.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    # check_same_thread=False is needed only for SQLite
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
