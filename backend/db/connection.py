from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DB_URL")
print("DATABASE_URL =", repr(DATABASE_URL))

# Add connection pooling and optimization with correct psycopg2 parameters
engine = create_engine(
    DATABASE_URL,
    pool_size=20,          # Number of connections to maintain in pool
    max_overflow=30,       # Additional connections beyond pool_size
    pool_pre_ping=True,    # Validate connections before use
    pool_recycle=3600,     # Recycle connections every hour
    connect_args={
        "connect_timeout": 30,
        "options": "-c statement_timeout=60000 -c jit=off",  # 60 second query timeout, disable JIT
        "application_name": "voicebot_app"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
