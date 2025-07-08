import os
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from .connection import engine

def db_has_tables():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public';
        """))
        count = result.scalar()
        return count > 0

def run_sql_file(filepath: str):
    with engine.connect() as conn:
        with open(filepath, "r") as f:
            sql = f.read()
        conn.execute(text(sql))
        print("✅ schema.sql executed")

if __name__ == "__main__":
    if db_has_tables():
        print("✅ Database already initialized. No action taken.")
    else:
        print("⏳ No tables found. Initializing database...")
        run_sql_file(os.path.join(os.path.dirname(__file__), "schema.sql"))
