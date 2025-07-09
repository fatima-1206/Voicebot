import os
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from .connection import engine
import dotenv
db_name = os.getenv("DB_NAME")
def db_has_tables():
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """))
        count = result.scalar()
        return count > 0

def run_sql_file(filepath: str):
    raw_conn = engine.raw_connection()
    try:
        with open(filepath, "r") as f:
            sql = f.read()
        cursor = raw_conn.cursor()
        cursor.execute(sql)
        raw_conn.commit()
        print("✅ schema.sql executed")
    except Exception as e:
        print("❌ Failed to initialize DB:", e)
        import traceback
        traceback.print_exc()
    finally:
        raw_conn.close()

if __name__ == "__main__":
    if db_has_tables():
        print("✅ Postgresql Database already initialized. No action taken.")
    else:
        print("⏳ No tables found. Initializing database...")
        run_sql_file(os.path.join(os.path.dirname(__file__), "schema.sql"))
