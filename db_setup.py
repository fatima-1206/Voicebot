import psycopg2
import os
from dotenv import load_dotenv

# Get full URL from env
load_dotenv()
DB_URL = os.getenv("DB_URL")

def run_sql_file(file_path):
    with open(file_path, 'r') as f:
        sql = f.read()

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()
    conn.close()
    print("✅ SQL executed successfully.")

if __name__ == "__main__":
    run_sql_file("schema.sql")
