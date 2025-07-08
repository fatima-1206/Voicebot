from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
import backend.db.models  # makes sure models are registered
from backend.db.setup import db_has_tables, run_sql_file  # 👈 add this
import os
from backend.auth.routes import router as auth_router
from backend.rag.retrieverconstruction import construct_retrievers

app = FastAPI()
app.include_router(auth_router, prefix="/auth")
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"msg": "SaaS voicebot backend live"}

# 👇 Add this startup hook
@app.on_event("startup")
def init_db():
    if db_has_tables():
        print("✅ Database already initialized. No action taken.")
    else:
        print("⏳ No tables found. Initializing database...")
        run_sql_file(os.path.join(os.path.dirname(__file__), "db/schema.sql"))

    try:
        construct_retrievers()  # Ensure retrievers are constructed on startup
    except Exception as e:
        print(f"🚫 Failed to construct retrievers: {e}")
