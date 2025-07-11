from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
import backend.db.models  # makes sure models are registered
from backend.db.setup import db_has_tables, run_sql_file  # 👈 add this
import os
from backend.auth.routes import router as auth_router
from backend.documents.routes import router as documents_router  # 👈 add this
app = FastAPI()
app.include_router(auth_router, prefix="/auth")
app.include_router(documents_router, prefix="/documents")  # Add this

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

   