from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from backend.db.connection import SessionLocal
import backend.db.models  # makes sure models are registered

app = FastAPI()

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
