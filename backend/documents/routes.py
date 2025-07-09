from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from backend.db.connection import SessionLocal
from backend.db import models
from backend.auth.jwt_ import get_current_user, UserTokenData
from backend.workers.tasks import process_uploaded_document

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    # Validate file type
    if not file.filename.endswith((".pdf", ".txt", ".docx", ".epub")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc_id = uuid4()
    contents = await file.read()
    filename = file.filename

    # Save metadata to DB
    document = models.Document(
        id=doc_id,
        tenant_id=current_user.tenant_id,
        filename=filename,
        upload_time=datetime.utcnow()
    )
    db.add(document)
    db.commit()

    # Send to Celery
    process_uploaded_file.delay(
        file_bytes=contents.decode("latin1"),  # Store as string
        filename=filename,
        doc_id=str(doc_id),
        tenant_id=str(current_user.tenant_id)
    )

    return {"status": "upload successful", "doc_id": str(doc_id)}


@router.get("/documents")
def list_documents(
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    docs = db.query(models.Document).filter_by(tenant_id=current_user.tenant_id).all()
    return [{"id": str(d.id), "filename": d.filename, "uploaded": d.upload_time} for d in docs]