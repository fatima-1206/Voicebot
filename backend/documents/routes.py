from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from backend.db.connection import SessionLocal
from backend.db import models
from backend.auth.jwt_ import get_current_user, UserTokenData
from backend.workers.tasks import process_uploaded_document
import os

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")  # Fixed: removed /documents prefix
async def upload_document(
    file: UploadFile = File(...),
    is_public: bool = False,  # Add optional parameter for public documents
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    # Validate file type
    # print stuff about the file:
    print(f"Received file: {file.filename}, size: {file.size} bytes, content type: {file.content_type}")
    if not file.filename.endswith((".pdf", ".txt", ".docx", ".epub")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc_id = uuid4()
    contents = await file.read()
    filename = file.filename
    
    # Get file extension and size
    extension = os.path.splitext(filename)[-1].lower()
    file_size = len(contents)

    # Save metadata to DB with all required fields
    document = models.Document(
        id=doc_id,
        tenant_id=current_user.tenant_id,
        user_id=current_user.user_id,  # Add user_id
        filename=filename,
        filetype=extension,  # Add filetype
        size=file_size,  # Add file size
        is_public=is_public,  # Add is_public flag
        status="processing",  # Set initial status
        upload_time=datetime.utcnow()
    )
    db.add(document)
    db.commit()

    # Send to Celery with correct function name and all required parameters
    process_uploaded_document.delay(  # Fixed function name
        file_bytes=contents.decode("latin1"),  # Store as string
        filename=filename,
        doc_id=str(doc_id),
        tenant_id=str(current_user.tenant_id),
        user_id=str(current_user.user_id),  # Add user_id
        is_public=is_public  # Add is_public
    )

    return {
        "status": "upload successful", 
        "doc_id": str(doc_id),
        "filename": filename,
        "message": "Document uploaded and processing started"
    }

@router.get("/")  # Fixed: removed /documents prefix
def list_documents(
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    # Get documents for the current tenant
    docs = db.query(models.Document).filter(
        models.Document.tenant_id == current_user.tenant_id
    ).order_by(models.Document.upload_time.desc()).all()
    
    return [
        {
            "id": str(d.id), 
            "filename": d.filename, 
            "uploaded": d.upload_time,
            "status": d.status,
            "size": d.size,
            "filetype": d.filetype,
            "is_public": d.is_public
        } 
        for d in docs
    ]

@router.get("/{doc_id}")
def get_document_details(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Get details of a specific document"""
    try:
        doc_uuid = uuid4() if doc_id == "new" else doc_id
    except:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    document = db.query(models.Document).filter(
        models.Document.id == doc_uuid,
        models.Document.tenant_id == current_user.tenant_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get chunks for this document
    chunks = db.query(models.Chunk).filter(
        models.Chunk.doc_id == document.id
    ).all()
    
    return {
        "id": str(document.id),
        "filename": document.filename,
        "uploaded": document.upload_time,
        "status": document.status,
        "size": document.size,
        "filetype": document.filetype,
        "is_public": document.is_public,
        "chunks_count": len(chunks),
        "chunks": [
            {
                "id": str(chunk.id),
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            for chunk in chunks[:5]  # Show first 5 chunks as preview
        ]
    }

@router.delete("/{doc_id}")
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """Delete a document and its chunks"""
    try:
        doc_uuid = doc_id
    except:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    document = db.query(models.Document).filter(
        models.Document.id == doc_uuid,
        models.Document.tenant_id == current_user.tenant_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete chunks first (due to foreign key constraint)
    db.query(models.Chunk).filter(models.Chunk.doc_id == document.id).delete()
    
    # Delete document
    db.delete(document)
    db.commit()
    
    return {"message": f"Document '{document.filename}' deleted successfully"}