from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import uuid4, UUID
from datetime import datetime
from backend.db.connection import SessionLocal
from backend.db import models
from backend.auth.jwt_ import get_current_user, UserTokenData
from backend.workers.tasks import scrape_link
from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

# Pydantic schemas
class LinkCreate(BaseModel):
    url: HttpUrl
    name: Optional[str] = None
    description: Optional[str] = None
    depth: Optional[int] = 1
    excluded_sublinks: Optional[str] = None

class LinkUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    depth: Optional[int] = None
    excluded_sublinks: Optional[str] = None
    status: Optional[str] = None

class LinkResponse(BaseModel):
    id: str
    url: str
    name: Optional[str]
    description: Optional[str]
    depth: int
    excluded_sublinks: Optional[str]
    status: str
    last_scraped: Optional[datetime]
    created_at: datetime
    updated_at: datetime

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# @router.post("/", response_model=LinkResponse)
# async def create_link(
#     link_data: LinkCreate,
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Create a new scraping link"""
    
#     # Check if URL already exists for this tenant
#     existing_link = db.query(models.Link).filter(
#         models.Link.tenant_id == current_user.tenant_id,
#         models.Link.url == str(link_data.url)
#     ).first()
    
#     if existing_link:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"URL '{link_data.url}' already exists for this tenant"
#         )
    
#     # Create new link
#     new_link = models.Link(
#         id=uuid4(),
#         tenant_id=current_user.tenant_id,
#         user_id=current_user.user_id,
#         url=str(link_data.url),
#         name=link_data.name,
#         description=link_data.description,
#         depth=link_data.depth,
#         excluded_sublinks=link_data.excluded_sublinks,
#         status="active"
#     )
    
#     db.add(new_link)
#     db.commit()
#     db.refresh(new_link)
    
#     return LinkResponse(
#         id=str(new_link.id),
#         url=new_link.url,
#         name=new_link.name,
#         description=new_link.description,
#         depth=new_link.depth,
#         excluded_sublinks=new_link.excluded_sublinks,
#         status=new_link.status,
#         last_scraped=new_link.last_scraped,
#         created_at=new_link.created_at,
#         updated_at=new_link.updated_at
#     )

# @router.get("/", response_model=List[LinkResponse])
# def list_links(
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Get all links for the current tenant"""
#     links = db.query(models.Link).filter(
#         models.Link.tenant_id == current_user.tenant_id
#     ).order_by(models.Link.created_at.desc()).all()
    
#     return [
#         LinkResponse(
#             id=str(link.id),
#             url=link.url,
#             name=link.name,
#             description=link.description,
#             depth=link.depth,
#             excluded_sublinks=link.excluded_sublinks,
#             status=link.status,
#             last_scraped=link.last_scraped,
#             created_at=link.created_at,
#             updated_at=link.updated_at
#         )
#         for link in links
#     ]

# @router.get("/{link_id}", response_model=LinkResponse)
# def get_link(
#     link_id: str,
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Get a specific link"""
#     try:
#         link_uuid = UUID(link_id)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid link ID")
    
#     link = db.query(models.Link).filter(
#         models.Link.id == link_uuid,
#         models.Link.tenant_id == current_user.tenant_id
#     ).first()
    
#     if not link:
#         raise HTTPException(status_code=404, detail="Link not found")
    
#     return LinkResponse(
#         id=str(link.id),
#         url=link.url,
#         name=link.name,
#         description=link.description,
#         depth=link.depth,
#         excluded_sublinks=link.excluded_sublinks,
#         status=link.status,
#         last_scraped=link.last_scraped,
#         created_at=link.created_at,
#         updated_at=link.updated_at
#     )

# @router.put("/{link_id}", response_model=LinkResponse)
# def update_link(
#     link_id: str,
#     link_data: LinkUpdate,
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Update a link"""
#     try:
#         link_uuid = UUID(link_id)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid link ID")
    
#     link = db.query(models.Link).filter(
#         models.Link.id == link_uuid,
#         models.Link.tenant_id == current_user.tenant_id
#     ).first()
    
#     if not link:
#         raise HTTPException(status_code=404, detail="Link not found")
    
#     # Check permissions (only owner or admin can update)
#     if link.user_id != current_user.user_id and not current_user.is_admin:
#         raise HTTPException(status_code=403, detail="Not authorized to update this link")
    
#     # Update fields
#     update_data = link_data.dict(exclude_unset=True)
#     for field, value in update_data.items():
#         setattr(link, field, value)
    
#     link.updated_at = datetime.utcnow()
#     db.commit()
#     db.refresh(link)
    
#     return LinkResponse(
#         id=str(link.id),
#         url=link.url,
#         name=link.name,
#         description=link.description,
#         depth=link.depth,
#         excluded_sublinks=link.excluded_sublinks,
#         status=link.status,
#         last_scraped=link.last_scraped,
#         created_at=link.created_at,
#         updated_at=link.updated_at
#     )

# @router.delete("/{link_id}")
# def delete_link(
#     link_id: str,
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Delete a link"""
#     try:
#         link_uuid = UUID(link_id)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid link ID")
    
#     link = db.query(models.Link).filter(
#         models.Link.id == link_uuid,
#         models.Link.tenant_id == current_user.tenant_id
#     ).first()
    
#     if not link:
#         raise HTTPException(status_code=404, detail="Link not found")
    
#     # Check permissions (only owner or admin can delete)
#     if link.user_id != current_user.user_id and not current_user.is_admin:
#         raise HTTPException(status_code=403, detail="Not authorized to delete this link")
    
#     db.delete(link)
#     db.commit()
    
#     return {"message": f"Link '{link.name or link.url}' deleted successfully"}

# @router.post("/{link_id}/scrape")
# async def trigger_scraping(
#     link_id: str,
#     db: Session = Depends(get_db),
#     current_user: UserTokenData = Depends(get_current_user)
# ):
#     """Trigger scraping for a specific link"""
#     try:
#         link_uuid = UUID(link_id)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid link ID")
    
#     link = db.query(models.Link).filter(
#         models.Link.id == link_uuid,
#         models.Link.tenant_id == current_user.tenant_id
#     ).first()
    
#     if not link:
#         raise HTTPException(status_code=404, detail="Link not found")
    
#     if link.status == "scraping":
#         raise HTTPException(status_code=400, detail="Link is already being scraped")
    
#     if link.status == "disabled":
#         raise HTTPException(status_code=400, detail="Link is disabled")
    
#     # Update status to scraping
#     link.status = "scraping"
#     link.updated_at = datetime.utcnow()
#     db.commit()
    
#     # Trigger Celery task for scraping
#     scrape_link.delay(
#         link_id=str(link.id),
#         url=link.url,
#         depth=link.depth,
#         excluded_sublinks=link.excluded_sublinks or "",
#         tenant_id=str(current_user.tenant_id),
#         user_id=str(current_user.user_id)
#     )
    
#     return {
#         "message": f"Scraping started for '{link.name or link.url}'",
#         "link_id": str(link.id),
#         "status": "scraping"
#     }