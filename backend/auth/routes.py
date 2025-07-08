from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from backend.db.connection import SessionLocal
from ..db import models
from .schemas import *
from .utils import hash_password, verify_password
from .jwt_ import create_access_token

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register-org", response_model=TokenResponse)
def register_org(data: RegisterOrgRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter_by(email=data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create tenant
    tenant_id = uuid4()
    join_code = data.join_code or f"{data.tenant_name.lower().replace(' ', '')}-{str(tenant_id)[:6]}"
    join_code_hash = hash_password(join_code)

    tenant = models.Tenant(
        id=tenant_id,
        name=data.tenant_name,
        join_code_hash=join_code_hash
    )
    db.add(tenant)
    db.flush()

    # Create user
    user = models.User(
        id=uuid4(),
        tenant_id=tenant.id,
        fname=data.fname,
        lname=data.lname,
        email=data.email,
        password_hash=hash_password(data.password),
        is_admin=True
    )
    db.add(user)
    db.commit()

    token = create_access_token({
        "user_id": str(user.id),
        "tenant_id": str(user.tenant_id),
        "is_admin": True
    })
    return TokenResponse(access_token=token)

@router.post("/register-user", response_model=TokenResponse)
def register_user(data: RegisterUserRequest, db: Session = Depends(get_db)):
    tenant = db.query(models.Tenant).filter_by(id=data.tenant_id).first()
    if not tenant or not verify_password(data.join_code, tenant.join_code_hash):
        raise HTTPException(status_code=403, detail="Invalid join code")

    if db.query(models.User).filter_by(email=data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = models.User(
        id=uuid4(),
        tenant_id=tenant.id,
        fname=data.fname,
        lname=data.lname,
        email=data.email,
        password_hash=hash_password(data.password)
    )
    db.add(user)
    db.commit()

    token = create_access_token({
        "user_id": str(user.id),
        "tenant_id": str(user.tenant_id),
        "is_admin": False
    })
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({
        "user_id": str(user.id),
        "tenant_id": str(user.tenant_id),
        "is_admin": user.is_admin
    })
    return TokenResponse(access_token=token)
