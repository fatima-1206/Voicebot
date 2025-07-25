from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from backend.db.connection import SessionLocal
from ..db import models
from .schemas import *
from .utils import hash_password, verify_password
from .jwt_ import create_access_token
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from .jwt_ import get_current_user  # Assuming this returns the user dict from JWT

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


@router.post("/register-solo-user", response_model=TokenResponse)
def register_solo_user(data: RegisterSoloUserRequest, db: Session = Depends(get_db)):  # Fixed function name
    if db.query(models.User).filter_by(email=data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create tenant
    tenant_id = uuid4()
    
    # Use provided tenant_name or create one from user's name
    tenant_name = data.tenant_name or f"{data.fname}'s Workspace"
    join_code = data.join_code or f"{data.fname.lower().replace(' ', '')}-{str(tenant_id)[:6]}"
    join_code_hash = hash_password(join_code)

    tenant = models.Tenant(
        id=tenant_id,
        name=tenant_name,  # Fixed: use tenant_name variable
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


def is_admin(user: dict = Depends(get_current_user)):
    if not user.is_admin:
        return False
    return True

# Usage in a route
@router.get("/is_admin")
def admin_route(is_admin: bool = Depends(is_admin)):
    return {"is_admin": is_admin}

# ...existing code...

@router.get("/admin/join-code")
def get_join_code(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    tenant = db.query(models.Tenant).filter_by(id=user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"join_code_hash": tenant.join_code_hash}

def get_join_code(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    tenant = db.query(models.Tenant).filter_by(id=user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"tenant_id": tenant.id}




