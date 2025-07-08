from pydantic import BaseModel, EmailStr
from uuid import UUID
from typing import Optional

class RegisterOrgRequest(BaseModel):
    tenant_name: str
    join_code: Optional[str] = None  # optional if auto-generated
    fname: str
    lname: str
    email: EmailStr
    password: str

class RegisterUserRequest(BaseModel):
    tenant_id: UUID
    join_code: str
    fname: str
    lname: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
