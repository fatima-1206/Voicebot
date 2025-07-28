from pydantic import BaseModel, EmailStr
from uuid import UUID
from typing import Optional
from datetime import date

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
    DoB: Optional[date] = None  # Optional field for date of birth
    gender: Optional[str] = None  # Optional
    display_name: Optional[str] = None  # Optional field for display name
class RegisterSoloUserRequest(BaseModel):
    fname: str
    lname: str
    email: EmailStr
    password: str
    tenant_name: Optional[str] = None  # Add this field
    join_code: Optional[str] = None    # Add this field
    DoB: Optional[date] = None  # Optional field for date of birth
    gender: Optional[str] = None  # Optional
    display_name: Optional[str] = None  # Optional field for display name
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"