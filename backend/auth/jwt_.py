# import jwt
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
import dotenv
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel

dotenv.load_dotenv()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60*24))  # Default to 1 day

class UserTokenData(BaseModel):
    user_id: str
    tenant_id: str
    is_admin: bool

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserTokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return UserTokenData(
            user_id=payload["user_id"],
            tenant_id=payload["tenant_id"],
            is_admin=payload["is_admin"]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")