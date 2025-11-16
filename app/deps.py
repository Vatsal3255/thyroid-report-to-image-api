# backend/app/deps.py
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from . import config, database, models

bearer = HTTPBearer(auto_error=False)

def get_current_doctor(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(database.get_db)
) -> models.Doctor:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = creds.credentials
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        doctor_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token")
    doc = db.query(models.Doctor).get(doctor_id)
    if not doc or not doc.is_active:
        raise HTTPException(status_code=401, detail="Invalid user")
    return doc
