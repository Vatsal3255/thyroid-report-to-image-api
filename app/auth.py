# backend/app/auth.py
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from jose import jwt
from passlib.hash import bcrypt

from . import schemas, models, database, config
from .deps import get_current_doctor

router = APIRouter(prefix="/auth", tags=["auth"])

def create_access_token(sub: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=config.JWT_EXPIRE_HOURS)
    payload = {"sub": sub, "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)

@router.post("/signup", response_model=schemas.DoctorOut)
def signup(payload: schemas.DoctorSignup, db: Session = Depends(database.get_db)):
    existing = db.query(models.Doctor).filter(models.Doctor.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = bcrypt.hash(payload.password)
    doc = models.Doctor(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=hashed.encode(),
        is_active=True,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

@router.post("/login", response_model=schemas.Token)
def login(payload: schemas.DoctorLogin, db: Session = Depends(database.get_db)):
    doc = db.query(models.Doctor).filter(models.Doctor.email == payload.email).first()
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # stored as bytes -> decode to str for passlib verify
    stored = doc.password_hash.decode()
    if not bcrypt.verify(payload.password, stored):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(doc.doctor_id))
    doc.last_login_at = datetime.utcnow()
    db.add(doc); db.commit()
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.DoctorOut)
def read_current_doctor(current: models.Doctor = Depends(get_current_doctor)):
    return current
