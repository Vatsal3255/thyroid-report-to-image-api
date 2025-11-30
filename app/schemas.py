# backend/app/schemas.py
from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, Any, Dict

# Doctor auth
class DoctorSignup(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class DoctorLogin(BaseModel):
    email: EmailStr
    password: str

class DoctorOut(BaseModel):
    doctor_id: int
    email: EmailStr
    full_name: str
    class Config:
        from_attributes = True

# Token
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Reports
class ReportOut(BaseModel):
    report_id: int
    generated_at: datetime
    ultrasound_date: Optional[datetime]
    class Config:
        from_attributes = True

class ReportDetail(BaseModel):
    report_id: int
    raw_report: str
    json_report: Optional[Dict[str, Any]]
    generated_at: datetime
    thyroid_report_date: Optional[datetime]
    has_image: bool
    image_content_type: Optional[str]
    class Config:
        from_attributes = True
