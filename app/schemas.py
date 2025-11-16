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

# Patient / Report
class PatientCreate(BaseModel):
    patient_id: int
    full_name: str

class PatientOut(BaseModel):
    patient_id: int
    full_name: str
    class Config:
        from_attributes = True

class ReportOut(BaseModel):
    report_id: int
    generated_at: datetime
    class Config:
        from_attributes = True

class ReportDetail(BaseModel):
    report_id: int
    patient_id: int
    patient_name: str
    raw_report: str
    json_report: Optional[Dict[str, Any]]
    generated_at: datetime
    has_image: bool
    image_content_type: Optional[str]
    class Config:
        from_attributes = True
