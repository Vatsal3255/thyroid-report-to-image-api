# backend/app/models.py
from sqlalchemy import Column, Integer, String, LargeBinary, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(200), nullable=False)
    password_hash = Column(LargeBinary(128), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PatientReport(Base):
    __tablename__ = "patient_reports"

    report_id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"))
    raw_report = Column(Text)
    json_report = Column(JSON)
    image_filename = Column(String(255))
    image_content_type = Column(String(50))
    image_blob = Column(LargeBinary)
    thyroid_report_date = Column(DateTime(timezone=True), nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at         = Column(DateTime(timezone=True), nullable=True)
