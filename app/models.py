# backend/app/models.py
from sqlalchemy import BigInteger, Column, Integer, String, LargeBinary, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(200), nullable=False)
    password_hash = Column(LargeBinary(128), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, index=True, autoincrement=False)
    doctor_id   = Column(BigInteger, ForeignKey("doctors.doctor_id"), nullable=False)
    full_name = Column(String(200), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PatientReport(Base):
    __tablename__ = "patient_reports"

    report_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"))
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"))
    raw_report = Column(Text)
    json_report = Column(JSON)
    image_filename = Column(String(255))
    image_content_type = Column(String(50))
    image_blob = Column(LargeBinary)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at         = Column(DateTime(timezone=True), nullable=True)
