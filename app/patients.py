# backend/app/patients.py
from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, status
from sqlalchemy.orm import Session
from typing import Optional
import json

from . import database, models, schemas
from .deps import get_current_doctor

router = APIRouter(prefix="/patients", tags=["patients"])

# ---- Get a single patient (ensures ownership)
@router.get("/{patient_id}", response_model=schemas.PatientOut)
def get_patient(
    patient_id: int,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    patient = (
        db.query(models.Patient)
        .filter(
            models.Patient.patient_id == patient_id,
            models.Patient.doctor_id == current.doctor_id,
        )
        .first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# ---- List patients for current doctor
@router.get("/", response_model=list[schemas.PatientOut])
def list_patients(
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    return (
        db.query(models.Patient)
        .filter(models.Patient.doctor_id == current.doctor_id)
        .order_by(models.Patient.full_name.asc())
        .all()
    )

# ---- Create patient owned by current doctor
@router.post("/", response_model=schemas.PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: schemas.PatientCreate,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    name = payload.full_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="full_name is required")

    if payload.patient_id <= 0:
        raise HTTPException(status_code=400, detail="patient_id must be positive")

    existing = (
        db.query(models.Patient)
        .filter(models.Patient.patient_id == payload.patient_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="patient_id already exists")

    p = models.Patient(
        patient_id=payload.patient_id,
        full_name=name,
        doctor_id=current.doctor_id,
    )  # ← ownership
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

# ---- List reports for a patient (must be owned by current doctor)
@router.get("/{patient_id}/reports", response_model=list[schemas.ReportOut])
def list_reports(
    patient_id: int,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    patient = (
        db.query(models.Patient)
        .filter(models.Patient.patient_id == patient_id, models.Patient.doctor_id == current.doctor_id)
        .first()
    )
    if not patient:
        # either not found or not owned by this doctor
        raise HTTPException(status_code=404, detail="Patient not found")

    rows = (
        db.query(models.PatientReport)
        .filter(
            models.PatientReport.patient_id == patient_id,
            models.PatientReport.deleted_at.is_(None),
        )
        .order_by(models.PatientReport.generated_at.desc())
        .all()
    )
    return rows

# ---- Create a report for current doctor's patient
@router.post("/{patient_id}/reports", status_code=status.HTTP_201_CREATED)
async def create_report_for_patient(
    patient_id: int,
    raw_report: str = Form(...),
    json_report: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    # 1) Validate patient ownership
    patient = (
        db.query(models.Patient)
        .filter(models.Patient.patient_id == patient_id, models.Patient.doctor_id == current.doctor_id)
        .first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    raw = (raw_report or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="raw_report is required")

    # 2) Parse JSON if provided
    json_payload = None
    if json_report:
        try:
            json_payload = json.loads(json_report)
        except Exception:
            raise HTTPException(status_code=400, detail="json_report must be valid JSON")

    # 3) Optional image
    image_bytes = None
    image_ct = None
    image_name = None
    if image and image.filename:
        image_bytes = await image.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        image_ct = image.content_type
        image_name = image.filename

    # 4) Insert report; attribute to current doctor
    report = models.PatientReport(
        patient_id=patient_id,
        doctor_id=current.doctor_id,     # ← provenance
        raw_report=raw,
        json_report=json_payload,
        image_filename=image_name,
        image_content_type=image_ct,
        image_blob=image_bytes,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "report_id": report.report_id,
        "generated_at": report.generated_at.isoformat() if report.generated_at else "",
    }
