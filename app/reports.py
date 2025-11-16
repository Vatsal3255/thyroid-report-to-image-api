# backend/app/reports.py
import json
from io import BytesIO
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import database, models, schemas
from .deps import get_current_doctor

router = APIRouter(prefix="/reports", tags=["reports"])

def _owned_report(db: Session, report_id: int, doctor_id: int) -> models.PatientReport:
    return (
        db.query(models.PatientReport)
        .filter(
            models.PatientReport.report_id == report_id,
            models.PatientReport.doctor_id == doctor_id,
            models.PatientReport.deleted_at.is_(None),
        )
        .first()
    )

@router.get("/{report_id}", response_model=schemas.ReportDetail)
def get_report(
    report_id: int,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    r = _owned_report(db, report_id, current.doctor_id)
    if not r:
        raise HTTPException(status_code=404, detail="Report not found")

    patient = db.query(models.Patient).get(r.patient_id)
    patient_name = patient.full_name if patient else ""

    return {
        "report_id": r.report_id,
        "patient_id": r.patient_id,
        "patient_name": patient_name,
        "raw_report": r.raw_report or "",
        "json_report": r.json_report or {},
        "generated_at": r.generated_at.isoformat() if r.generated_at else "",
        "has_image": bool(r.image_blob),
        "image_content_type": r.image_content_type,
    }

@router.get("/{report_id}/image")
def get_report_image(
    report_id: int,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    r = _owned_report(db, report_id, current.doctor_id)
    if not r or not r.image_blob:
        raise HTTPException(status_code=404, detail="Image not found")

    return StreamingResponse(
        BytesIO(r.image_blob),
        media_type=r.image_content_type or "image/png",
        headers={"Content-Disposition": f'inline; filename="{r.image_filename or "report.png"}"'},
    )

@router.put("/{report_id}")
async def update_report(
    report_id: int,
    raw_report: str = Form(...),
    json_report: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    r = _owned_report(db, report_id, current.doctor_id)
    if not r:
        raise HTTPException(status_code=404, detail="Report not found")

    raw = (raw_report or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="raw_report is required")
    r.raw_report = raw

    if json_report:
        try:
            r.json_report = json.loads(json_report)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="json_report must be valid JSON")

    if image and image.filename:
        blob = await image.read()
        if len(blob) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        r.image_blob = blob
        r.image_filename = image.filename
        r.image_content_type = image.content_type or "image/png"

    db.add(r)
    db.commit()
    db.refresh(r)
    return {"report_id": r.report_id}
