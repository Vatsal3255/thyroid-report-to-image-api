# backend/app/reports.py
import json
from io import BytesIO
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import database, models, schemas
from .deps import get_current_doctor

router = APIRouter(prefix="/reports", tags=["reports"])

def _parse_ultrasound_date(payload: Optional[dict]) -> Optional[datetime]:
    """Extract and normalize the ultrasound date from the parsed JSON payload."""
    if not isinstance(payload, dict):
        return None
    raw_date = payload.get("ultrasound_date")
    if not raw_date or (isinstance(raw_date, str) and raw_date.strip().lower() == "not mentioned"):
        return None
    raw = str(raw_date).strip()
    # Try strict ISO first
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    # Try common day-first formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y", "%d %b, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None

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

@router.get("/", response_model=list[schemas.ReportOut])
def list_reports(
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    rows = (
        db.query(models.PatientReport)
        .filter(
            models.PatientReport.doctor_id == current.doctor_id,
            models.PatientReport.deleted_at.is_(None),
        )
        .order_by(models.PatientReport.generated_at.desc())
        .all()
    )
    return [
        {
            "report_id": r.report_id,
            "generated_at": r.generated_at,
            "ultrasound_date": r.thyroid_report_date,
        }
        for r in rows
    ]

@router.post("/", status_code=201)
async def create_report(
    raw_report: str = Form(...),
    json_report: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    raw = (raw_report or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="raw_report is required")

    json_payload = None
    if json_report:
        try:
            json_payload = json.loads(json_report)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="json_report must be valid JSON")
    report_date = _parse_ultrasound_date(json_payload)

    image_bytes = None
    image_ct = None
    image_name = None
    if image and image.filename:
        image_bytes = await image.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        image_ct = image.content_type
        image_name = image.filename

    report = models.PatientReport(
        doctor_id=current.doctor_id,
        raw_report=raw,
        json_report=json_payload,
        thyroid_report_date=report_date,
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

@router.get("/{report_id}", response_model=schemas.ReportDetail)
def get_report(
    report_id: int,
    db: Session = Depends(database.get_db),
    current = Depends(get_current_doctor),
):
    r = _owned_report(db, report_id, current.doctor_id)
    if not r:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "report_id": r.report_id,
        "raw_report": r.raw_report or "",
        "json_report": r.json_report or {},
        "generated_at": r.generated_at.isoformat() if r.generated_at else "",
        "thyroid_report_date": r.thyroid_report_date.isoformat() if r.thyroid_report_date else None,
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
        r.thyroid_report_date = _parse_ultrasound_date(r.json_report)

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
