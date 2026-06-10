import csv
from datetime import datetime, timezone
from io import StringIO

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.database.connection import get_db
from app.database.models import DetectionQuality, ModelIA, PlateDetection

router = APIRouter()


def _format_dt(value):
    if value is None:
        return ""
    return value.isoformat()


def _detection_query(db: Session):
    return (
        db.query(PlateDetection)
        .options(
            joinedload(PlateDetection.model),
            joinedload(PlateDetection.vehicle),
            joinedload(PlateDetection.quality_checks).joinedload(DetectionQuality.label),
            joinedload(PlateDetection.audit_logs),
        )
        .order_by(PlateDetection.detection_date.desc(), PlateDetection.id.desc())
    )


def _apply_filters(query, plate: str | None, validated: str | None, model_id: int | None):
    if plate:
        query = query.filter(PlateDetection.plate_text.ilike(f"%{plate.strip()}%"))
    if validated == "validated":
        query = query.filter(PlateDetection.user_validated.is_(True))
    elif validated == "pending":
        query = query.filter(PlateDetection.user_validated.is_(False))
    if model_id:
        query = query.filter(PlateDetection.model_id == model_id)
    return query


def _sorted_audit_logs(detection: PlateDetection):
    return sorted(
        detection.audit_logs,
        key=lambda item: item.check_date or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _serialize_detection(detection: PlateDetection):
    return {
        "id": detection.id,
        "plate_text": detection.plate_text,
        "confidence": detection.confidence,
        "model": detection.model.name if detection.model else "",
        "vehicle_type": detection.vehicle.name if detection.vehicle else "",
        "inference_time_ms": detection.inference_time_ms,
        "image_path": detection.image_path or "",
        "detection_date": _format_dt(detection.detection_date),
        "user_validated": detection.user_validated,
        "user_is_correct": detection.user_is_correct,
        "user_corrected_text": detection.user_corrected_text,
        "user_feedback_date": _format_dt(detection.user_feedback_date),
        "user_feedback_by": detection.user_feedback_by,
        "quality_checks": [
            {
                "label": check.label.name if check.label else str(check.quality_label_id),
                "value": check.value,
            }
            for check in detection.quality_checks
        ],
        "audit_logs": [
            {
                "checked_by": log.checked_by,
                "check_reason": log.check_reason,
                "check_date": _format_dt(log.check_date),
            }
            for log in _sorted_audit_logs(detection)
        ],
    }


@router.get("/anti-corruption/detections")
def list_detections(
    plate: str | None = None,
    validated: str | None = None,
    model_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    query = _apply_filters(_detection_query(db), plate, validated, model_id)
    total = query.count()
    detections = query.offset(safe_offset).limit(safe_limit).all()

    validated_count = db.query(PlateDetection).filter(PlateDetection.user_validated.is_(True)).count()
    pending_count = db.query(PlateDetection).filter(PlateDetection.user_validated.is_(False)).count()
    incorrect_count = db.query(PlateDetection).filter(PlateDetection.user_is_correct.is_(False)).count()
    avg_confidence = db.query(func.avg(PlateDetection.confidence)).scalar() or 0
    models = db.query(ModelIA).order_by(ModelIA.name.asc()).all()

    return {
        "total": total,
        "limit": safe_limit,
        "offset": safe_offset,
        "summary": {
            "total_detections": db.query(PlateDetection).count(),
            "validated": validated_count,
            "pending": pending_count,
            "incorrect": incorrect_count,
            "avg_confidence": float(avg_confidence),
        },
        "models": [{"id": model.id, "name": model.name} for model in models],
        "detections": [_serialize_detection(detection) for detection in detections],
    }


@router.get("/anti-corruption/reports/detections.csv")
def download_detection_report(
    plate: str | None = None,
    validated: str | None = None,
    model_id: int | None = None,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    safe_limit = max(1, min(limit, 5000))
    detections = _apply_filters(_detection_query(db), plate, validated, model_id).limit(safe_limit).all()

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "id",
        "plate_text",
        "confidence",
        "model",
        "vehicle_type",
        "inference_time_ms",
        "image_path",
        "detection_date",
        "user_validated",
        "user_is_correct",
        "user_corrected_text",
        "user_feedback_date",
        "user_feedback_by",
        "quality_checks",
        "audit_logs",
    ])

    for detection in detections:
        quality_checks = "; ".join(
            f"{check.label.name if check.label else check.quality_label_id}: {check.value}"
            for check in detection.quality_checks
        )
        audit_logs = "; ".join(
            f"{_format_dt(log.check_date)} | {log.checked_by} | {log.check_reason}"
            for log in _sorted_audit_logs(detection)
        )

        writer.writerow([
            detection.id,
            detection.plate_text,
            detection.confidence,
            detection.model.name if detection.model else "",
            detection.vehicle.name if detection.vehicle else "",
            detection.inference_time_ms,
            detection.image_path or "",
            _format_dt(detection.detection_date),
            "si" if detection.user_validated else "no",
            "" if detection.user_is_correct is None else ("si" if detection.user_is_correct else "no"),
            detection.user_corrected_text or "",
            _format_dt(detection.user_feedback_date),
            detection.user_feedback_by or "",
            quality_checks,
            audit_logs,
        ])

    buffer.seek(0)
    filename = f"reporte-detecciones-{datetime.now(timezone.utc).date().isoformat()}.csv"
    return StreamingResponse(
        iter([buffer.getvalue().encode("utf-8-sig")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
