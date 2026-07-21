# app/services/persistence_service.py
# Lógica de guardado en base de datos extraída de main.py.
# Centraliza toda interacción con PostgreSQL para las detecciones de placas.

from sqlalchemy.orm import Session

from app.core.config import QUALITY_LABEL_MAP, DEFAULT_QUALITY_LABELS
from app.database.models import (
    PlateDetection,
    DetectionQuality,
    AuditLog,
    ModelIA,
    VehicleType,
    QualityLabel,
)


def ensure_quality_labels(db: Session) -> dict[str, QualityLabel]:
    """
    Garantiza que todos los QualityLabel existan en la BD.
    Retorna un dict {clave_interna: objeto QualityLabel}.
    """
    labels = (
        db.query(QualityLabel)
        .filter(QualityLabel.name.in_(QUALITY_LABEL_MAP.values()))
        .all()
    )
    labels_by_name = {label.name: label for label in labels}

    for db_name in QUALITY_LABEL_MAP.values():
        if db_name not in labels_by_name:
            new_label = QualityLabel(
                name=db_name,
                description="Etiqueta de calidad de placa"
            )
            db.add(new_label)
            db.flush()
            labels_by_name[db_name] = new_label

    return {key: labels_by_name[db_name] for key, db_name in QUALITY_LABEL_MAP.items()}


def save_detections_to_db(
    db: Session,
    plates_found: list[dict],
    image_path: str,
) -> None:
    """
    Persiste las detecciones de placa, sus etiquetas de calidad y el log
    de auditoría correspondiente en PostgreSQL.

    Args:
        db:           Sesión de SQLAlchemy activa.
        plates_found: Lista de dicts con los resultados del pipeline.
        image_path:   Ruta de la imagen procesada.
    """
    yolo_model = db.query(ModelIA).filter_by(name="YOLOv11n").first()

    for plate in plates_found:
        vtype_name = plate["vehicle"]["type_es"] if plate["vehicle"] else "Desconocido"
        vtype      = db.query(VehicleType).filter_by(name=vtype_name).first()

        new_detection = PlateDetection(
            plate_text        = plate["plate"][:15],
            confidence        = plate["yolo_confidence"],
            model_id          = yolo_model.id if yolo_model else None,
            vehicle_type_id   = vtype.id if vtype else None,
            inference_time_ms = 120.0,  # TODO: medir tiempo real en el pipeline
            image_path        = image_path,
        )
        db.add(new_detection)
        db.flush()  # Obtener ID generado

        quality_labels = ensure_quality_labels(db)
        labels_dict    = plate.get("labels") or DEFAULT_QUALITY_LABELS

        for key, q_label in quality_labels.items():
            db.add(DetectionQuality(
                detection_id     = new_detection.id,
                quality_label_id = q_label.id,
                value            = str(labels_dict.get(key, DEFAULT_QUALITY_LABELS[key])),
            ))

        db.add(AuditLog(
            detection_id = new_detection.id,
            checked_by   = "TrafficVision AI",
            check_reason = "Detección en tiempo real",
        ))
