from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Double, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class VehicleType(Base):
    __tablename__ = "vehicle_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)
    
class ModelIA(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    architecture_type = Column(String(50))
    model_type = Column(String(50))

    detections = relationship("PlateDetection", back_populates="model")
    training_runs = relationship("ModelTraining", back_populates="model")

class QualityLabel(Base):
    __tablename__ = "quality_labels"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(Text)

class ModelTraining(Base):
    __tablename__ = "model_training"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(Integer, nullable=False)
    accuracy = Column(Double)
    loss = Column(Double)
    training_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    model = relationship("ModelIA", back_populates="training_runs")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    version = Column(String(20))
    created_by = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    files = relationship("DatasetFile", back_populates="dataset", cascade="all, delete-orphan")


class DatasetFile(Base):
    __tablename__ = "dataset_files"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50))
    file_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    dataset = relationship("Dataset", back_populates="files")


class PlateDetection(Base):
    __tablename__ = "plate_detections"
    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String(15), nullable=False, index=True)
    confidence = Column(Double)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="SET NULL"))
    vehicle_type_id = Column(Integer, ForeignKey("vehicle_types.id", ondelete="SET NULL"))
    inference_time_ms = Column(Double)
    image_path = Column(Text)
    user_validated = Column(Boolean, nullable=False, default=False, server_default="false")
    user_is_correct = Column(Boolean, nullable=True)
    user_corrected_text = Column(String(15), nullable=True)
    user_feedback_date = Column(DateTime(timezone=True), nullable=True)
    user_feedback_by = Column(String(100), nullable=True)
    detection_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relaciones
    model = relationship("ModelIA", back_populates="detections")
    vehicle = relationship("VehicleType")
    quality_checks = relationship("DetectionQuality", back_populates="detection", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="detection", cascade="all, delete-orphan")

class DetectionQuality(Base):
    __tablename__ = "detection_quality"
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("plate_detections.id", ondelete="CASCADE"), nullable=False)
    quality_label_id = Column(Integer, ForeignKey("quality_labels.id", ondelete="CASCADE"), nullable=False)
    value = Column(String(50), nullable=False, index=True)

    detection = relationship("PlateDetection", back_populates="quality_checks")
    label = relationship("QualityLabel")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("plate_detections.id", ondelete="CASCADE"), nullable=False)
    checked_by = Column(String(100), nullable=False)
    check_reason = Column(Text, nullable=False)
    check_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    detection = relationship("PlateDetection", back_populates="audit_logs")
