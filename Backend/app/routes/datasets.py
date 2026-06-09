from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload

from app.database.connection import get_db
from app.database.models import Dataset, DatasetFile

router = APIRouter()


class DatasetFileCreate(BaseModel):
    file_name: str
    file_type: Optional[str] = None
    file_path: HttpUrl


class DatasetFileResponse(BaseModel):
    id: int
    file_name: str
    file_type: Optional[str] = None
    file_path: str

    class Config:
        orm_mode = True


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    created_by: Optional[str] = None
    files: Optional[List[DatasetFileCreate]] = []


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    created_by: Optional[str] = None
    created_at: str
    files: List[DatasetFileResponse] = []

    class Config:
        orm_mode = True


@router.get("/datasets", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    datasets = (
        db.query(Dataset)
        .options(joinedload(Dataset.files))
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return datasets


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = (
        db.query(Dataset)
        .options(joinedload(Dataset.files))
        .filter(Dataset.id == dataset_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return dataset


@router.post("/datasets", response_model=DatasetResponse)
def create_dataset(dataset_in: DatasetCreate, db: Session = Depends(get_db)):
    dataset = Dataset(
        name=dataset_in.name,
        description=dataset_in.description,
        version=dataset_in.version,
        created_by=dataset_in.created_by,
    )
    db.add(dataset)
    db.flush()

    for file_data in dataset_in.files or []:
        dataset_file = DatasetFile(
            dataset_id=dataset.id,
            file_name=file_data.file_name,
            file_type=file_data.file_type,
            file_path=str(file_data.file_path),
        )
        db.add(dataset_file)

    db.commit()
    db.refresh(dataset)
    # Cargar archivos asociados para la respuesta
    dataset.files
    return dataset
