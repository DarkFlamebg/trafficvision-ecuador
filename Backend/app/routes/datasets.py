from fastapi import APIRouter

router = APIRouter()

@router.get("/datasets")
def get_datasets():
    return {"message": "Lista de datasets"}