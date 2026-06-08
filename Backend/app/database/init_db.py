from sqlalchemy.orm import Session
from app.database.connection import SessionLocal, engine
from app.database.models import Base, ModelIA, VehicleType, QualityLabel
import os

def seed_db():
    # Asegurarnos de que las tablas existan (útil si no se ha corrido el .sql manual, aunque preferimos el .sql)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Seed Modelos usados por el pipeline de detección
        model_names = [
            ("YOLOv11n", "YOLOv11n", "plate_detection"),
            ("RT-DETR", "RT-DETR", "plate_detection"),
            ("EfficientDet-D2", "EfficientDet-D2", "plate_detection"),
            ("EasyOCR", "EasyOCR", "ocr"),
            ("Gemini 2.5 Flash", "Gemini", "classification"),
        ]
        for name, arch, mtype in model_names:
            if not db.query(ModelIA).filter_by(name=name).first():
                db.add(ModelIA(name=name, architecture_type=arch, model_type=mtype))
        
        # Seed Vehículos
        vehicle_names = ["Automóvil", "Motocicleta", "Autobús", "Camión", "Desconocido"]
        for vname in vehicle_names:
            if not db.query(VehicleType).filter_by(name=vname).first():
                db.add(VehicleType(name=vname))
                
        # Seed Quality Labels
        labels = [
            ("Oclusión", "Obstrucción física de la placa"),
            ("Reflejo", "Destello de luz que impide lectura"),
            ("Suciedad", "Lodo, polvo o deterioro"),
            ("Legibilidad", "Estado general de lectura")
        ]
        for lname, ldesc in labels:
            if not db.query(QualityLabel).filter_by(name=lname).first():
                db.add(QualityLabel(name=lname, description=ldesc))
                
        db.commit()
        print("Base de datos inicializada y poblada con catálogos base.")
    except Exception as e:
        print(f"Error inicializando BD: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_db()
