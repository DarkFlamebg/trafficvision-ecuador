import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import psycopg2
from sqlalchemy.pool import StaticPool

# Forzar encoding UTF-8 al leer el .env
load_dotenv(encoding="utf-8")

# Leer credenciales por separado
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "postgres")


def _create_psycopg2_connection():
    """
    Usa psycopg2 con parámetros individuales (no URL), lo que permite
    contraseñas con caracteres especiales (@, ó, ñ, etc.) sin errores de encoding.
    """
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        client_encoding="UTF8",
        sslmode="require",
    )


# SQLAlchemy con creator personalizado (evita parsear URL con caracteres especiales)
engine = create_engine(
    "postgresql+psycopg2://",
    creator=_create_psycopg2_connection,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()