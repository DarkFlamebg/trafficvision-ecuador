-- ==============================================================================
-- 🚗 TrafficVision Database Schema
-- 🐘 Motor: PostgreSQL
-- 📝 Descripción: Esquema relacional para el sistema de detección, 
-- lectura y clasificación de placas vehiculares.
-- ==============================================================================

-- ==========================================
-- SECCIÓN 1: TABLAS BASE Y DE CONFIGURACIÓN
-- ==========================================

-- 1. Tabla Maestra: Tipos de Vehículo
CREATE TABLE vehicle_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

-- 2. Tabla Maestra: Modelos de IA
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    architecture_type VARCHAR(50), -- ej. YOLO, RT-DETR, EasyOCR
    model_type VARCHAR(50)         -- ej. identification, classification
);

-- 3. Tabla Maestra: Etiquetas de Calidad (Gemini)
CREATE TABLE quality_labels (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,  -- ej. Legibilidad, Oclusión, Reflejo
    description TEXT
);

-- ==========================================
-- SECCIÓN 2: GESTIÓN DE DATASETS Y ENTRENAMIENTO
-- ==========================================

-- 4. Datasets utilizados para entrenamiento
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    created_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Archivos físicos de los datasets
CREATE TABLE dataset_files (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_dataset_files FOREIGN KEY (dataset_id) 
        REFERENCES datasets(id) ON DELETE CASCADE
);

-- 6. Historial de entrenamiento de modelos
CREATE TABLE model_training (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    accuracy DOUBLE PRECISION,
    loss DOUBLE PRECISION,
    training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_training_model FOREIGN KEY (model_id) 
        REFERENCES models(id) ON DELETE CASCADE,
    CONSTRAINT fk_training_dataset FOREIGN KEY (dataset_id) 
        REFERENCES datasets(id) ON DELETE SET NULL
);

-- ==========================================
-- SECCIÓN 3: INFERENCIA Y DETECCIÓN EN VIVO
-- ==========================================

-- 7. Tabla Principal: Detecciones de Placas
CREATE TABLE plate_detections (
    id SERIAL PRIMARY KEY,
    plate_text VARCHAR(15) NOT NULL,
    confidence DOUBLE PRECISION,           -- Confianza final / OCR
    model_id INTEGER,                      -- Modelo que generó la inferencia
    vehicle_type_id INTEGER,
    inference_time_ms DOUBLE PRECISION,
    image_path TEXT,                       -- Ruta de la captura en el Storage
    detection_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_detection_model FOREIGN KEY (model_id) 
        REFERENCES models(id) ON DELETE SET NULL,
    CONSTRAINT fk_detection_vehicle FOREIGN KEY (vehicle_type_id) 
        REFERENCES vehicle_types(id) ON DELETE SET NULL
);

-- 8. Detalles de Calidad de Imagen (Evaluación Gemini u OCR)
CREATE TABLE detection_quality (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER NOT NULL,
    quality_label_id INTEGER NOT NULL,
    value VARCHAR(50) NOT NULL,            -- "Sí", "No", "Parcial", "Legible"
    
    CONSTRAINT fk_quality_detection FOREIGN KEY (detection_id) 
        REFERENCES plate_detections(id) ON DELETE CASCADE,
    CONSTRAINT fk_quality_label FOREIGN KEY (quality_label_id) 
        REFERENCES quality_labels(id) ON DELETE CASCADE
);

-- ==========================================
-- SECCIÓN 4: SEGURIDAD Y AUDITORÍA
-- ==========================================

-- 9. Auditoría Anticorrupción y Trazabilidad
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER NOT NULL,
    checked_by VARCHAR(100) NOT NULL,      -- Usuario o ID del oficial/sistema
    check_reason TEXT NOT NULL,            -- Razón de la consulta (ej. Control ruta)
    check_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_audit_detection FOREIGN KEY (detection_id) 
        REFERENCES plate_detections(id) ON DELETE CASCADE
);

-- ==========================================
-- SECCIÓN 5: ÍNDICES DE RENDIMIENTO (B-TREE)
-- ==========================================

-- Índice para búsquedas rápidas por placa (muy común)
CREATE INDEX idx_plate_text ON plate_detections(plate_text);

-- Índice para reportes por rango de fechas
CREATE INDEX idx_detection_date ON plate_detections(detection_date);

-- Índice para filtrar rápidamente la calidad de las detecciones
CREATE INDEX idx_quality_value ON detection_quality(value);

-- Índice para revisar historiales de auditoría
CREATE INDEX idx_audit_date ON audit_logs(check_date);
