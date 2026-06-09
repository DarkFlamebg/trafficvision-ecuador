-- 1. Creación de la Base de Datos
USE placas_ia;

-- 2. Tabla Maestra: Modelos de IA
CREATE TABLE IF NOT EXISTS models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    architecture_type VARCHAR(50), -- CNN, Transformer, etc.
    model_type VARCHAR(50)         -- identification, classification
) ENGINE=InnoDB;

-- 3. Tabla Maestra: Tipos de Vehículo (Clases COCO)
CREATE TABLE IF NOT EXISTS vehicle_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL -- Automóvil, Motocicleta, etc.
) ENGINE=InnoDB;

-- 4. Tabla Maestra: Catálogo de Etiquetas de Calidad
CREATE TABLE IF NOT EXISTS quality_labels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,    -- Oclusión, Reflejo, Suciedad
    description TEXT
) ENGINE=InnoDB;

-- 5. Tabla Principal: Detecciones de Placas
CREATE TABLE IF NOT EXISTS plate_detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_text VARCHAR(10) NOT NULL,
    confidence FLOAT,              -- Confianza del OCR
    model_id INT,                  -- FK a models
    vehicle_type_id INT,           -- FK a vehicle_types
    inference_time_ms FLOAT,       -- Tiempo de procesamiento
    image_path TEXT,               -- Ruta de la captura
    user_validated TINYINT(1) DEFAULT 0,
    user_is_correct TINYINT(1) DEFAULT NULL,
    user_corrected_text VARCHAR(15) DEFAULT NULL,
    user_feedback_date TIMESTAMP NULL DEFAULT NULL,
    user_feedback_by VARCHAR(100) DEFAULT NULL,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_plate_detections_model FOREIGN KEY (model_id) 
        REFERENCES models(id) ON DELETE SET NULL,
    CONSTRAINT fk_plate_detections_vehicle FOREIGN KEY (vehicle_type_id) 
        REFERENCES vehicle_types(id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- 6. Tabla Relacional: Calidad de la Detección (Resultados Gemini) Una detección puede tener múltiples etiquetas de calidad (1-N)
CREATE TABLE IF NOT EXISTS detection_quality (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT NOT NULL,
    quality_label_id INT NOT NULL,
    value VARCHAR(50),             -- "Sí", "No", "Parcial", "Severa"
    
    CONSTRAINT fk_quality_detection FOREIGN KEY (detection_id) 
        REFERENCES plate_detections(id) ON DELETE CASCADE,
    CONSTRAINT fk_quality_label FOREIGN KEY (quality_label_id) 
        REFERENCES quality_labels(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 7. Tabla de Auditoría: Control de Seguridad
CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT,
    checked_by VARCHAR(50),        -- Usuario o Sistema
    check_reason TEXT,
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_audit_detection FOREIGN KEY (detection_id) 
        REFERENCES plate_detections(id) ON DELETE CASCADE
) ENGINE=InnoDB;


SHOW CREATE TABLE plate_detections;