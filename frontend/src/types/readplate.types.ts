// ── Types ──────────────────────────────────────────────────────────────────────

export interface PlateLabels {
  legible:  "Legible" | "Ilegible"
  oclusion: "No" | "Parcial" | "Severa"
  reflejo:  "No" | "Sí"
  sucia:    "No" | "Sí"
}

export interface VehicleInfo {
  type:       string
  type_es:    string
  bbox:       [number, number, number, number]
  confidence: number
}

export interface PlateResult {
  bbox:             [number, number, number, number]
  yolo_confidence:  number
  plate:            string
  ocr_confidence:   number
  labels:           PlateLabels
  vehicle:          VehicleInfo | null
}

// ── Tipos para Comparación de Modelos ──────────────────────────────────────
export interface DetectionResult {
  plate:                string
  ocr_confidence:       number
  detector_confidence:  number
  bbox:                 [number, number, number, number]
}

export interface ModelDetectionResult {
  model:         string
  total:         number
  detections:    DetectionResult[]
  image_base64:  string
}

export interface ComparisonApiResponse {
  yolo:    ModelDetectionResult
  rtdetr:  ModelDetectionResult
  summary: {
    yolo_plates:   number
    rtdetr_plates: number
    total_unique:  number
  }
}

export interface ApiResponse {
  total:    number
  vehicles: number
  plates:   PlateResult[]
  processing_time_ms?: number
  video_metrics?: {
    total_unique_vehicles: number
    total_raw_detections:  number
    frames_processed:      number
    video_duration_s:      number
    processing_time_ms:    number
    vehicles_per_minute:   number
    by_type:               VideoTypeMetric[]
  }
  // Para imágenes: resultado de comparación
  yolo?:    ModelDetectionResult
  rtdetr?:  ModelDetectionResult
  summary?: {
    yolo_plates:   number
    rtdetr_plates: number
    total_unique:  number
  }
}

export interface DetectionReport {
  id:             number
  filename:       string
  location:       string
  vehicleType:    string
  confidence:     number
  dateTime:       string
  processingTime: number
  processed:      boolean
  coordinates:    string
}

export interface VideoTypeMetric {
  type:    string
  count:   number
  percent: number
}
