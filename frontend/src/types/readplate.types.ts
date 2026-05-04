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

// readplate.types.ts / detection_type.ts
export interface PlateResult {
  bbox:                 [number, number, number, number]
  detector_confidence:  number  
  detector:             string
  plate:                string
  ocr_confidence:       number
  labels:               PlateLabels | null
  vehicle:              VehicleInfo | null
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
