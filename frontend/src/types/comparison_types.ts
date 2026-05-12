// ── Comparison Types ────────────────────────────────────────────────────────

export type ModelType = "yolo" | "rtdetr"

export interface ComparisonMetrics {
  model:                   string
  inference_ms:            number
  vehicles_detected:       number
  avg_vehicle_confidence:  number
  plates_detected:         number
  avg_plate_confidence:    number
  plates_with_ocr:         number
  vehicles_by_type:        Record<string, number>
  // Para video
  total_unique_vehicles?:  number
  total_raw_detections?:   number
  total_plates_detected?:  number
  frames_processed?:       number
  video_duration_s?:       number
  processing_time_ms?:     number
  vehicles_per_minute?:    number
  avg_inference_ms?:       number
  by_type?:                VideoTypeMetric[]
  plates?:                 PlateDetection[]
}

export interface VideoTypeMetric {
  type:    string
  count:   number
  percent: number
}

export interface PlateDetection {
  bbox:                [number, number, number, number]
  plate:               string
  ocr_confidence:      number
  detector_confidence: number
  vehicle_type:        string | null
  frame?:              number
  timestamp_video?:    number
  vehicle_bbox?:       [number, number, number, number] | null
  // Para OCR visual por carácter (si el backend lo soporta)
  char_confidences?:   number[]
  image_base64?:       string
}

export interface VehicleDetection {
  type:       string
  type_es:    string
  bbox:       [number, number, number, number]
  confidence: number
}

export interface ComparisonImageResponse {
  model:    string
  metrics:  ComparisonMetrics
  vehicles: VehicleDetection[]
  plates:   PlateDetection[]
  // Imagen procesada con bounding boxes en base64
  processed_image?: string
}

export interface ComparisonResult {
  yolo?:   ComparisonImageResponse | ComparisonMetrics
  rtdetr?: ComparisonImageResponse | ComparisonMetrics
}

export interface WebSocketFrameMessage {
  type:            "frame"
  frame:           string
  progress:        number
  vehicle_counter: Record<string, number>
  plates_count:    number
  inference_ms:    number
  frame_num:       number
}

export interface WebSocketDoneMessage {
  type:    "done"
  metrics: ComparisonMetrics
}

export interface WebSocketStatusMessage {
  type:    "status"
  message: string
}

export interface WebSocketErrorMessage {
  type:    "error"
  message: string
}

export type WebSocketMessage = 
  | WebSocketFrameMessage 
  | WebSocketDoneMessage 
  | WebSocketStatusMessage 
  | WebSocketErrorMessage