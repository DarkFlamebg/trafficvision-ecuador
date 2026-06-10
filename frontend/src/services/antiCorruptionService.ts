import API from "./api"

export interface DetectionQualityCheck {
  label: string
  value: string
}

export interface DetectionAuditLog {
  checked_by: string | null
  check_reason: string | null
  check_date: string
}

export interface DetectionRow {
  id: number
  plate_text: string
  confidence: number
  model: string
  vehicle_type: string
  inference_time_ms: number
  image_path: string
  detection_date: string
  user_validated: boolean
  user_is_correct: boolean | null
  user_corrected_text: string | null
  user_feedback_date: string
  user_feedback_by: string | null
  quality_checks: DetectionQualityCheck[]
  audit_logs: DetectionAuditLog[]
}

export interface ModelOption {
  id: number
  name: string
}

export interface AntiCorruptionSummary {
  total_detections: number
  validated: number
  pending: number
  incorrect: number
  avg_confidence: number
}

export interface AntiCorruptionResponse {
  total: number
  limit: number
  offset?: number
  summary: AntiCorruptionSummary
  models: ModelOption[]
  detections: DetectionRow[]
}

export interface AntiCorruptionFilters {
  plate?: string
  validated?: string
  model_id?: number
  limit?: number
  offset?: number
}

export async function getDetectionRows(filters: AntiCorruptionFilters = {}): Promise<AntiCorruptionResponse> {
  const response = await API.get<AntiCorruptionResponse>("/api/v1/anti-corruption/detections", {
    params: filters,
  })
  return response.data
}

export interface DetectionFeedbackPayload {
  detection_id: number
  is_correct: boolean
  corrected_plate_text?: string | null
  comments?: string | null
  user_id?: string | null
}

export async function submitDetectionFeedback(payload: DetectionFeedbackPayload) {
  await API.post(
    "/api/v1/compare/feedback",
    payload,
    { headers: { "Content-Type": "application/json" } }
  )
}

export async function downloadDetectionReport(filters: AntiCorruptionFilters = {}) {
  const response = await API.get<Blob>("/api/v1/anti-corruption/reports/detections.csv", {
    params: filters,
    responseType: "blob",
    headers: {
      Accept: "text/csv",
    },
  })

  const blob = new Blob([response.data], { type: "text/csv;charset=utf-8;" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = `reporte-detecciones-${new Date().toISOString().slice(0, 10)}.csv`
  link.click()
  URL.revokeObjectURL(url)
}
