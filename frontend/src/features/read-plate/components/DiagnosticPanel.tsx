import type { PlateLabels } from "../types/readplate.types"
import "./DiagnosticPanel.css"

interface DiagnosticPanelProps {
  labels: PlateLabels
}

function getStatusColor(key: string, value: string): "success" | "warning" | "danger" | "neutral" {
  if (key === "legible") {
    return value === "Legible" ? "success" : "danger"
  }
  if (key === "oclusion") {
    if (value === "No") return "success"
    if (value === "Parcial") return "warning"
    return "danger"
  }
  if (key === "reflejo" || key === "sucia") {
    return value === "No" ? "success" : "danger"
  }
  return "neutral"
}

function StatusIcon({ status }: { status: string }) {
  if (status === "success") {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="diag-icon diag-success">
        <circle cx="7" cy="7" r="6" fill="currentColor" fillOpacity="0.2"/>
        <path d="M4 7.5L6 9.5L10 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    )
  }
  if (status === "warning") {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="diag-icon diag-warning">
        <circle cx="7" cy="7" r="6" fill="currentColor" fillOpacity="0.2"/>
        <path d="M7 4V7.5M7 10V10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  }
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="diag-icon diag-danger">
      <circle cx="7" cy="7" r="6" fill="currentColor" fillOpacity="0.2"/>
      <path d="M4.5 4.5L9.5 9.5M9.5 4.5L4.5 9.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  )
}

export function DiagnosticPanel({ labels }: DiagnosticPanelProps) {
  const items = [
    { label: "Legibilidad", value: labels.legible, status: getStatusColor("legible", labels.legible) },
    { label: "Oclusión", value: labels.oclusion, status: getStatusColor("oclusion", labels.oclusion) },
    { label: "Reflejo", value: labels.reflejo, status: getStatusColor("reflejo", labels.reflejo) },
    { label: "Suciedad", value: labels.sucia, status: getStatusColor("sucia", labels.sucia) },
  ]

  const hasIssues = items.some(item => item.status === "danger" || item.status === "warning")

  return (
    <div className={`diag-panel ${hasIssues ? "diag-panel--issues" : "diag-panel--clean"}`}>
      <div className="diag-header">
        <div className="diag-title-row">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="diag-sparkle">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
          <span className="diag-title">Diagnóstico Multimodal</span>
        </div>
        <span className="diag-model">Gemini 2.5 Flash</span>
      </div>

      <div className="diag-grid">
        {items.map((item) => (
          <div key={item.label} className={`diag-item diag-item--${item.status}`}>
            <StatusIcon status={item.status} />
            <div className="diag-item-text">
              <span className="diag-item-label">{item.label}</span>
              <span className="diag-item-value">{item.value}</span>
            </div>
          </div>
        ))}
      </div>
      
      {hasIssues && (
        <div className="diag-alert">
          Condiciones físicas detectadas podrían afectar la precisión del OCR.
        </div>
      )}
    </div>
  )
}
