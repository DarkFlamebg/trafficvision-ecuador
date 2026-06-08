import type { PlateLabels } from "../../../features/read-plate/types/readplate.types"
import "./DiagnosticPanelCompact.css"

interface DiagnosticPanelCompactProps {
  labels: PlateLabels
  color?: string
}

function getStatus(key: string, value: string): "ok" | "warn" | "err" {
  if (key === "legible") return value === "Legible" ? "ok" : "err"
  if (key === "oclusion") {
    if (value === "No") return "ok"
    if (value === "Parcial") return "warn"
    return "err"
  }
  if (key === "reflejo" || key === "sucia") {
    return value === "No" ? "ok" : "warn"
  }
  return "ok"
}

export function DiagnosticPanelCompact({ labels }: DiagnosticPanelCompactProps) {
  const items = [
    { label: "Legibilidad", value: labels.legible.substring(0, 3).toUpperCase(), status: getStatus("legible", labels.legible) },
    { label: "Oclusion", value: labels.oclusion.substring(0, 3).toUpperCase(), status: getStatus("oclusion", labels.oclusion) },
    { label: "Reflejo", value: labels.reflejo.substring(0, 2).toUpperCase(), status: getStatus("reflejo", labels.reflejo) },
    { label: "Suciedad", value: labels.sucia.substring(0, 3).toUpperCase(), status: getStatus("sucia", labels.sucia) },
  ]

  const hasIssues = items.some(item => item.status === "warn" || item.status === "err")

  return (
    <div className={`diag-compact ${hasIssues ? "diag-compact--issues" : "diag-compact--clean"}`}>
      <div className="diag-compact-header">
        <span>Clasificación de Placas</span>
      </div>
      <div className="diag-compact-grid">
        {items.map((item) => (
          <div key={item.label} className={`diag-badge diag-badge--${item.status}`}>
            <span className="diag-badge-lbl">{item.label}</span>
            <span className="diag-badge-val">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
