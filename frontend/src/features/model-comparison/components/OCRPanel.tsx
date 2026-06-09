import { useState, useRef } from "react"
import type { PlateDetection } from "../types/comparison_types"
import { DiagnosticPanelCompact } from "./DiagnosticPanelCompact"

interface OCRPanelProps {
  plates: PlateDetection[]        // array — puede tener 0, 1 o N placas
  realPlate?: string
  color: string
  modelName: string
  onFeedback?: (plate: PlateDetection, isCorrect: boolean, correctedPlateText?: string) => Promise<void>
}

function SinglePlateCard({
  plate,
  realPlate,
  color,
  onFeedback,
  index,
  total,
}: {
  plate: PlateDetection
  realPlate?: string
  color: string
  onFeedback?: (plate: PlateDetection, isCorrect: boolean, correctedPlateText?: string) => Promise<void>
  index: number
  total: number
}) {
  const [feedback, setFeedback] = useState<boolean | null>(null)
  const [correctedText, setCorrectedText] = useState<string>("")
  const [submitting, setSubmitting] = useState(false)
  const [popupMessage, setPopupMessage] = useState<string | null>(null)
  const [popupType, setPopupType] = useState<"success" | "error" | null>(null)
  const timeoutRef = useRef<number | null>(null)

  const isValid   = (plate as any).ocr_valid !== false   // undefined treated as valid (backwards compat)
  const textColor = plate.plate ? (isValid ? color : "#f59e0b") : "var(--text-lo)"
  const isCorrect = !!(realPlate && plate.plate && plate.plate === realPlate)

  const showPopup = (message: string, type: "success" | "error") => {
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current)
    }
    setPopupMessage(message)
    setPopupType(type)
    timeoutRef.current = window.setTimeout(() => {
      setPopupMessage(null)
      setPopupType(null)
      timeoutRef.current = null
    }, 2800)
  }

  const handleSelectFeedback = (correct: boolean) => {
    setFeedback(correct)
    setPopupMessage(null)
    setPopupType(null)
  }

  const handleSubmitFeedback = async () => {
    if (feedback === null || !plate.detection_id || !onFeedback) return

    setSubmitting(true)
    try {
      await onFeedback(plate, feedback, correctedText.trim() || undefined)
      showPopup("Feedback enviado correctamente.", "success")
    } catch (error) {
      showPopup("Error al enviar feedback. Intenta de nuevo.", "error")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="ocr-plate-card"
      style={{ position: "relative", borderColor: (plate.plate ? textColor : "var(--border)") + "44" }}
    >
      {/* Indicador de placa N/Total si hay más de una */}
      {total > 1 && (
        <div className="ocr-plate-index" style={{ color: textColor }}>
          PLACA {index + 1} / {total}
        </div>
      )}

      {/* Badge de incerteza si no pasa el filtro */}
      {plate.plate && !isValid && (
        <div className="ocr-uncertain-tag">⚠ Lectura incierta</div>
      )}

      {/* Texto de placa */}
      <div className="ocr-plate-display" style={{ color: textColor }}>
        {plate.plate || "Sin lectura"}
      </div>

      {/* Barra de confianza */}
      <div className="ocr-char-bars">
        <div className="ocr-char-item" style={{ gridTemplateColumns: "1fr auto 46px" }}>
          <div className="ocr-char-bar-wrap">
            <div
              className="ocr-char-bar"
              style={{
                width: `${plate.ocr_confidence * 100}%`,
                background: color,
              }}
            />
          </div>
          <span className="ocr-char-text" style={{ color: "var(--text-mid)", fontSize: "0.65rem" }}>
            CONF.
          </span>
          <span className="ocr-char-pct">{(plate.ocr_confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* ── Etiquetas de Calidad (Gemini) ── */}
      {plate.labels && (
        <DiagnosticPanelCompact labels={plate.labels} color={color} />
      )}

      {/* Validación automática */}
      {realPlate && (
        <div className={`ocr-validation-result ${isCorrect ? "correct" : "error"}`}>
          {isCorrect ? (
            <>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M3 8L6.5 11.5L13 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span>RESULTADO CORRECTO</span>
            </>
          ) : (
            <>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span>RESULTADO INCORRECTO</span>
            </>
          )}
        </div>
      )}

      {/* Feedback manual */}
      <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
        <button
          onClick={() => handleSelectFeedback(true)}
          style={{
            flex: 1, padding: "0.45rem",
            background: feedback === true ? "rgba(34,197,94,0.12)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${feedback === true ? "var(--ok)" : "var(--border)"}`,
            borderRadius: "var(--radius-sm)",
            color: feedback === true ? "var(--ok)" : "var(--text-mid)",
            fontSize: "0.68rem", cursor: "pointer", transition: "all 0.2s",
            fontFamily: "var(--font-mono)", fontWeight: 700, letterSpacing: "0.05em",
          }}
        >
          ✓ CORRECTO
        </button>
        <button
          onClick={() => handleSelectFeedback(false)}
          style={{
            flex: 1, padding: "0.45rem",
            background: feedback === false ? "rgba(239,68,68,0.12)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${feedback === false ? "var(--err)" : "var(--border)"}`,
            borderRadius: "var(--radius-sm)",
            color: feedback === false ? "var(--err)" : "var(--text-mid)",
            fontSize: "0.68rem", cursor: "pointer", transition: "all 0.2s",
            fontFamily: "var(--font-mono)", fontWeight: 700, letterSpacing: "0.05em",
          }}
        >
          ✕ INCORRECTO
        </button>
      </div>

      {feedback === false && (
        <div style={{ marginTop: "0.75rem" }}>
          <label style={{ display: "block", marginBottom: "0.4rem", fontSize: "0.72rem", color: "var(--text-mid)" }}>
            Corregir texto (opcional)
          </label>
          <input
            value={correctedText}
            onChange={(event) => setCorrectedText(event.target.value)}
            placeholder="Ej. ABC-1234"
            style={{
              width: "100%",
              padding: "0.55rem 0.75rem",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)",
              background: "var(--bg-panel)",
              color: "var(--text)",
              fontSize: "0.88rem",
            }}
          />
        </div>
      )}

      <button
        onClick={handleSubmitFeedback}
        disabled={feedback === null || submitting || !plate.detection_id || !onFeedback}
        style={{
          width: "100%",
          marginTop: "0.75rem",
          padding: "0.65rem",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border)",
          background: plate.detection_id ? "var(--primary)" : "var(--bg-panel)",
          color: plate.detection_id ? "white" : "var(--text-lo)",
          cursor: feedback === null || submitting || !plate.detection_id ? "not-allowed" : "pointer",
          fontSize: "0.85rem",
          fontWeight: 700,
        }}
      >
        {submitting ? "Enviando..." : "Aceptar feedback"}
      </button>

      {plate.detection_id == null && (
        <div style={{ marginTop: "0.5rem", color: "var(--text-lo)", fontSize: "0.75rem" }}>
          Esta placa no se guardó como detección válida, por lo que no puede enviar feedback.
        </div>
      )}

      {popupMessage && (
        <div
          role="status"
          style={{
            position: "absolute",
            top: "0.75rem",
            left: "0.75rem",
            right: "0.75rem",
            padding: "0.85rem 1rem",
            borderRadius: "0.9rem",
            background: popupType === "success" ? "rgba(16,185,129,0.95)" : "rgba(239,68,68,0.95)",
            color: "white",
            boxShadow: "0 18px 40px rgba(0,0,0,0.16)",
            zIndex: 10,
            fontSize: "0.85rem",
            textAlign: "center",
          }}
        >
          {popupMessage}
        </div>
      )}
    </div>
  )
}

export function OCRPanel({ plates, realPlate, color, modelName, onFeedback }: OCRPanelProps) {
  const hasPlates = plates.length > 0

  return (
    <div className="ocr-panel">
      <div className="ocr-panel-header" style={{ borderColor: color }}>
        <span className="ocr-panel-title">OCR</span>
        <span className="ocr-panel-model" style={{ color }}>{modelName}</span>
      </div>

      <div className="ocr-panel-body">
        {hasPlates ? (
          <div className="ocr-plates-list">
            {plates.map((plate, i) => (
              <SinglePlateCard
                key={i}
                plate={plate}
                realPlate={realPlate}
                color={color}
                onFeedback={onFeedback}
                index={i}
                total={plates.length}
              />
            ))}
          </div>
        ) : (
          <div className="ocr-no-detection">
            <div className="ocr-no-icon">⚠</div>
            <p>Sin detección OCR</p>
            <span>No se encontró placa en la imagen</span>
          </div>
        )}
      </div>
    </div>
  )
}