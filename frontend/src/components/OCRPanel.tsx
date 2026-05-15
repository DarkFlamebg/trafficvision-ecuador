import { useState } from "react"
import type { PlateDetection } from "../types/comparison_types"

interface OCRPanelProps {
  plates: PlateDetection[]        // array — puede tener 0, 1 o N placas
  realPlate?: string
  color: string
  modelName: string
  onFeedback?: (plate: PlateDetection, isCorrect: boolean) => void
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
  onFeedback?: (plate: PlateDetection, isCorrect: boolean) => void
  index: number
  total: number
}) {
  const [feedback, setFeedback] = useState<boolean | null>(null)

  const isValid   = plate.ocr_valid !== false   // undefined treated as valid (backwards compat)
  const textColor = plate.plate ? (isValid ? color : "#f59e0b") : "var(--text-lo)"
  const isCorrect = !!(realPlate && plate.plate && plate.plate === realPlate)

  const handleFeedback = (correct: boolean) => {
    setFeedback(correct)
    onFeedback?.(plate, correct)
  }

  return (
    <div
      className="ocr-plate-card"
      style={{ borderColor: (plate.plate ? textColor : "var(--border)") + "44" }}
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
          onClick={() => handleFeedback(true)}
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
          onClick={() => handleFeedback(false)}
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