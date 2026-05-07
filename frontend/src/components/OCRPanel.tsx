import type { PlateDetection } from "../types/comparison_types"

interface OCRPanelProps {
  plate: PlateDetection | null
  realPlate?: string
  color: string
  modelName: string
}

export function OCRPanel({ plate, realPlate, color, modelName }: OCRPanelProps) {
  const isCorrect = !!(realPlate && plate?.plate && plate.plate === realPlate)
  const hasPlate = !!plate?.plate

  // Simular confianza por carácter si no viene del backend
  const charConfidences = plate?.char_confidences || 
    (plate?.plate ? plate.plate.split('').map(() => plate.ocr_confidence) : [])

  return (
    <div className="ocr-panel">
      <div className="ocr-panel-header" style={{ borderColor: color }}>
        <span className="ocr-panel-title">OCR</span>
        <span className="ocr-panel-model" style={{ color }}>{modelName}</span>
      </div>
      
      <div className="ocr-panel-body">
        {hasPlate ? (
          <>
            {/* Texto de placa grande */}
            <div className="ocr-plate-display" style={{ color }}>
              {plate.plate}
            </div>

            {/* Barras de confianza por carácter */}
            <div className="ocr-char-bars">
              {plate.plate.split('').map((char, idx) => {
                const conf = (charConfidences[idx] || 0) * 100
                const isHigh = conf > 80
                return (
                  <div key={idx} className="ocr-char-item">
                    <div className="ocr-char-bar-wrap">
                      <div 
                        className="ocr-char-bar" 
                        style={{ 
                          width: `${conf}%`,
                          background: isHigh ? '#22d38a' : '#ef4444'
                        }}
                      />
                    </div>
                    <span className="ocr-char-text" style={{ color: isHigh ? '#22d38a' : '#ef4444' }}>
                      {char}
                    </span>
                    <span className="ocr-char-pct">{conf.toFixed(1)}%</span>
                  </div>
                )
              })}
            </div>

            {/* Resultado de validación */}
            <div className={`ocr-validation-result ${isCorrect ? 'correct' : 'error'}`}>
              {isCorrect ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8L6.5 11.5L13 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>RESULTADO CORRECTO</span>
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span>RESULTADO INCORRECTO</span>
                </>
              )}
            </div>
          </>
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