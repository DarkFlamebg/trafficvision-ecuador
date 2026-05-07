interface ComparisonVideoStreamProps {
  model:      "YOLO" | "RT-DETR"
  frameSrc:   string | null
  progress:   number
  status:     string
  loading:    boolean
  color:      string
}

export function ComparisonVideoStream({
  model,
  frameSrc,
  progress,
  status,
  loading,
  color,
}: ComparisonVideoStreamProps) {
  if (!loading && !frameSrc) return null

  return (
    <div className="comparison-stream-panel">
      <div className="comparison-stream-header">
        <div className="comparison-model-badge" style={{ borderColor: color, color }}>
          {model}
        </div>
        <div className="comparison-stream-status">
          {progress > 0 && progress < 100 && (
            <span className="comparison-progress-text">{progress}%</span>
          )}
        </div>
      </div>

      {/* Barra de progreso */}
      <div
        className="comparison-progress-bar-wrap"
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="comparison-progress-bar"
          style={{
            width: `${progress}%`,
            background: progress === 100
              ? "linear-gradient(90deg, #00FF88, #22D3EE)"
              : `linear-gradient(90deg, ${color}, ${color}dd)`,
          }}
        />
      </div>

      {/* Estado */}
      {status && (
        <p className="comparison-stream-status-text" aria-live="polite">
          {status}
        </p>
      )}

      {/* Frame o spinner */}
      {frameSrc ? (
        <div className="comparison-frame-container">
          <img
            src={frameSrc}
            alt={`${model} frame procesado`}
            className="comparison-frame-image"
          />
        </div>
      ) : (
        <div className="comparison-frame-loading">
          <div className="comparison-loading-dots">
            <span /><span /><span />
          </div>
          <span className="comparison-loading-text">Iniciando stream...</span>
        </div>
      )}
    </div>
  )
}