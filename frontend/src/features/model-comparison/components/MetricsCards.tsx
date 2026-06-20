import type { ComparisonMetrics } from "../types/comparison_types"

interface MetricsCardProps {
  model:   "YOLO" | "RT-DETR" | "Vision Mamba"
  metrics: ComparisonMetrics | null
  color:   string
}

export function MetricsCard({ model, metrics, color }: MetricsCardProps) {
  if (!metrics) {
    return (
      <div className="comparison-metrics-card comparison-metrics-card--empty">
        <div className="comparison-model-badge" style={{ borderColor: color, color }}>
          {model}
        </div>
        <div className="comparison-empty-state">
          <div className="comparison-loading-dots">
            <span /><span /><span />
          </div>
          <p>Esperando resultados...</p>
        </div>
      </div>
    )
  }

  const isVideoMetrics = !!metrics.total_unique_vehicles

  return (
    <div className="comparison-metrics-card">
      <div className="comparison-card-header">
        <div className="comparison-model-badge" style={{ borderColor: color, color }}>
          {model}
        </div>
        {metrics.inference_ms !== undefined && (
          <div className="comparison-inference-time">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1.2"/>
              <path d="M6 3V6L8 7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
            </svg>
            <span>{metrics.inference_ms.toFixed(1)}<span className="comparison-stat-unit" style={{ color: 'inherit' }}>ms</span></span>
          </div>
        )}
      </div>

      <div className="comparison-stats-grid">
        {isVideoMetrics ? (
          <>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Vehículos únicos</span>
              <span className="comparison-stat-value" style={{ color }}>
                {metrics.total_unique_vehicles}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Detecciones totales</span>
              <span className="comparison-stat-value">
                {metrics.total_raw_detections}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Placas detectadas</span>
              <span className="comparison-stat-value" style={{ color }}>
                {metrics.total_plates_detected ?? 0}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Frames procesados</span>
              <span className="comparison-stat-value">
                {metrics.frames_processed}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Veh/min</span>
              <span className="comparison-stat-value">
                {metrics.vehicles_per_minute?.toFixed(1)}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Tiempo procesamiento</span>
              <span className="comparison-stat-value">
                {((metrics.processing_time_ms ?? 0) / 1000).toFixed(2)}<span className="comparison-stat-unit">s</span>
              </span>
            </div>
            {metrics.avg_inference_ms !== undefined && (
              <div className="comparison-stat comparison-stat--highlight">
                <span className="comparison-stat-label">Inferencia promedio</span>
                <span className="comparison-stat-value" style={{ color }}>
                  {metrics.avg_inference_ms.toFixed(2)}<span className="comparison-stat-unit">ms</span>
                </span>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Vehículos</span>
              <span className="comparison-stat-value" style={{ color }}>
                {metrics.vehicles_detected}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Placas</span>
              <span className="comparison-stat-value" style={{ color }}>
                {metrics.plates_detected}
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Conf. vehículos</span>
              <span className="comparison-stat-value">
                {(metrics.avg_vehicle_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Conf. placas</span>
              <span className="comparison-stat-value">
                {(metrics.avg_plate_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="comparison-stat">
              <span className="comparison-stat-label">Placas con OCR</span>
              <span className="comparison-stat-value">
                {metrics.plates_with_ocr}
              </span>
            </div>
            <div className="comparison-stat comparison-stat--highlight">
              <span className="comparison-stat-label">Tiempo inferencia</span>
              <span className="comparison-stat-value" style={{ color }}>
                {metrics.inference_ms.toFixed(2)}<span className="comparison-stat-unit">ms</span>
              </span>
            </div>
          </>
        )}
      </div>

      {/* Distribución por tipo de vehículo */}
      {(isVideoMetrics ? metrics.by_type : null) && metrics.by_type!.length > 0 && (
        <div className="comparison-type-distribution">
          <div className="comparison-section-label">DISTRIBUCIÓN POR TIPO</div>
          <div className="comparison-type-bars">
            {metrics.by_type!.map((item) => (
              <div key={item.type} className="comparison-type-bar-row">
                <span className="comparison-type-label">{item.type}</span>
                <div className="comparison-type-bar-wrap">
                  <div
                    className="comparison-type-bar"
                    style={{
                      width: `${item.percent}%`,
                      background: color,
                    }}
                  />
                </div>
                <span className="comparison-type-value">
                  {item.count} ({item.percent}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {!isVideoMetrics && metrics.vehicles_by_type && Object.keys(metrics.vehicles_by_type).length > 0 && (
        <div className="comparison-type-distribution">
          <div className="comparison-section-label">VEHÍCULOS POR TIPO</div>
          <div className="comparison-type-list">
            {Object.entries(metrics.vehicles_by_type).map(([type, count]) => (
              <div key={type} className="comparison-type-item">
                <span className="comparison-type-name">{type}</span>
                <span className="comparison-type-count" style={{ color }}>
                  {count}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}