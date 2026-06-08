import { Skeleton, SkeletonBlock } from "../../../components/Skeleton"
import { DiagnosticPanel } from "./DiagnosticPanel"
import type { ApiResponse } from "../types/readplate.types"

interface ResultsPanelProps {
  result:   ApiResponse | null
  loading:  boolean
  fileType: "image" | "video" | null
}

export function ResultsPanel({ result, loading, fileType }: ResultsPanelProps) {
  return (
    <div className="rp-right">

      {/* Empty state */}
      {!result && !loading && (
        <div className="rp-empty" role="status">
          <div className="rp-empty-icon" aria-hidden="true">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3"/>
              <circle cx="16" cy="16" r="6" stroke="currentColor" strokeWidth="1" opacity="0.5"/>
              <circle cx="16" cy="16" r="2" fill="currentColor" opacity="0.3"/>
            </svg>
          </div>
          <p>Carga una imagen o video<br />para ver los resultados aquí</p>
        </div>
      )}

      {/* Loading state */}
      {loading && !result && (
        <div className="rp-loading-skeleton" role="status" aria-live="polite">
          <div style={{ marginBottom: "1rem" }}>
            <Skeleton width="40%" height="1.5rem" />
            <Skeleton width="70%" height="0.875rem" style={{ marginTop: "0.5rem" }} />
          </div>
          <div className="rp-results-grid" style={{ marginBottom: "2rem" }}>
            <div className="rp-plate-card">
              <SkeletonBlock lines={4} />
            </div>
            <div className="rp-plate-card">
              <SkeletonBlock lines={4} />
            </div>
          </div>
          <p style={{ color: "var(--text-mid)", fontSize: "0.85rem", textAlign: "center" }}>
            Procesando {fileType === "video" ? "video fotograma a fotograma" : "imagen"}...
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div role="region" aria-label="Resultados de detección">

          {/* Summary stats */}
          <div className="rp-summary" role="list">
            <div className="rp-summary-stat" role="listitem">
              <span className="rp-summary-num" aria-label={`${result.vehicles} vehículos`}>
                {result.vehicles}
              </span>
              <span className="rp-summary-label">
                Vehículo{result.vehicles !== 1 ? "s" : ""} detectado{result.vehicles !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="rp-summary-stat" role="listitem">
              <span className="rp-summary-num" aria-label={`${result.total} placas`}>
                {result.total}
              </span>
              <span className="rp-summary-label">
                Placa{result.total !== 1 ? "s" : ""} detectada{result.total !== 1 ? "s" : ""}
              </span>
            </div>
            {result.processing_time_ms != null && result.processing_time_ms > 0 && (
              <div className="rp-summary-stat" role="listitem">
                <span className="rp-summary-num">
                  {(result.processing_time_ms / 1000).toFixed(1)}s
                </span>
                <span className="rp-summary-label">Tiempo servidor</span>
              </div>
            )}
          </div>

          {/* Video type breakdown */}
          {fileType === "video" && (result.video_metrics?.by_type ?? []).length > 0 && (
            <div className="rp-results-grid" role="list">
              {(result.video_metrics?.by_type ?? []).map((item) => (
                <div className="rp-plate-card" key={item.type} role="listitem">
                  <div className="rp-plate-header">
                    <span className="rp-plate-index">{item.type}</span>
                    <span className="rp-plate-badge rp-badge--high">{item.percent}%</span>
                  </div>
                  <div className="rp-plate-number">{item.count}</div>
                  <div className="rp-bbox">
                    <span className="rp-bbox-label">Detecciones</span>
                    <span className="rp-bbox-val">Participación: {item.percent}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No plates */}
          {result.total === 0 && fileType === "image" && (
            <div className="rp-no-plates" role="alert">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2"/>
                <path d="M7 4.5V7M7 9V9.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              No se detectaron placas. Intenta con una foto más clara o cercana.
            </div>
          )}

          {/* Plate cards */}
          {(result.plates ?? []).length > 0 && (
            <div className="rp-results-grid" role="list">
              {(result.plates ?? []).map((plate, i) => {
                const conf = plate.ocr_confidence
                const badgeClass = conf > 0.7 ? "rp-badge--high" : conf > 0.4 ? "rp-badge--mid" : "rp-badge--low"
                const badgeLabel = conf > 0.7 ? "Alta confianza" : conf > 0.4 ? "Media confianza" : "Baja confianza"

                return (
                  <article className="rp-plate-card" key={i} aria-label={`Vehículo ${i + 1}: placa ${plate.plate}`}>
                    <div className="rp-plate-header">
                      <span className="rp-plate-index">Vehículo #{i + 1}</span>
                      <span className={`rp-plate-badge ${badgeClass}`}>{badgeLabel}</span>
                    </div>

                    {plate.vehicle && (
                      <div style={{
                        display: "flex", alignItems: "center", gap: "0.5rem",
                        padding: "0.4rem 0.75rem", borderRadius: 8, marginBottom: "0.75rem",
                        background: "rgba(59,130,246,0.07)", border: "1px solid rgba(59,130,246,0.18)",
                      }}>
                        <span style={{ fontSize: 13, color: "#93c5fd", fontWeight: 600, fontFamily: "var(--font-body)" }}>
                          {plate.vehicle.type_es}
                        </span>
                        <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                          {(plate.vehicle.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    <div className="rp-plate-number" aria-label={`Número de placa: ${plate.plate}`}>
                      {plate.plate}
                    </div>

                    {plate.labels && (
                      <DiagnosticPanel labels={plate.labels} />
                    )}

                    <div className="rp-metrics" aria-label="Métricas de confianza">
                      <div className="rp-metric">
                        <span className="rp-metric-label">Detec. YOLO</span>
                        <div className="rp-metric-bar-wrap" role="progressbar"
                          aria-valuenow={Math.round(plate.yolo_confidence * 100)}
                          aria-valuemin={0} aria-valuemax={100}>
                          <div className="rp-metric-bar" style={{ width: `${plate.yolo_confidence * 100}%` }} />
                        </div>
                        <span className="rp-metric-val">{(plate.yolo_confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="rp-metric">
                        <span className="rp-metric-label">Lectura OCR</span>
                        <div className="rp-metric-bar-wrap" role="progressbar"
                          aria-valuenow={Math.round(plate.ocr_confidence * 100)}
                          aria-valuemin={0} aria-valuemax={100}>
                          <div className="rp-metric-bar rp-metric-bar--ocr" style={{ width: `${plate.ocr_confidence * 100}%` }} />
                        </div>
                        <span className="rp-metric-val">{(plate.ocr_confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="rp-bbox">
                      <span className="rp-bbox-label">Bounding box placa</span>
                      <span className="rp-bbox-val">[{plate.bbox.join(", ")}]</span>
                    </div>
                  </article>
                )
              })}
            </div>
          )}

          <details className="rp-json">
            <summary>Ver respuesta JSON completa</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}