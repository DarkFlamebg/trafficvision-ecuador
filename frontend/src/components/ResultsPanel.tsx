import { LabelBadge } from "./LabelBadge"
import type { ApiResponse } from "../types/readplate.types"

interface ResultsPanelProps {
  result:   ApiResponse | null
  loading:  boolean
  fileType: "image" | "video" | null
}

export function ResultsPanel({ result, loading, fileType }: ResultsPanelProps) {
  return (
    <div className="rp-right">
      <div className="rp-section-label">RESULTADOS</div>

      {!result && !loading && (
        <div className="rp-empty">
          <div className="rp-empty-icon">◎</div>
          <p>Sube una imagen o video para ver<br />los resultados aquí</p>
        </div>
      )}

      {loading && !result && (
        <div className="rp-empty">
          <div className="rp-loading-dots"><span /><span /><span /></div>
          <p>Procesando {fileType === "video" ? "video frame por frame" : "imagen"}...</p>
        </div>
      )}

      {result && (
        <>
          {/* Summary stats */}
          <div className="rp-summary">
            <div className="rp-summary-stat">
              <span className="rp-summary-num">{result.vehicles}</span>
              <span className="rp-summary-label">
                Vehículo{result.vehicles !== 1 ? "s" : ""} detectado{result.vehicles !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="rp-summary-stat">
              <span className="rp-summary-num">{result.total}</span>
              <span className="rp-summary-label">
                Placa{result.total !== 1 ? "s" : ""} detectada{result.total !== 1 ? "s" : ""}
              </span>
            </div>
            {result.processing_time_ms != null && result.processing_time_ms > 0 && (
              <div className="rp-summary-stat">
                <span className="rp-summary-num">{(result.processing_time_ms / 1000).toFixed(1)}s</span>
                <span className="rp-summary-label">Tiempo servidor</span>
              </div>
            )}
          </div>

          {/* Vehicle type cards (video) */}
          {fileType === "video" && (result.video_metrics?.by_type ?? []).length > 0 && (
            <div className="rp-results-grid">
              {(result.video_metrics?.by_type ?? []).map((item) => (
                <div className="rp-plate-card" key={item.type}>
                  <div className="rp-plate-header">
                    <span className="rp-plate-index">{item.type}</span>
                    <span className="rp-plate-badge rp-badge--high">{item.percent}%</span>
                  </div>
                  <div className="rp-plate-number">{item.count}</div>
                  <div className="rp-bbox">
                    <span className="rp-bbox-label">Vehículos detectados</span>
                    <span className="rp-bbox-val">Participación del total: {item.percent}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No plates warning */}
          {result.total === 0 && fileType === "image" && (
            <div className="rp-no-plates">
              <span>⚠</span> No se detectaron placas. Intenta con una foto más clara o cercana.
            </div>
          )}

          {/* Plate cards (image) */}
          {(result.plates ?? []).length > 0 && (
            <div className="rp-results-grid">
              {(result.plates ?? []).map((plate, i) => (
                <div className="rp-plate-card" key={i}>
                  <div className="rp-plate-header">
                    <span className="rp-plate-index">Vehiculo #{i + 1}</span>
                    <span className={`rp-plate-badge ${plate.ocr_confidence > 0.7 ? "rp-badge--high" : plate.ocr_confidence > 0.4 ? "rp-badge--mid" : "rp-badge--low"}`}>
                      {plate.ocr_confidence > 0.7 ? "Alta confianza" : plate.ocr_confidence > 0.4 ? "Media confianza" : "Baja confianza"}
                    </span>
                  </div>

                  {plate.vehicle && (
                    <div style={{
                      display: "flex", alignItems: "center", gap: "0.5rem",
                      padding: "0.4rem 0.75rem", borderRadius: 8, marginBottom: "0.75rem",
                      background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)",
                    }}>
                      <span style={{ fontSize: 13, color: "#93c5fd", fontWeight: 600 }}>{plate.vehicle.type_es}</span>
                      <span style={{ fontSize: 11, color: "#475569", marginLeft: "auto" }}>
                        {(plate.vehicle.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  <div className="rp-plate-number">{plate.plate}</div>

                  {plate.labels && (
                    <div style={{ marginBottom: "1rem" }}>
                      <div style={{ fontSize: 9, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700, marginBottom: "0.5rem" }}>
                        Calidad de imagen
                      </div>
                      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                        <LabelBadge name="Legibilidad" value={plate.labels.legible}  />
                        <LabelBadge name="Oclusión"   value={plate.labels.oclusion} />
                        <LabelBadge name="Reflejo"    value={plate.labels.reflejo}  />
                        <LabelBadge name="Suciedad"   value={plate.labels.sucia}    />
                      </div>
                    </div>
                  )}

                  <div className="rp-metrics">
                    <div className="rp-metric">
                      <span className="rp-metric-label">Detección YOLO</span>
                      <div className="rp-metric-bar-wrap">
                        <div className="rp-metric-bar" style={{ width: `${plate.yolo_confidence * 100}%` }} />
                      </div>
                      <span className="rp-metric-val">{(plate.yolo_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="rp-metric">
                      <span className="rp-metric-label">Lectura OCR</span>
                      <div className="rp-metric-bar-wrap">
                        <div className="rp-metric-bar rp-metric-bar--ocr" style={{ width: `${plate.ocr_confidence * 100}%` }} />
                      </div>
                      <span className="rp-metric-val">{(plate.ocr_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="rp-bbox">
                    <span className="rp-bbox-label">Bounding box placa</span>
                    <span className="rp-bbox-val">[{plate.bbox.join(", ")}]</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          <details className="rp-json">
            <summary>Ver respuesta JSON completa</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </>
      )}
    </div>
  )
}
