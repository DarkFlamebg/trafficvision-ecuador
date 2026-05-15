import "./ModelComparison.css"
import { useModelComparison } from "../hooks/useModelComparison"
import { MetricsCard } from "../components/MetricsCards"
import { ComparisonVideoStream } from "../components/ComparisonVideoStream"
import { OCRPanel } from "../components/OCRPanel"
import { ValidationTable } from "../components/ValidationTable"
import { BoundingBoxImage } from "../components/BoundingBoxImage"
import type { ComparisonMetrics, ComparisonImageResponse } from "../types/comparison_types"

function ModelComparison() {
  const mc = useModelComparison()

  const yoloData   = mc.comparisonResult.yolo   as ComparisonImageResponse | undefined
  const rtdetrData = mc.comparisonResult.rtdetr as ComparisonImageResponse | undefined

  const yoloMetrics   = yoloData   ? ("metrics" in yoloData   ? yoloData.metrics   : yoloData)   : null
  const rtdetrMetrics = rtdetrData ? ("metrics" in rtdetrData ? rtdetrData.metrics : rtdetrData) : null

  const yoloPlate   = yoloData?.plates?.[0]   || null
  const rtdetrPlate = rtdetrData?.plates?.[0] || null

  const yoloImage   = yoloData?.processed_image   || (mc.fileType === "image" ? mc.preview : mc.yoloFrame)
  const rtdetrImage = rtdetrData?.processed_image || (mc.fileType === "image" ? mc.preview : mc.rtdetrFrame)

  const hasResults      = !!(yoloMetrics || rtdetrMetrics)
  const hasVideoActivity = mc.fileType === "video" && (mc.loading || mc.yoloFrame || mc.rtdetrFrame)

  return (
    <div className="comparison-page">
      <div className="comparison-grid-bg" aria-hidden="true" />
      <div className="comparison-scan-line"  aria-hidden="true" />

      {/* ── HEADER ── */}
      <header className="comparison-header" role="banner">
        <a href="/" className="comparison-back" aria-label="Volver al inicio">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M9 2.5L4.5 7L9 11.5" stroke="currentColor" strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Inicio
        </a>

        <div className="comparison-title-wrap">
          <div className="comparison-logo-row">
            <div className="comparison-logo-icon" aria-hidden="true">
              <div className="comparison-logo-ring" />
              <div className="comparison-logo-dot"  />
            </div>
            <h1 className="comparison-title">COMPARACIÓN DE MODELOS</h1>
          </div>
          <p className="comparison-tag">YOLOv11n vs RT-DETR · Análisis comparativo</p>
        </div>

        {mc.loading && (
          <div className="comparison-progress-indicator" role="status" aria-live="polite">
            <div className="comparison-loading-spinner" aria-hidden="true" />
            <span>Procesando con ambos modelos...</span>
          </div>
        )}
      </header>

      {/* ── TOOLBAR ROW ── */}
      <div className="comparison-toolbar" role="toolbar" aria-label="Acciones de comparación">
        {/* Upload trigger */}
        <button
          className="comparison-upload-btn"
          onClick={() => mc.inputRef.current?.click()}
          disabled={mc.loading}
          aria-label="Cargar imagen o video"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 9.5V2M7 2L4 5M7 2L10 5" stroke="currentColor"
              strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M1 10.5V12.5H13V10.5" stroke="currentColor"
              strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          CARGAR IMAGEN
        </button>

        <input
          ref={mc.inputRef}
          type="file"
          accept="image/jpeg,image/png,video/mp4"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) mc.handleFile(f)
          }}
          aria-hidden="true"
        />

        {/* Model tag pills */}
        <div className="comparison-model-tags" aria-label="Modelos activos">
          <span className="comparison-model-tag comparison-model-tag--yolo">YOLO</span>
          <span className="comparison-model-tag comparison-model-tag--rtdetr">RTDETR</span>
        </div>

        {/* File info */}
        {mc.file && (
          <span style={{
            fontSize: "0.75rem", color: "var(--text-mid)",
            fontFamily: "var(--font-mono)", marginLeft: "0.25rem",
          }}>
            {mc.file.name} · {(mc.file.size / 1024 / 1024).toFixed(2)} MB
          </span>
        )}

        {/* Right: actions */}
        <div className="comparison-toolbar-right">
          {mc.error && (
            <div className="comparison-error" role="alert">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1.2" />
                <path d="M6 3.5V6.5M6 8V8.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
              {mc.error}
            </div>
          )}

          {mc.loading && mc.fileType === "video" && (
            <button className="comparison-btn-cancel" onClick={mc.cancelComparison}
              aria-label="Cancelar procesamiento">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 2L10 10M10 2L2 10" stroke="currentColor"
                  strokeWidth="1.4" strokeLinecap="round" />
              </svg>
              Cancelar
            </button>
          )}

          {mc.file && !mc.loading && (
            <button
              className="comparison-btn-cancel"
              onClick={mc.reset}
              aria-label="Eliminar archivo y reiniciar"
              style={{ fontSize: "0.75rem" }}
            >
              Limpiar
            </button>
          )}

          <button
            className="comparison-btn-analyze"
            onClick={mc.fileType === "image" ? mc.runImageComparison : mc.runVideoComparison}
            disabled={!mc.file || mc.loading}
            aria-busy={mc.loading}
          >
            {mc.loading ? (
              <span className="comparison-btn-inner">
                <span className="comparison-spinner" aria-hidden="true" />
                <span>Comparando...</span>
              </span>
            ) : (
              <span className="comparison-btn-inner">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M2.5 7H11.5M11.5 7L7.5 3M11.5 7L7.5 11"
                    stroke="currentColor" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                PROBAR MODELOS
              </span>
            )}
            <span className="comparison-btn-shine" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* ── MAIN BODY ── */}
      <main className="comparison-body" role="main">

        {/* ═══ LEFT COL — preview ═══ */}
        <aside className="comparison-left-col">
          <section className="comparison-input-section" aria-label="Carga de archivo">
            <div className="comparison-panel-header">
              <span className="comparison-panel-label" aria-hidden="true">01</span>
              <h2 className="comparison-panel-title">ARCHIVO</h2>
            </div>
            <div className="comparison-card">
              <div
                className={`comparison-dropzone ${mc.preview ? "comparison-dropzone--has-file" : ""}`}
                onClick={() => !mc.preview && mc.inputRef.current?.click()}
                onDrop={mc.handleDrop}
                onDragOver={(e) => e.preventDefault()}
                role={!mc.preview ? "button" : undefined}
                tabIndex={!mc.preview ? 0 : undefined}
                aria-label={!mc.preview ? "Área para arrastrar archivo" : undefined}
              >
                {mc.preview ? (
                  <div className="comparison-file-ready">
                    <div className="comparison-file-ready-icon">✓</div>
                    <div className="comparison-file-ready-info">
                      <span className="comparison-file-ready-name">{mc.file?.name}</span>
                      <span className="comparison-file-ready-meta">
                        {mc.fileType === "video" ? "Video" : "Imagen"} ·{" "}
                        {(mc.file!.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                    <button
                      className="comparison-clear"
                      onClick={(e) => { e.stopPropagation(); mc.reset() }}
                      aria-label="Eliminar archivo"
                    >
                      <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                        <path d="M1 1L8 8M8 1L1 8" stroke="currentColor"
                          strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                    </button>
                  </div>
                ) : (
                  <div className="comparison-drop-hint">
                    <div className="comparison-drop-icon">
                      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                        <path d="M11 15V7M11 7L7.5 10.5M11 7L14.5 10.5"
                          stroke="currentColor" strokeWidth="1.4"
                          strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M19 15.5A3.5 3.5 0 0 0 17 8.5h-1.1A6.5 6.5 0 1 0 3.5 14.5"
                          stroke="currentColor" strokeWidth="1.4"
                          strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <p>Arrastra aquí</p>
                    <span>JPG · PNG · MP4</span>
                  </div>
                )}
              </div>
            </div>
          </section>

          {mc.preview && (
            <section className="comparison-preview-section" aria-label="Vista previa">
              <div className="comparison-panel-header">
                <span className="comparison-panel-label" aria-hidden="true">01b</span>
                <h2 className="comparison-panel-title">VISTA PREVIA</h2>
              </div>
              <div className="comparison-preview-card">
                {mc.fileType === "video" ? (
                  <video src={mc.preview} controls className="comparison-preview-media" />
                ) : (
                  <img src={mc.preview} alt="Vista previa del archivo" className="comparison-preview-media" />
                )}
              </div>
            </section>
          )}
        </aside>

        {/* ═══ RIGHT COL — results ═══ */}
        <div className="comparison-right-col">

        {/* Video streams */}
        {hasVideoActivity && (
            <section className="comparison-streams-section" aria-label="Streams de procesamiento">
            <div className="comparison-panel-header">
                <span className="comparison-panel-label" aria-hidden="true">02</span>
                <h2 className="comparison-panel-title">PROCESAMIENTO EN VIVO</h2>
            </div>
            <div className="comparison-streams-grid">
                <ComparisonVideoStream
                model="YOLO" frameSrc={mc.yoloFrame}
                progress={mc.yoloProgress} status={mc.yoloStatus}
                loading={mc.loading} color="#22d3ee"
                telemetry={mc.yoloTelemetry}
                />
                <ComparisonVideoStream
                model="RT-DETR" frameSrc={mc.rtdetrFrame}
                progress={mc.rtdetrProgress} status={mc.rtdetrStatus}
                loading={mc.loading} color="#f59e0b"
                telemetry={mc.rtdetrTelemetry}
                />
            </div>
            </section>
        )}

        {/* Resultados comparativos - UNIFICADO */}
        {hasResults && (
            <section 
            className={mc.fileType === "image" ? "prototype-results-section" : "comparison-results-section"} 
            aria-label="Resultados comparativos"
            >
            <div className="comparison-panel-header">
                <span className="comparison-panel-label" aria-hidden="true">
                {mc.fileType === "video" ? "03" : "02"}
                </span>
                <h2 className="comparison-panel-title">RESULTADOS COMPARATIVOS</h2>
            </div>

            {/* LAYOUT PARA IMÁGENES - Filas con 3 columnas */}
            {mc.fileType === "image" ? (
                <>
                {/* ── YOLO ROW: [BBox] [Metrics] [OCR] ── */}
                <ModelResultRow
                    modelKey="yolo"
                    modelLabel="YOLO"
                    color="#22d3ee"
                    imageSrc={yoloImage}
                    metrics={yoloMetrics as ComparisonMetrics | null}
                    realPlate={mc.realPlate}
                    vehicles={yoloData?.vehicles || []}
                    allPlates={yoloData?.plates || []}
                />

                <div className="prototype-model-divider" />

                {/* ── RT-DETR ROW ── */}
                <ModelResultRow
                    modelKey="rtdetr"
                    modelLabel="RT-DETR"
                    color="#f59e0b"
                    imageSrc={rtdetrImage}
                    metrics={rtdetrMetrics as ComparisonMetrics | null}
                    realPlate={mc.realPlate}
                    vehicles={rtdetrData?.vehicles || []}
                    allPlates={rtdetrData?.plates || []}
                />

                {/* Winner */}
                {yoloMetrics && rtdetrMetrics && (
                    <div className="comparison-winner-section">
                    <div className="comparison-section-label">GANADOR POR VELOCIDAD</div>
                    {renderWinner(yoloMetrics as ComparisonMetrics, rtdetrMetrics as ComparisonMetrics)}
                    </div>
                )}

                {/* Validation table */}
                <ValidationTable
                    results={mc.comparisonResult}
                    realPlate={mc.realPlate}
                />
                </>
            ) : (
                <>
                <div className="comparison-video-integrated">
                  {/* YOLO Column */}
                  <div className="comparison-model-column">
                    <div className="comparison-model-header-integrated" style={{ color: "#22d3ee" }}>
                      YOLOv11n
                    </div>
                    <MetricsCard
                      model="YOLO"
                      metrics={yoloMetrics as ComparisonMetrics | null}
                      color="#22d3ee"
                    />
                    
                    {yoloData?.plates && yoloData.plates.length > 0 && (
                      <div className="comparison-integrated-plates">
                        <div className="comparison-section-label">PLACAS DETECTADAS</div>
                        <div className="video-plates-list">
                          {yoloData.plates.map((plate: any, idx: number) => (
                            <VideoPlateCard key={idx} plate={plate} color="#22d3ee" />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* RT-DETR Column */}
                  <div className="comparison-model-column">
                    <div className="comparison-model-header-integrated" style={{ color: "#f59e0b" }}>
                      RT-DETR
                    </div>
                    <MetricsCard
                      model="RT-DETR"
                      metrics={rtdetrMetrics as ComparisonMetrics | null}
                      color="#f59e0b"
                    />

                    {rtdetrData?.plates && rtdetrData.plates.length > 0 && (
                      <div className="comparison-integrated-plates">
                        <div className="comparison-section-label">PLACAS DETECTADAS</div>
                        <div className="video-plates-list">
                          {rtdetrData.plates.map((plate: any, idx: number) => (
                            <VideoPlateCard key={idx} plate={plate} color="#f59e0b" />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Winner badge at bottom */}
                  {yoloMetrics && rtdetrMetrics && (
                    <div className="comparison-winner-row">
                      {renderWinner(yoloMetrics as ComparisonMetrics, rtdetrMetrics as ComparisonMetrics)}
                    </div>
                  )}
                </div>

                {/* Validation summary — veredicto + resumen simplificado */}
                <ValidationTable
                  results={mc.comparisonResult}
                  realPlate={mc.realPlate}
                />
                </>
              )}
            </section>
        )}

        {/* Empty state */}
        {!hasResults && !hasVideoActivity && (
            <div className="comparison-empty-right">
            <div className="comparison-empty-illustration" aria-hidden="true">
                <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                <rect x="6" y="10" width="44" height="32" rx="4"
                    stroke="currentColor" strokeWidth="1.5" />
                <path d="M6 32L18 20L28 30L38 18L50 28"
                    stroke="currentColor" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="40" cy="20" r="3.5"
                    stroke="currentColor" strokeWidth="1.5" />
                <path d="M20 46H36" stroke="currentColor"
                    strokeWidth="1.5" strokeLinecap="round" />
                </svg>
            </div>
            <p className="comparison-empty-title">Esperando archivo</p>
            <p className="comparison-empty-desc">
                Sube una imagen o video y presiona{" "}
                <strong>PROBAR MODELOS</strong> para ver el análisis comparativo.
            </p>
            </div>
        )}
        </div>
      </main>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   MODEL RESULT ROW — [BoundingBox] [Metrics] [OCR]
   ══════════════════════════════════════════════════════════════ */
interface ModelResultRowProps {
  modelKey:   string
  modelLabel: string
  color:      string
  imageSrc:   string | null | undefined
  metrics:    ComparisonMetrics | null
  realPlate:  string
  vehicles?:  any[]
  allPlates?: any[]
}

function ModelResultRow({
  modelLabel, color, imageSrc, metrics, realPlate, vehicles = [], allPlates = []
}: ModelResultRowProps) {
  return (
    <div className="prototype-model-row">
      {/* Row label */}
      <div className="prototype-model-row-header">
        <span className="comparison-model-tag"
          style={{ color, borderColor: color + "55", background: color + "0f" }}>
          {modelLabel}
        </span>
        <span style={{
          fontSize: "0.68rem", color: "var(--text-lo)",
          fontFamily: "var(--font-mono)", letterSpacing: "0.05em",
        }}>
          {allPlates.length > 0 ? `${allPlates.length} placa${allPlates.length !== 1 ? 's' : ''} detectada${allPlates.length !== 1 ? 's' : ''}` : 'Sin placas'}
        </span>
      </div>

      {/* Col 1 — Bounding box image */}
      <BoundingBoxImage
        src={imageSrc ?? null}
        alt={`${modelLabel} bounding box detection`}
        modelName={modelLabel}
        color={color}
        vehicles={vehicles}
        plates={allPlates}
      />

      {/* Col 2 — Metrics */}
      <MetricsCard
        model={modelLabel as "YOLO" | "RT-DETR"}
        metrics={metrics}
        color={color}
      />

      {/* Col 3 — OCR (todas las placas) */}
      <OCRPanel
        plates={allPlates}
        realPlate={realPlate}
        color={color}
        modelName={modelLabel}
      />
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   WINNER HELPER
   ══════════════════════════════════════════════════════════════ */
function renderWinner(yolo: ComparisonMetrics, rtdetr: ComparisonMetrics) {
  const yoloTime   = yolo.avg_inference_ms   ?? yolo.inference_ms
  const rtdetrTime = rtdetr.avg_inference_ms ?? rtdetr.inference_ms
  if (!yoloTime || !rtdetrTime) return null

  const winner   = yoloTime < rtdetrTime ? "YOLO" : "RT-DETR"
  const color    = winner === "YOLO" ? "#22d3ee" : "#f59e0b"
  const timeDiff = Math.abs(yoloTime - rtdetrTime).toFixed(2)
  const pctDiff  = (
    ((Math.max(yoloTime, rtdetrTime) - Math.min(yoloTime, rtdetrTime)) /
      Math.max(yoloTime, rtdetrTime)) * 100
  ).toFixed(1)

  return (
    <div className="comparison-winner-card">
      <div className="comparison-winner-badge" style={{ borderColor: color, color }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M9 2L11.2 7L16.5 7.3L12.5 11L13.9 16.5L9 13.6L4.1 16.5L5.5 11L1.5 7.3L6.8 7L9 2Z"
            fill={color} stroke={color} strokeWidth="1.3" />
        </svg>
        {winner}
      </div>
      <p className="comparison-winner-text">
        Más rápido por <strong>{timeDiff}ms</strong> ({pctDiff}% mejor)
      </p>
      <div className="comparison-winner-details">
        <div>
          <span className="comparison-detail-label">YOLO</span>
          <span className="comparison-detail-value">{yoloTime.toFixed(2)}ms</span>
        </div>
        <div>
          <span className="comparison-detail-label">RT-DETR</span>
          <span className="comparison-detail-value">{rtdetrTime.toFixed(2)}ms</span>
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   VIDEO PLATE CARD
   ══════════════════════════════════════════════════════════════ */
function VideoPlateCard({ plate, color }: { plate: any; color: string }) {
  return (
    <div className="video-plate-card" style={{ borderColor: color + '33' }}>
      <div className="vpc-image-wrap">
        {plate.image_base64 ? (
          <img src={`data:image/jpeg;base64,${plate.image_base64}`} alt="Placa" className="vpc-image" />
        ) : (
          <div className="vpc-no-image">No Image</div>
        )}
      </div>
      <div className="vpc-body">
        <div className="vpc-plate-text" style={{ color, fontSize: plate.plate ? '0.95rem' : '0.65rem' }}>
          {plate.plate || "No Detectada Correctamente"}
        </div>
        <div className="vpc-metrics">
          <div className="vpc-metric">
            <span>OCR</span>
            <strong>{(plate.ocr_confidence * 100).toFixed(1)}%</strong>
          </div>
          <div className="vpc-metric">
            <span>DET</span>
            <strong>{(plate.detector_confidence * 100).toFixed(1)}%</strong>
          </div>
          <div className="vpc-metric" title="Segundo en el video">
            <span>TIME</span>
            <strong>{plate.timestamp_video?.toFixed(1)}s</strong>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelComparison