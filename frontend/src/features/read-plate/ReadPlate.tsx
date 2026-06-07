import "./ReadPlate.css"
import { useReadPlate }           from "./hooks/useReadPlate"
import { ProcessStepper }         from "./components/ProcessStepper"
import { DropZone }               from "./components/DropZone"
import { VideoFrameStrip }        from "./components/VideoFrameStrip"
import { ImageDetectionCanvas }   from "./components/ImageDetectionCanvas"
import { VideoStreamPanel }       from "./components/VideoStreamPanel"
import { ResultsPanel }           from "./components/ResultsPanel"
import { DetectionReportTable }   from "./components/DetectionReportTable"

function ReadPlate() {
  const rp = useReadPlate()

  // Determinar step actual para el stepper
  const getCurrentStep = (): "upload" | "analyze" | "results" => {
    if (rp.result) return "results"
    if (rp.loading) return "analyze"
    return "upload"
  }

  return (
    <div className="rp-page">

      {/* ── GRID LINES BACKGROUND ── */}
      <div className="rp-grid-bg" aria-hidden="true" />
      <div className="rp-scan-line" aria-hidden="true" />

      {/* ── HEADER ── */}
      <header className="rp-header" role="banner">
        <div className="rp-title-wrap">
          <div className="rp-logo-row">
            <div className="rp-logo-icon" aria-hidden="true">
              <div className="rp-logo-ring" />
              <div className="rp-logo-dot" />
            </div>
            <h1 className="rp-title">
              DETECTOR VEHICULAR
            </h1>
          </div>
          <p className="rp-tag">
            Análisis profundo de vehículos y placas mediante modelos de IA entrenados
          </p>
        </div>

        {/* STEPPER - Muestra el progreso del flujo */}
        <ProcessStepper 
          currentStep={getCurrentStep()}
          hasFile={!!rp.file}
          loading={rp.loading}
          hasResults={!!rp.result}
        />
      </header>

      {/* ── MAIN BODY ── */}
      <main className="rp-body" role="main">

        {/* ══ COLUMNA IZQUIERDA ══ */}
        <section className="rp-left" aria-label="Panel de entrada">

          <div className="rp-panel-header">
            <span className="rp-panel-label" aria-hidden="true">01</span>
            <h2 className="rp-panel-title">ARCHIVO DE ENTRADA</h2>
          </div>

          <div className="rp-card">
            <DropZone
              preview={rp.preview}
              fileType={rp.fileType}
              file={rp.file}
              videoRef={rp.videoRef}
              inputRef={rp.inputRef}
              onDrop={rp.handleDrop}
              onFile={rp.handleFile}
              onReset={rp.reset}
            />
          </div>

          {rp.fileType === "video" && (
            <div className="rp-card rp-card--frames">
              <VideoFrameStrip
                frames={rp.videoFrames}
                currentFrame={rp.currentFrame}
                onSelect={rp.setCurrentFrame}
              />
            </div>
          )}

          {/* ── ACCIONES ── */}
          <div className="rp-actions">
            <button
              className="rp-btn-analyze"
              onClick={rp.handleUpload}
              disabled={!rp.file || rp.loading}
              aria-busy={rp.loading}
              aria-label={rp.loading ? "Procesando archivo" : "Iniciar análisis"}
            >
              {rp.loading ? (
                <span className="rp-btn-inner">
                  <span className="rp-spinner" aria-hidden="true" />
                  <span>
                    Procesando {rp.fileType === "video" ? "video" : "imagen"}
                    <span className="rp-ellipsis" aria-hidden="true">...</span>
                  </span>
                </span>
              ) : (
                <span className="rp-btn-inner">
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
                    <path d="M3 9H15M15 9L10 4M15 9L10 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>Iniciar Análisis</span>
                </span>
              )}
              <span className="rp-btn-shine" aria-hidden="true" />
            </button>

            {rp.loading && rp.fileType === "video" && (
              <button
                className="rp-btn-cancel"
                onClick={rp.cancelWs}
                aria-label="Cancelar procesamiento"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M2 2L12 12M12 2L2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                Cancelar procesamiento
              </button>
            )}
          </div>

          {rp.error && (
            <div className="rp-error" role="alert" aria-live="assertive">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2"/>
                <path d="M7 4V7.5M7 9.5V10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              {rp.error}
            </div>
          )}

          {/* ── DETECCIÓN VISUAL IMAGEN ── */}
          {rp.result && rp.result.total > 0 && rp.fileType === "image" && (
            <div className="rp-card rp-card--detection">
              <ImageDetectionCanvas canvasRef={rp.canvasRef} onDownload={rp.downloadCanvas} />
            </div>
          )}

          {/* ── STREAMING VIDEO ── */}
          {rp.fileType === "video" && (
            <div className="rp-card rp-card--stream">
              <VideoStreamPanel
                wsFrameSrc={rp.wsFrameSrc}
                wsProgress={rp.wsProgress}
                wsStatus={rp.wsStatus}
                isRecording={rp.isRecording}
                isConverting={rp.isConverting}
                downloadUrl={rp.downloadUrl}
                loading={rp.loading}
                fileName={rp.file?.name}
              />
            </div>
          )}

          {/* ── REPORTE TOGGLE - Ahora más visible ── */}
          {rp.report.length > 0 && (
            <button
              className="rp-btn-report"
              onClick={() => rp.setShowReport(!rp.showReport)}
              aria-expanded={rp.showReport}
              aria-controls="detection-report"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <rect x="2" y="2" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M5 6H11M5 8H11M5 10H8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <div className="rp-btn-report-content">
                <span className="rp-btn-report-label">
                  {rp.showReport ? "Ocultar" : "Ver"} Reporte de Detección
                </span>
                <span className="rp-report-count">{rp.report.length} registros</span>
              </div>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" 
                style={{ transform: rp.showReport ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>
                <path d="M3 5L7 9L11 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          )}
        </section>

        {/* ══ COLUMNA DERECHA ══ */}
        <section aria-label="Panel de resultados">
          <div className="rp-panel-header">
            <span className="rp-panel-label" aria-hidden="true">02</span>
            <h2 className="rp-panel-title">RESULTADOS</h2>
          </div>
          <ResultsPanel result={rp.result} loading={rp.loading} fileType={rp.fileType} />
        </section>
      </main>

      {/* ── REPORTE DE DETECCIÓN ── */}
      {rp.showReport && (
        <section id="detection-report" className="rp-report-section-wrap" aria-label="Tabla de detecciones">
          <div className="rp-panel-header">
            <span className="rp-panel-label" aria-hidden="true">03</span>
            <h2 className="rp-panel-title">REPORTE DE DETECCIÓN</h2>
          </div>
          <DetectionReportTable report={rp.report} onDownload={rp.downloadReportCSV} />
        </section>
      )}
    </div>
  )
}

export default ReadPlate