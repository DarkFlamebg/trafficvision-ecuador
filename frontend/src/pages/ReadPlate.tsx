import "./ReadPlate.css"
import { useReadPlate }           from "../hooks/useReadPlate"
import { DropZone }               from "../components/DropZone"
import { VideoFrameStrip }        from "../components/VideoFrameStrip"
import { ImageDetectionCanvas }   from "../components/ImageDetectionCanvas"
import { VideoStreamPanel }       from "../components/VideoStreamPanel"
import { ResultsPanel }           from "../components/ResultsPanel"
import { DetectionReportTable }   from "../components/DetectionReportTable"

function ReadPlate() {
  const rp = useReadPlate()

  return (
    <div className="rp-page">

      {/* HEADER */}
      <div className="rp-header">
        <a href="/" className="rp-back">← Volver</a>
        <div className="rp-title-wrap">
          <h1 className="rp-title">Detector <span>Vehicular</span></h1>
          <span className="rp-tag">ANÁLISIS PROFUNDO EN VEHICULOS Y SUS PLACAS UTILIZANDO MODELOS DE INTELIGENCIA ARTIFICIAL ENTRENADOS</span>
        </div>
      </div>

      <div className="rp-body">

        {/* ══ COLUMNA IZQUIERDA ══ */}
        <div className="rp-left">
          <div className="rp-section-label">ARCHIVO DE ENTRADA</div>

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

          {rp.fileType === "video" && (
            <VideoFrameStrip
              frames={rp.videoFrames}
              currentFrame={rp.currentFrame}
              onSelect={rp.setCurrentFrame}
            />
          )}

          {/* Botón analizar */}
          <button
            className="rp-btn-analyze"
            onClick={rp.handleUpload}
            disabled={!rp.file || rp.loading}
          >
            {rp.loading
              ? <><span className="rp-spinner" /> Procesando {rp.fileType === "video" ? "video" : "imagen"}...</>
              : <>Analizar <span>→</span></>
            }
          </button>

          {/* Botón cancelar — solo durante streaming de video */}
          {rp.loading && rp.fileType === "video" && (
            <button
              onClick={rp.cancelWs}
              style={{
                marginTop: "0.5rem", width: "100%", padding: "0.5rem",
                background: "transparent", border: "1px solid #ef4444",
                borderRadius: 8, color: "#ef4444", cursor: "pointer",
                fontFamily: "DM Mono, monospace", fontSize: "0.75rem",
              }}
            >
              ✕ Cancelar procesamiento
            </button>
          )}

          {rp.error && <div className="rp-error">{rp.error}</div>}

          {/* Detección visual IMAGEN */}
          {rp.result && rp.result.total > 0 && rp.fileType === "image" && (
            <ImageDetectionCanvas canvasRef={rp.canvasRef} onDownload={rp.downloadCanvas} />
          )}

          {/* Streaming en tiempo real (VIDEO) */}
          {rp.fileType === "video" && (
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
          )}

          {/* Reporte toggle */}
          {rp.report.length > 0 && (
            <button
              className="rp-btn-report"
              onClick={() => rp.setShowReport(!rp.showReport)}
              style={{ marginTop: "1rem" }}
            >
              {rp.showReport ? "Ocultar" : "Ver"} Reporte de Detección
            </button>
          )}
        </div>

        {/* ══ COLUMNA DERECHA ══ */}
        <ResultsPanel result={rp.result} loading={rp.loading} fileType={rp.fileType} />
      </div>

      {/* REPORTE DE DETECCIÓN */}
      {rp.showReport && (
        <DetectionReportTable report={rp.report} onDownload={rp.downloadReportCSV} />
      )}
    </div>
  )
}

export default ReadPlate
