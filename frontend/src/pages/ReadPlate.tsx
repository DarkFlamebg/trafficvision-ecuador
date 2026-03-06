import { useState, useRef, useEffect } from "react"
import API from "../services/api"
import "./ReadPlate.css"

// ── Types ──────────────────────────────────────────────────────────────────────
interface PlateLabels {
  legible:  "Legible" | "Ilegible"
  oclusion: "No" | "Parcial" | "Severa"
  reflejo:  "No" | "Sí"
  sucia:    "No" | "Sí"
}

interface PlateResult {
  bbox:             [number, number, number, number]
  yolo_confidence:  number
  plate:            string
  ocr_confidence:   number
  labels:           PlateLabels
}

interface ApiResponse {
  total:  number
  plates: PlateResult[]
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function drawBoxes(canvas: HTMLCanvasElement, img: HTMLImageElement, plates: PlateResult[]) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  plates.forEach((plate) => {
    const [x1, y1, x2, y2] = plate.bbox
    ctx.strokeStyle = "#22d3ee"
    ctx.lineWidth   = Math.max(2, canvas.width * 0.003)
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    const fontSize = Math.max(14, canvas.width * 0.018)
    ctx.font = `bold ${fontSize}px monospace`
    const textW = ctx.measureText(plate.plate).width
    const padX  = fontSize * 0.5
    const padY  = fontSize * 0.4
    const tagH  = fontSize + padY * 2

    ctx.fillStyle = "#22d3ee"
    ctx.fillRect(x1, y1 - tagH, textW + padX * 2, tagH)
    ctx.fillStyle = "#080c18"
    ctx.fillText(plate.plate, x1 + padX, y1 - padY)
  })
}

// Colores por valor de etiqueta
const LABEL_STYLES: Record<string, { bg: string; color: string }> = {
  // legible
  "Legible":   { bg: "rgba(16,185,129,0.12)",  color: "#10b981" },
  "Ilegible":  { bg: "rgba(239,68,68,0.12)",   color: "#ef4444" },
  // oclusion
  "No":        { bg: "rgba(16,185,129,0.12)",  color: "#10b981" },
  "Parcial":   { bg: "rgba(245,158,11,0.12)",  color: "#f59e0b" },
  "Severa":    { bg: "rgba(239,68,68,0.12)",   color: "#ef4444" },
  // reflejo / sucia — "Sí"
  "Sí":        { bg: "rgba(245,158,11,0.12)",  color: "#f59e0b" },
}

function LabelBadge({ name, value }: { name: string; value: string }) {
  const style = LABEL_STYLES[value] ?? { bg: "rgba(255,255,255,0.06)", color: "#94a3b8" }
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 3,
      padding: "0.45rem 0.75rem",
      borderRadius: 8,
      background: style.bg,
      border: `1px solid ${style.color}30`,
      minWidth: 80,
    }}>
      <span style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>
        {name}
      </span>
      <span style={{ fontSize: 12, color: style.color, fontWeight: 700 }}>
        {value}
      </span>
    </div>
  )
}

// ── Componente ─────────────────────────────────────────────────────────────────
function ReadPlate() {
  const [file,    setFile]    = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result,  setResult]  = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  const inputRef  = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef(new Image())

  useEffect(() => {
    if (!result || !preview || !canvasRef.current) return
    const img = imgRef.current
    img.onload = () => { if (canvasRef.current) drawBoxes(canvasRef.current, img, result.plates) }
    img.src = preview
  }, [result, preview])

  const handleFile = (f: File) => {
    setFile(f); setResult(null); setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(f)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && (f.type === "image/jpeg" || f.type === "image/png")) handleFile(f)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const res = await API.post<ApiResponse>("/detect-plate", formData)
      setResult(res.data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Error al conectar con el servidor.")
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const downloadCanvas = () => {
    if (!canvasRef.current) return
    const a = document.createElement("a")
    a.download = "placa-detectada.png"
    a.href = canvasRef.current.toDataURL("image/png")
    a.click()
  }

  return (
    <div className="rp-page">

      {/* HEADER */}
      <div className="rp-header">
        <a href="/" className="rp-back">← Volver</a>
        <div className="rp-title-wrap">
          <span className="rp-tag">ANÁLISIS DE PLACAS</span>
          <h1 className="rp-title">Detector<br /><span>Vehicular</span></h1>
        </div>
      </div>

      <div className="rp-body">

        {/* ══ COLUMNA IZQUIERDA ══ */}
        <div className="rp-left">
          <div className="rp-section-label">// IMAGEN DE ENTRADA</div>

          <div
            className={`rp-dropzone ${preview ? "rp-dropzone--has-image" : ""}`}
            onClick={() => !preview && inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {preview ? (
              <>
                <img src={preview} alt="preview" className="rp-preview" />
                <button className="rp-clear" onClick={(e) => { e.stopPropagation(); reset() }}>✕</button>
              </>
            ) : (
              <div className="rp-drop-hint">
                <div className="rp-drop-icon">⊕</div>
                <p>Arrastra una imagen aquí</p>
                <span>o haz clic para seleccionar · JPG / PNG</span>
              </div>
            )}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
          />

          {file && (
            <div className="rp-file-info">
              <span className="rp-file-name">{file.name}</span>
              <span className="rp-file-size">{(file.size / 1024).toFixed(1)} KB</span>
            </div>
          )}

          <button className="rp-btn-analyze" onClick={handleUpload} disabled={!file || loading}>
            {loading ? <><span className="rp-spinner" /> Analizando...</> : <>Analizar placa <span>→</span></>}
          </button>

          {error && <div className="rp-error">{error}</div>}

          {/* Canvas con bboxes */}
          {result && result.total > 0 && (
            <div className="rp-canvas-wrap">
              <div className="rp-section-label">// DETECCIÓN VISUAL</div>
              <canvas ref={canvasRef} className="rp-canvas" />
              <button className="rp-btn-download" onClick={downloadCanvas}>
                ⬇ Descargar imagen
              </button>
            </div>
          )}
        </div>

        {/* ══ COLUMNA DERECHA ══ */}
        <div className="rp-right">
          <div className="rp-section-label">// RESULTADOS</div>

          {!result && !loading && (
            <div className="rp-empty">
              <div className="rp-empty-icon">◎</div>
              <p>Sube una imagen para ver<br />los resultados aquí</p>
            </div>
          )}

          {loading && (
            <div className="rp-empty">
              <div className="rp-loading-dots"><span /><span /><span /></div>
              <p>Procesando imagen...</p>
            </div>
          )}

          {result && (
            <>
              <div className="rp-summary">
                <div className="rp-summary-stat">
                  <span className="rp-summary-num">{result.total}</span>
                  <span className="rp-summary-label">
                    Placa{result.total !== 1 ? "s" : ""} detectada{result.total !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>

              {result.total === 0 && (
                <div className="rp-no-plates">
                  <span>⚠</span> No se detectaron placas. Intenta con una foto más clara o cercana.
                </div>
              )}

              {result.plates.map((plate, i) => (
                <div className="rp-plate-card" key={i}>

                  {/* Header */}
                  <div className="rp-plate-header">
                    <span className="rp-plate-index">#{i + 1}</span>
                    <span className={`rp-plate-badge ${plate.ocr_confidence > 0.7 ? "rp-badge--high" : plate.ocr_confidence > 0.4 ? "rp-badge--mid" : "rp-badge--low"}`}>
                      {plate.ocr_confidence > 0.7 ? "Alta confianza" : plate.ocr_confidence > 0.4 ? "Media confianza" : "Baja confianza"}
                    </span>
                  </div>

                  {/* Número de placa */}
                  <div className="rp-plate-number">{plate.plate}</div>

                  {/* ── Etiquetas de calidad ── */}
                  {plate.labels && (
                    <div style={{ marginBottom: "1rem" }}>
                      <div style={{
                        fontSize: 9, color: "#475569", textTransform: "uppercase",
                        letterSpacing: "0.1em", fontWeight: 700, marginBottom: "0.5rem"
                      }}>
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

                  {/* Métricas */}
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

                  {/* BBox */}
                  <div className="rp-bbox">
                    <span className="rp-bbox-label">Bounding box</span>
                    <span className="rp-bbox-val">[{plate.bbox.join(", ")}]</span>
                  </div>

                </div>
              ))}

              <details className="rp-json">
                <summary>Ver respuesta JSON completa</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </>
          )}
        </div>

      </div>
    </div>
  )
}

export default ReadPlate