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

interface VehicleInfo {
  type:       string
  type_es:    string
  bbox:       [number, number, number, number]
  confidence: number
}

interface PlateResult {
  bbox:             [number, number, number, number]
  yolo_confidence:  number
  plate:            string
  ocr_confidence:   number
  labels:           PlateLabels
  vehicle:          VehicleInfo | null
}

interface ApiResponse {
  total:    number
  vehicles: number
  plates:   PlateResult[]
  processing_time_ms?: number
  video_metrics?: {
    total_unique_vehicles: number
    total_raw_detections:  number
    frames_processed:      number
    video_duration_s:      number
    processing_time_ms:    number
    vehicles_per_minute:   number
    by_type:               { type: string; count: number; percent: number }[]
  }
}

interface DetectionReport {
  id:             number
  filename:       string
  location:       string
  vehicleType:    string
  confidence:     number
  dateTime:       string
  processingTime: number
  processed:      boolean
  coordinates:    string
}

interface VideoTypeMetric {
  type:    string
  count:   number
  percent: number
}

// ── Canvas — dibuja bboxes de vehículos y placas ──────────────────────────────
function drawBoxes(canvas: HTMLCanvasElement, img: HTMLImageElement, plates: PlateResult[]) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  plates.forEach((plate) => {
    if (plate.vehicle) {
      const [vx1, vy1, vx2, vy2] = plate.vehicle.bbox
      ctx.strokeStyle = "#3b82f6"
      ctx.lineWidth   = Math.max(2, canvas.width * 0.003)
      ctx.setLineDash([8, 4])
      ctx.strokeRect(vx1, vy1, vx2 - vx1, vy2 - vy1)
      ctx.setLineDash([])

      const vLabel = `${plate.vehicle.type_es}`
      const vFont  = Math.max(14, canvas.width * 0.016)
      ctx.font      = `bold ${vFont}px sans-serif`
      const vTextW  = ctx.measureText(vLabel).width
      const vPadX   = vFont * 0.5
      const vPadY   = vFont * 0.4
      const vTagH   = vFont + vPadY * 2
      ctx.fillStyle = "#3b82f6"
      ctx.fillRect(vx1, vy1 - vTagH, vTextW + vPadX * 2, vTagH)
      ctx.fillStyle = "#ffffff"
      ctx.fillText(vLabel, vx1 + vPadX, vy1 - vPadY)
    }

    const [x1, y1, x2, y2] = plate.bbox
    ctx.strokeStyle = "#22d3ee"
    ctx.lineWidth   = Math.max(2, canvas.width * 0.003)
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    const fontSize = Math.max(12, canvas.width * 0.016)
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

// ── Helpers ────────────────────────────────────────────────────────────────────
const LABEL_STYLES: Record<string, { bg: string; color: string }> = {
  "Legible":  { bg: "rgba(16,185,129,0.12)", color: "#10b981" },
  "Ilegible": { bg: "rgba(239,68,68,0.12)",  color: "#ef4444" },
  "No":       { bg: "rgba(16,185,129,0.12)", color: "#10b981" },
  "Parcial":  { bg: "rgba(245,158,11,0.12)", color: "#f59e0b" },
  "Severa":   { bg: "rgba(239,68,68,0.12)",  color: "#ef4444" },
  "Sí":       { bg: "rgba(245,158,11,0.12)", color: "#f59e0b" },
}

function LabelBadge({ name, value }: { name: string; value: string }) {
  const style = LABEL_STYLES[value] ?? { bg: "rgba(255,255,255,0.06)", color: "#94a3b8" }
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 3,
      padding: "0.45rem 0.75rem", borderRadius: 8,
      background: style.bg, border: `1px solid ${style.color}30`, minWidth: 80,
    }}>
      <span style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>
        {name}
      </span>
      <span style={{ fontSize: 12, color: style.color, fontWeight: 700 }}>{value}</span>
    </div>
  )
}

// ── Extraer frames de video para previsualización ─────────────────────────────
async function extractVideoFrames(videoFile: File, maxFrames = 10): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const video       = document.createElement("video")
    video.src         = URL.createObjectURL(videoFile)
    video.muted       = true
    video.playsInline = true

    video.onloadedmetadata = () => {
      const duration = video.duration
      const interval = duration / maxFrames
      const frames: string[] = []
      const canvas = document.createElement("canvas")
      const ctx    = canvas.getContext("2d")
      let currentTime = 0
      let count       = 0

      const captureFrame = () => {
        if (count >= maxFrames || currentTime >= duration) {
          URL.revokeObjectURL(video.src)
          resolve(frames)
          return
        }
        video.currentTime = currentTime
      }

      video.onseeked = () => {
        if (!ctx) return
        canvas.width  = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        frames.push(canvas.toDataURL("image/jpeg", 0.85))
        count++
        currentTime += interval
        captureFrame()
      }

      video.onerror = () => { URL.revokeObjectURL(video.src); reject(new Error("Error al cargar el video")) }
      captureFrame()
    }

    video.onerror = () => { URL.revokeObjectURL(video.src); reject(new Error("Error al cargar el video")) }
  })
}

// ── Componente Principal ───────────────────────────────────────────────────────
function ReadPlate() {
  const [file,         setFile]         = useState<File | null>(null)
  const [fileType,     setFileType]     = useState<"image" | "video" | null>(null)
  const [preview,      setPreview]      = useState<string | null>(null)
  const [result,       setResult]       = useState<ApiResponse | null>(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState<string | null>(null)
  const [report,       setReport]       = useState<DetectionReport[]>([])
  const [showReport,   setShowReport]   = useState(false)
  const [videoFrames,  setVideoFrames]  = useState<string[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)

  // ── WebSocket streaming ────────────────────────────────────────────────────
  const [wsFrameSrc, setWsFrameSrc] = useState<string | null>(null)
  const [wsProgress, setWsProgress] = useState(0)
  const [wsStatus,   setWsStatus]   = useState<string>("")
  const wsRef = useRef<WebSocket | null>(null)

  // ── MediaRecorder — grabación del video anotado ────────────────────────────
  const [downloadUrl,  setDownloadUrl]  = useState<string | null>(null)
  const [isRecording,  setIsRecording]  = useState(false)
  const mediaRecorderRef   = useRef<MediaRecorder | null>(null)
  const recordedChunksRef  = useRef<Blob[]>([])
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef          = useRef<MediaStream | null>(null)

  const inputRef  = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef(new Image())
  const videoRef  = useRef<HTMLVideoElement>(null)

  // Dibuja bboxes cuando llega resultado de imagen
  useEffect(() => {
    if (!result || !preview || !canvasRef.current) return
    if (fileType === "image") {
      const img = imgRef.current
      img.onload = () => { if (canvasRef.current) drawBoxes(canvasRef.current, img, result.plates) }
      img.src = preview
    }
  }, [result, preview, fileType])

  // ── Iniciar MediaRecorder con el primer frame ──────────────────────────────
  const initRecorder = (width: number, height: number) => {
    if (offscreenCanvasRef.current) return // ya iniciado

    const canvas    = document.createElement("canvas")
    canvas.width    = width
    canvas.height   = height
    offscreenCanvasRef.current = canvas

    const stream = canvas.captureStream(15) // 15 fps de salida
    streamRef.current = stream

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm;codecs=vp8"

    const recorder = new MediaRecorder(stream, { mimeType })
    recordedChunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: "video/webm" })
      const url  = URL.createObjectURL(blob)
      setDownloadUrl(url)
      setIsRecording(false)
    }

    recorder.start(100) // chunk cada 100ms
    mediaRecorderRef.current = recorder
    setIsRecording(true)
  }

  // ── Dibujar frame en canvas offscreen (alimenta MediaRecorder) ─────────────
  const paintFrameToCanvas = (imgEl: HTMLImageElement) => {
    const canvas = offscreenCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.drawImage(imgEl, 0, 0)
  }

  // ── Detener grabación ──────────────────────────────────────────────────────
  const stopRecorder = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop()
    }
    offscreenCanvasRef.current = null
    streamRef.current          = null
  }

  // ── Selección de archivo ───────────────────────────────────────────────────
  const handleFile = async (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
    setReport([])
    setVideoFrames([])
    setWsFrameSrc(null)
    setWsProgress(0)
    setWsStatus("")
    setDownloadUrl(null)

    const isVideo = f.type.startsWith("video/")
    const isImage = f.type === "image/jpeg" || f.type === "image/png"

    if (preview && fileType === "video") URL.revokeObjectURL(preview)

    if (isImage) {
      setFileType("image")
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target?.result as string)
      reader.readAsDataURL(f)
    } else if (isVideo) {
      setFileType("video")
      const videoUrl = URL.createObjectURL(f)
      setPreview(videoUrl)
      try {
        const frames = await extractVideoFrames(f, 8)
        setVideoFrames(frames)
      } catch (e) {
        console.error("Error extrayendo frames:", e)
      }
    } else {
      setError("Formato no soportado. Usa JPG, PNG o MP4.")
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }

  // ── Upload / Analyze ───────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setWsFrameSrc(null)
    setWsProgress(0)
    setDownloadUrl(null)
    recordedChunksRef.current  = []
    offscreenCanvasRef.current = null
    const startTime = performance.now()

    // ── IMAGEN: HTTP normal ────────────────────────────────────────────────
    if (fileType === "image") {
      try {
        const formData = new FormData()
        formData.append("file", file)
        const res = await API.post<ApiResponse>("/detect-plate", formData)
        const processingTime = (performance.now() - startTime) / 1000
        setResult(res.data)
        generateReport(res.data, processingTime)
      } catch (e: any) {
        setError(e?.response?.data?.detail || "Error al conectar con el servidor.")
      } finally {
        setLoading(false)
      }
      return
    }

    // ── VIDEO: WebSocket streaming ─────────────────────────────────────────
    try {
      const WS_URL = "ws://localhost:8000/ws/detect-vehicle/video"
      const ws     = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = async () => {
        setWsStatus("Enviando video al servidor...")
        const buffer = await file.arrayBuffer()
        ws.send(buffer)
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        // ── Status ──────────────────────────────────────────────────────
        if (data.type === "status") {
          setWsStatus(data.message)
        }

        // ── Frame ───────────────────────────────────────────────────────
        if (data.type === "frame") {
          const src = `data:image/jpeg;base64,${data.frame}`

          // Actualizar <img> de preview en tiempo real
          setWsFrameSrc(src)
          setWsProgress(data.progress ?? 0)
          setWsStatus(`Procesando frame ${data.frame_num}...`)

          // Alimentar MediaRecorder con cada frame recibido
          const imgEl  = new Image()
          imgEl.onload = () => {
            // Inicializar grabación con las dimensiones del primer frame
            if (!offscreenCanvasRef.current) {
              initRecorder(imgEl.naturalWidth, imgEl.naturalHeight)
            }
            paintFrameToCanvas(imgEl)
          }
          imgEl.src = src

          // Actualizar panel de resultados en tiempo real
          const counter       = data.vehicle_counter as Record<string, number>
          const totalVehicles = Object.values(counter).reduce((a, b) => a + b, 0)
          setResult({
            total:    0,
            vehicles: totalVehicles,
            plates:   [],
            video_metrics: {
              total_unique_vehicles: totalVehicles,
              total_raw_detections:  data.frame_num,
              frames_processed:      data.frame_num,
              video_duration_s:      0,
              processing_time_ms:    0,
              vehicles_per_minute:   0,
              by_type: Object.entries(counter).map(([type, count]) => ({
                type,
                count,
                percent: totalVehicles > 0 ? Math.round((count / totalVehicles) * 100) : 0,
              })),
            },
          })
        }

        // ── Done ────────────────────────────────────────────────────────
        if (data.type === "done") {
          const processingTime = (performance.now() - startTime) / 1000
          const metrics        = data.metrics

          // Detener grabación → dispara onstop → genera blob descargable
          stopRecorder()

          const finalResult: ApiResponse = {
            total:              0,
            vehicles:           metrics.total_unique_vehicles,
            plates:             [],
            processing_time_ms: metrics.processing_time_ms,
            video_metrics: {
              total_unique_vehicles: metrics.total_unique_vehicles,
              total_raw_detections:  metrics.total_raw_detections,
              frames_processed:      metrics.total_raw_detections,
              video_duration_s:      metrics.video_duration_s,
              processing_time_ms:    metrics.processing_time_ms,
              vehicles_per_minute:   metrics.vehicles_per_minute,
              by_type:               metrics.by_type,
            },
          }

          setResult(finalResult)
          generateVideoReport(metrics.by_type, processingTime)
          setWsProgress(100)
          setWsStatus("✓ Procesamiento completo")
          setLoading(false)
          ws.close()
        }

        // ── Error ───────────────────────────────────────────────────────
        if (data.type === "error") {
          setError(data.message)
          stopRecorder()
          setLoading(false)
          setWsStatus("")
          ws.close()
        }
      }

      ws.onerror = () => {
        setError("Error en la conexión WebSocket. ¿Está el servidor corriendo?")
        stopRecorder()
        setLoading(false)
        setWsStatus("")
      }

      ws.onclose = () => {
        if (loading) setLoading(false)
      }

    } catch (e: any) {
      setError("Error al iniciar el análisis de video.")
      stopRecorder()
      setLoading(false)
    }
  }

  // ── Cancelar WebSocket activo ──────────────────────────────────────────────
  const cancelWs = () => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
    stopRecorder()
    setLoading(false)
    setWsStatus("Cancelado")
  }

  // ── Reportes ───────────────────────────────────────────────────────────────
  const generateVideoReport = (videoTypes: VideoTypeMetric[], processingTime: number) => {
    const now      = new Date()
    const dateTime = now.toLocaleString("es-EC", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    })
    const newReports: DetectionReport[] = videoTypes.map((item, i) => ({
      id:             report.length + i + 1,
      filename:       file?.name || "video",
      location:       "Video",
      vehicleType:    item.type,
      confidence:     Math.round(item.percent),
      dateTime,
      processingTime: Number(processingTime.toFixed(2)),
      processed:      true,
      coordinates:    `${item.count} detecciones`,
    }))
    setReport(newReports)
  }

  const generateReport = (data: ApiResponse, processingTime: number) => {
    const now      = new Date()
    const dateTime = now.toLocaleString("es-EC", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    })
    const newReports: DetectionReport[] = data.plates.map((plate, i) => ({
      id:             report.length + i + 1,
      filename:       file?.name || "desconocido",
      location:       "Pendiente",
      vehicleType:    plate.vehicle?.type_es || "Desconocido",
      confidence:     Math.round(plate.ocr_confidence * 100),
      dateTime,
      processingTime: Number(processingTime.toFixed(2)),
      processed:      true,
      coordinates:    `[${plate.bbox.join(", ")}]`,
    }))
    setReport(prev => [...prev, ...newReports])
  }

  // ── Reset ──────────────────────────────────────────────────────────────────
  const reset = () => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
    stopRecorder()
    if (downloadUrl) URL.revokeObjectURL(downloadUrl)
    if (preview && fileType === "video") URL.revokeObjectURL(preview)

    setFile(null)
    setFileType(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setReport([])
    setVideoFrames([])
    setCurrentFrame(0)
    setWsFrameSrc(null)
    setWsProgress(0)
    setWsStatus("")
    setDownloadUrl(null)
    setIsRecording(false)
    recordedChunksRef.current = []
    setLoading(false)
  }

  const downloadCanvas = () => {
    if (!canvasRef.current) return
    const a    = document.createElement("a")
    a.download = "deteccion-vehicular.png"
    a.href     = canvasRef.current.toDataURL("image/png")
    a.click()
  }

  const downloadReportCSV = () => {
    if (report.length === 0) return
    const headers = ["ID","Nombre de Archivo","Ubicación","Tipo de Vehículo","Confianza (%)","Fecha/Hora","Tiempo de Procesamiento (s)","Procesado","Coordenadas"]
    const rows    = report.map(r => [
      r.id, r.filename, r.location, r.vehicleType, r.confidence,
      r.dateTime, r.processingTime, r.processed ? "Sí" : "No", r.coordinates,
    ])
    const csv  = [headers.join(","), ...rows.map(r => r.join(","))].join("\n")
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement("a")
    a.href     = url
    a.download = `reporte-deteccion-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Render ─────────────────────────────────────────────────────────────────
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

          {/* Dropzone */}
          <div
            className={`rp-dropzone ${preview ? "rp-dropzone--has-image" : ""}`}
            onClick={() => !preview && inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {preview ? (
              <>
                {fileType === "video" ? (
                  <video
                    ref={videoRef}
                    src={preview}
                    controls
                    className="rp-preview"
                    style={{ maxHeight: 400 }}
                  />
                ) : (
                  <img src={preview} alt="preview" className="rp-preview" />
                )}
                <button className="rp-clear" onClick={(e) => { e.stopPropagation(); reset() }}>✕</button>
              </>
            ) : (
              <div className="rp-drop-hint">
                <div className="rp-drop-icon">⊕</div>
                <p>Arrastra imagen o video aquí</p>
                <span>o haz clic para seleccionar · JPG / PNG / MP4</span>
              </div>
            )}
          </div>

          {/* Frames extraídos del video */}
          {fileType === "video" && videoFrames.length > 0 && (
            <div className="rp-video-frames">
              <div className="rp-section-label">// FRAMES EXTRAÍDOS</div>
              <div style={{ display: "flex", gap: 8, overflowX: "auto", padding: "0.5rem 0" }}>
                {videoFrames.map((frame, i) => (
                  <img
                    key={i}
                    src={frame}
                    alt={`frame-${i}`}
                    style={{
                      width: 80, height: 60, objectFit: "cover", borderRadius: 4,
                      border: i === currentFrame ? "2px solid #22d3ee" : "2px solid transparent",
                      cursor: "pointer",
                    }}
                    onClick={() => setCurrentFrame(i)}
                  />
                ))}
              </div>
            </div>
          )}

          <input
            ref={inputRef} type="file"
            accept="image/jpeg,image/png,video/mp4,video/avi,video/quicktime"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
          />

          {file && (
            <div className="rp-file-info">
              <span className="rp-file-name">{file.name}</span>
              <span className="rp-file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              <span className="rp-file-type">{fileType === "video" ? "🎬 Video" : "🖼️ Imagen"}</span>
            </div>
          )}

          {/* Botón analizar */}
          <button
            className="rp-btn-analyze"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading
              ? <><span className="rp-spinner" /> Procesando {fileType === "video" ? "video" : "imagen"}...</>
              : <>Analizar <span>→</span></>
            }
          </button>

          {/* Botón cancelar — solo durante streaming de video */}
          {loading && fileType === "video" && (
            <button
              onClick={cancelWs}
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

          {error && <div className="rp-error">{error}</div>}

          {/* ── Detección visual IMAGEN ── */}
          {result && result.total > 0 && fileType === "image" && (
            <div className="rp-canvas-wrap">
              <div className="rp-section-label">DETECCIÓN VISUAL</div>
              <canvas ref={canvasRef} className="rp-canvas" />
              <button className="rp-btn-download" onClick={downloadCanvas}>
                ⬇ Descargar imagen
              </button>
            </div>
          )}

          {/* ── Streaming en tiempo real (VIDEO) ── */}
          {fileType === "video" && (wsFrameSrc || loading) && (
            <div className="rp-canvas-wrap" style={{ marginTop: "1rem" }}>

              {/* Label + indicador REC */}
              <div className="rp-section-label" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                DETECCIÓN VISUAL — EN VIVO
                {isRecording && (
                  <span style={{
                    display: "inline-flex", alignItems: "center", gap: "0.3rem",
                    fontSize: "0.65rem", color: "#ef4444",
                    fontFamily: "DM Mono, monospace",
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: "#ef4444",
                      animation: "rp-pulse 1s ease-in-out infinite",
                    }} />
                    REC
                  </span>
                )}
              </div>

              {/* Barra de progreso */}
              <div style={{
                width: "100%", height: 3, background: "#1e293b",
                borderRadius: 2, margin: "0.5rem 0", overflow: "hidden",
              }}>
                <div style={{
                  height: "100%", width: `${wsProgress}%`,
                  background: "linear-gradient(90deg, #22d3ee, #3b82f6)",
                  transition: "width 0.2s ease", borderRadius: 2,
                }} />
              </div>

              {/* Status text */}
              {wsStatus && (
                <span style={{
                  fontFamily: "DM Mono, monospace", fontSize: "0.68rem",
                  color: wsProgress === 100 ? "#10b981" : "#475569",
                  display: "block", marginBottom: "0.5rem",
                }}>
                  {wsStatus}{wsProgress > 0 && wsProgress < 100 ? ` · ${wsProgress}%` : ""}
                </span>
              )}

              {/* Frame en tiempo real */}
              {wsFrameSrc ? (
                <img
                  src={wsFrameSrc}
                  alt="frame en vivo"
                  style={{ width: "100%", borderRadius: 8, border: "1px solid #1e293b", display: "block" }}
                />
              ) : (
                <div style={{
                  width: "100%", minHeight: 180, borderRadius: 8,
                  background: "#0e1420", border: "1px solid #1c2a3a",
                  display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: 12,
                }}>
                  <div className="rp-loading-dots"><span /><span /><span /></div>
                  <span style={{ fontFamily: "DM Mono, monospace", fontSize: "0.68rem", color: "#2a3f55" }}>
                    Iniciando stream...
                  </span>
                </div>
              )}

              {/* Generando video — transición entre fin de proceso y blob listo */}
              {!isRecording && wsProgress === 100 && !downloadUrl && (
                <div style={{
                  marginTop: "0.75rem", padding: "0.5rem", borderRadius: 8,
                  background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)",
                  fontFamily: "DM Mono, monospace", fontSize: "0.7rem",
                  color: "#f59e0b", textAlign: "center",
                }}>
                  ⏳ Generando archivo de video...
                </div>
              )}

              {/* Botón descarga — aparece cuando MediaRecorder termina de escribir el blob */}
              {downloadUrl && !isRecording && (
                <a
                  href={downloadUrl}
                  download={`deteccion-${file?.name?.replace(/\.[^.]+$/, "") ?? "video"}.webm`}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "center",
                    gap: "0.4rem", marginTop: "0.75rem",
                    padding: "0.55rem 1rem", borderRadius: 8,
                    background: "rgba(34,211,238,0.08)",
                    border: "1px solid rgba(34,211,238,0.25)",
                    color: "#22d3ee", textDecoration: "none",
                    fontFamily: "DM Mono, monospace", fontSize: "0.75rem",
                    transition: "background 0.2s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(34,211,238,0.16)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(34,211,238,0.08)")}
                >
                  ⬇ Descargar video anotado (.webm)
                </a>
              )}
            </div>
          )}

          {/* Reporte Toggle */}
          {report.length > 0 && (
            <button
              className="rp-btn-report"
              onClick={() => setShowReport(!showReport)}
              style={{ marginTop: "1rem" }}
            >
              {showReport ? "Ocultar" : "Ver"} Reporte de Detección
            </button>
          )}
        </div>

        {/* ══ COLUMNA DERECHA ══ */}
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
              {/* Resumen */}
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

              {/* Tarjetas por tipo de vehículo (video) */}
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

              {/* Sin placas en imagen */}
              {result.total === 0 && fileType === "image" && (
                <div className="rp-no-plates">
                  <span>⚠</span> No se detectaron placas. Intenta con una foto más clara o cercana.
                </div>
              )}

              {/* Tarjetas de placas (imagen) */}
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
      </div>

      {/* ══ REPORTE DE DETECCIÓN ══ */}
      {showReport && report.length > 0 && (
        <div className="rp-report-section">
          <div className="rp-section-label">REPORTE DE DETECCIÓN</div>
          <div className="rp-report-header">
            <h3>Detección Actual</h3>
            <button className="rp-btn-download-csv" onClick={downloadReportCSV}>
              ⬇ Descargar CSV
            </button>
          </div>

          <div className="rp-report-table-wrap">
            <table className="rp-report-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Nombre de Archivo</th>
                  <th>Ubicación</th>
                  <th>Tipo de Vehículo</th>
                  <th>Confianza (%)</th>
                  <th>Fecha/Hora</th>
                  <th>Tiempo Proc. (s)</th>
                  <th>Procesado</th>
                  <th>Coordenadas</th>
                </tr>
              </thead>
              <tbody>
                {report.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.filename}</td>
                    <td>{r.location}</td>
                    <td>{r.vehicleType}</td>
                    <td>
                      <span className={`rp-conf-badge ${r.confidence > 70 ? "high" : r.confidence > 40 ? "mid" : "low"}`}>
                        {r.confidence}%
                      </span>
                    </td>
                    <td>{r.dateTime}</td>
                    <td>{r.processingTime}s</td>
                    <td>{r.processed ? "✅" : "❌"}</td>
                    <td><code>{r.coordinates}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default ReadPlate