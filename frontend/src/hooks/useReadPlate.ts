import { useState, useRef, useEffect } from "react"
import { FFmpeg } from "@ffmpeg/ffmpeg"
import { fetchFile, toBlobURL } from "@ffmpeg/util"
import API from "../services/api"
import { drawBoxes, extractVideoFrames, exportReportCSV } from "../utils/readplate.utils"
import type { ApiResponse, DetectionReport, VideoTypeMetric } from "../types/readplate.types"

const FFMPEG_CDN = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm"

export function useReadPlate() {
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

  // WebSocket streaming
  const [wsFrameSrc, setWsFrameSrc] = useState<string | null>(null)
  const [wsProgress, setWsProgress] = useState(0)
  const [wsStatus,   setWsStatus]   = useState<string>("")
  const wsRef = useRef<WebSocket | null>(null)

  // MediaRecorder
  const [downloadUrl,  setDownloadUrl]  = useState<string | null>(null)
  const [isRecording,  setIsRecording]  = useState(false)
  const [isConverting, setIsConverting] = useState(false)
  const mediaRecorderRef   = useRef<MediaRecorder | null>(null)
  const recordedChunksRef  = useRef<Blob[]>([])
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef          = useRef<MediaStream | null>(null)
  const cancelledRef       = useRef(false)

  // FFmpeg
  const ffmpegRef = useRef(new FFmpeg())

  const inputRef  = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef(new Image())
  const videoRef  = useRef<HTMLVideoElement>(null)

  // ── Cargar FFmpeg desde CDN ──────────────────────────────────────────────
  useEffect(() => {
    const loadFFmpeg = async () => {
      const ffmpeg = ffmpegRef.current
      if (ffmpeg.loaded) return
      try {
        await ffmpeg.load({
          coreURL: await toBlobURL(`${FFMPEG_CDN}/ffmpeg-core.js`, "text/javascript"),
          wasmURL: await toBlobURL(`${FFMPEG_CDN}/ffmpeg-core.wasm`, "application/wasm"),
        })
      } catch (err) {
        console.error("Error cargando FFmpeg:", err)
        setError("No se pudo cargar el conversor de video. Recargá la página.")
      }
    }
    loadFFmpeg()
  }, [])

  // Dibuja bboxes cuando llega resultado de imagen
  useEffect(() => {
    if (!result || !preview || !canvasRef.current) return
    if (fileType === "image") {
      const img = imgRef.current
      img.onload = () => { if (canvasRef.current) drawBoxes(canvasRef.current, img, result.plates) }
      img.src = preview
    }
  }, [result, preview, fileType])

  // ── MediaRecorder helpers ────────────────────────────────────────────────
  const initRecorder = (width: number, height: number) => {
    if (offscreenCanvasRef.current) return
    const canvas    = document.createElement("canvas")
    canvas.width    = width
    canvas.height   = height
    offscreenCanvasRef.current = canvas
    const stream = canvas.captureStream(15)
    streamRef.current = stream

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm;codecs=vp8"

    const recorder = new MediaRecorder(stream, { mimeType })
    recordedChunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data)
    }

    recorder.onstop = async () => {
      if (cancelledRef.current || recordedChunksRef.current.length === 0) {
        setIsRecording(false)
        return
      }

      const ffmpeg = ffmpegRef.current
      if (!ffmpeg.loaded) {
        setError("El conversor de video no está listo.")
        setIsRecording(false)
        return
      }

      setIsConverting(true)
      try {
        const webmBlob = new Blob(recordedChunksRef.current, { type: "video/webm" })
        await ffmpeg.writeFile("input.webm", await fetchFile(webmBlob))
        await ffmpeg.exec(["-i", "input.webm", "-c", "copy", "output.mp4"])
        const mp4Data = await ffmpeg.readFile("output.mp4") as unknown as Uint8Array
        const copy = new Uint8Array(mp4Data.byteLength)
        copy.set(mp4Data)
        const mp4Blob = new Blob([copy], { type: "video/mp4" })
        setDownloadUrl(URL.createObjectURL(mp4Blob))
      } catch (err) {
        console.error("Error convirtiendo a MP4:", err)
        setError("Error al convertir el video a MP4")
      } finally {
        setIsConverting(false)
        setIsRecording(false)
      }
    }

    recorder.start(100)
    mediaRecorderRef.current = recorder
    setIsRecording(true)
  }

  const paintFrameToCanvas = (imgEl: HTMLImageElement) => {
    const canvas = offscreenCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.drawImage(imgEl, 0, 0)
  }

  const stopRecorder = () => {
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop()
    offscreenCanvasRef.current = null
    streamRef.current          = null
  }

  // ── Selección de archivo ─────────────────────────────────────────────────
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
    setIsConverting(false)

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
      setPreview(URL.createObjectURL(f))
      try {
        setVideoFrames(await extractVideoFrames(f, 8))
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

  // ── Reportes ─────────────────────────────────────────────────────────────
  const generateVideoReport = (videoTypes: VideoTypeMetric[], processingTime: number) => {
    const dateTime = new Date().toLocaleString("es-EC", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    })
    setReport(videoTypes.map((item, i) => ({
      id:             i + 1,
      filename:       file?.name || "video",
      location:       "Video",
      vehicleType:    item.type,
      confidence:     Math.round(item.percent),
      dateTime,
      processingTime: Number(processingTime.toFixed(2)),
      processed:      true,
      coordinates:    `${item.count} detecciones`,
    })))
  }

  const generateReport = (data: ApiResponse, processingTime: number) => {
    const dateTime = new Date().toLocaleString("es-EC", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    })
    setReport(prev => [...prev, ...data.plates.map((plate, i) => ({
      id:             prev.length + i + 1,
      filename:       file?.name || "desconocido",
      location:       "Pendiente",
      vehicleType:    plate.vehicle?.type_es || "Desconocido",
      // FIX: usar detector_confidence (antes yolo_confidence — campo renombrado)
      confidence:     Math.round(plate.detector_confidence * 100),
      dateTime,
      processingTime: Number(processingTime.toFixed(2)),
      processed:      true,
      coordinates:    `[${plate.bbox.join(", ")}]`,
    }))])
  }

  // ── Upload / Analyze ─────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setWsFrameSrc(null)
    setWsProgress(0)
    setDownloadUrl(null)
    setIsConverting(false)
    cancelledRef.current = false
    recordedChunksRef.current  = []
    offscreenCanvasRef.current = null
    const startTime = performance.now()

    // ── IMAGEN: llama a /api/v2/detect/full con soporte multi-modelo ────────
    // FIX: era /detect-plate (endpoint legacy sin vehicles, sin detector_confidence)
    if (fileType === "image") {
      try {
        const formData = new FormData()
        formData.append("file", file)

        // FIX: endpoint v2 con parámetros correctos
        const res = await API.post<ApiResponse>(
          "/api/v2/detect/full",
          formData,
          {
            params: {
              detector:        "ensemble",  // yolo | rtdetr | ensemble
              include_vehicle: true,
              include_labels:  true,
            },
          }
        )

        const processingTime = (performance.now() - startTime) / 1000

        // FIX: el backend v2 devuelve detector_confidence, no yolo_confidence.
        // Normalizamos aquí para que el resto del front no necesite saber esto.
        const normalized: ApiResponse = {
          ...res.data,
          plates: res.data.plates.map(p => ({
            ...p,
            // Si por alguna razón llega yolo_confidence del legacy, lo mapeamos
            detector_confidence: p.detector_confidence ?? (p as any).yolo_confidence ?? 0,
          })),
        }

        setResult(normalized)
        generateReport(normalized, processingTime)
      } catch (e: any) {
        setError(e?.response?.data?.detail || "Error al conectar con el servidor.")
      } finally {
        setLoading(false)
      }
      return
    }

    // ── VIDEO: WebSocket streaming (sin cambios — el WS de vehículos es correcto aquí) ─
    try {
      const ws = new WebSocket("ws://localhost:8000/ws/detect-vehicle/video")
      wsRef.current = ws

      ws.onopen = async () => {
        setWsStatus("Enviando video al servidor...")
        ws.send(await file.arrayBuffer())
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === "status") {
          setWsStatus(data.message)
        }

        if (data.type === "frame") {
          const src = `data:image/jpeg;base64,${data.frame}`
          setWsFrameSrc(src)
          setWsProgress(data.progress ?? 0)
          setWsStatus(`Procesando frame ${data.frame_num}...`)

          const imgEl  = new Image()
          imgEl.onload = () => {
            if (!offscreenCanvasRef.current) initRecorder(imgEl.naturalWidth, imgEl.naturalHeight)
            paintFrameToCanvas(imgEl)
          }
          imgEl.src = src

          // FIX: vehicle_counter puede venir undefined si el frame no tiene conteo aún
          const counter = (data.vehicle_counter ?? {}) as Record<string, number>
          const totalVehicles = Object.values(counter).reduce((a, b) => a + b, 0)

          setResult({
            total:    0,
            vehicles: totalVehicles,
            plates:   [],
            video_metrics: {
              total_unique_vehicles: totalVehicles,
              total_raw_detections:  data.frame_num ?? 0,
              frames_processed:      data.frame_num ?? 0,
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

        if (data.type === "done") {
          const processingTime = (performance.now() - startTime) / 1000
          const metrics        = data.metrics
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
              by_type:               metrics.by_type ?? [],
            },
          }

          setResult(finalResult)
          // FIX: guardar con by_type seguro (puede ser undefined si el video está vacío)
          generateVideoReport(metrics.by_type ?? [], processingTime)
          setWsProgress(100)
          setWsStatus("✓ Procesamiento completo")
          setLoading(false)
          ws.close()
        }

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

      // FIX: no usar `loading` del closure — puede estar stale; usar cancelledRef
      ws.onclose = () => {
        if (!cancelledRef.current) setLoading(false)
      }

    } catch {
      setError("Error al iniciar el análisis de video.")
      stopRecorder()
      setLoading(false)
    }
  }

  const cancelWs = () => {
    cancelledRef.current = true
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
    stopRecorder()
    setLoading(false)
    setWsStatus("Cancelado")
  }

  const reset = () => {
    cancelledRef.current = true
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
    stopRecorder()
    if (downloadUrl) URL.revokeObjectURL(downloadUrl)
    if (preview && fileType === "video") URL.revokeObjectURL(preview)
    setFile(null); setFileType(null); setPreview(null); setResult(null)
    setError(null); setReport([]); setVideoFrames([]); setCurrentFrame(0)
    setWsFrameSrc(null); setWsProgress(0); setWsStatus(""); setDownloadUrl(null)
    setIsRecording(false); setIsConverting(false); recordedChunksRef.current = []; setLoading(false)
  }

  const downloadCanvas = () => {
    if (!canvasRef.current) return
    const a    = document.createElement("a")
    a.download = "deteccion-vehicular.png"
    a.href     = canvasRef.current.toDataURL("image/png")
    a.click()
  }

  return {
    file, fileType, preview, result, loading, error,
    report, showReport, setShowReport,
    videoFrames, currentFrame, setCurrentFrame,
    wsFrameSrc, wsProgress, wsStatus,
    downloadUrl, isRecording, isConverting,
    inputRef, canvasRef, videoRef,
    handleFile, handleDrop, handleUpload, cancelWs, reset,
    downloadCanvas,
    downloadReportCSV: () => exportReportCSV(report),
  }
}