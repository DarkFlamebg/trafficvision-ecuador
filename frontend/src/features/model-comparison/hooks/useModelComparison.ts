import { useState, useRef, useCallback } from "react"
import API from "../../../services/api"
import type {
  ModelType,
  ComparisonResult,
  ComparisonImageResponse,
  WebSocketMessage,
} from "../types/comparison_types"

export function useModelComparison() {
  const [file,         setFile]         = useState<File | null>(null)
  const [fileType,     setFileType]     = useState<"image" | "video" | null>(null)
  const [preview,      setPreview]      = useState<string | null>(null)
  const [loading,      setLoading]      = useState(false)
  const [activeModel,  ]               = useState<ModelType | null>(null)
  const [error,        setError]        = useState<string | null>(null)

  const [realPlate,        setRealPlate]        = useState<string>("")
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult>({})

  const [yoloFrame,      setYoloFrame]      = useState<string | null>(null)
  const [rtdetrFrame,    setRtdetrFrame]    = useState<string | null>(null)
  const [mambaFrame, setMambaFrame] = useState<string | null>(null)
  const [yoloProgress,   setYoloProgress]   = useState(0)
  const [rtdetrProgress, setRtdetrProgress] = useState(0)
  const [mambaProgress, setMambaProgress] = useState(0)
  const [yoloStatus,     setYoloStatus]     = useState("")
  const [rtdetrStatus,   setRtdetrStatus]   = useState("")
  const [mambaStatus, setMambaStatus] = useState("")

  const [yoloTelemetry,   setYoloTelemetry]   = useState<any>(null)
  const [rtdetrTelemetry, setRtdetrTelemetry] = useState<any>(null)
  const [mambaTelemetry, setMambaTelemetry] = useState<any>(null)

  const yoloWsRef   = useRef<WebSocket | null>(null)
  const rtdetrWsRef = useRef<WebSocket | null>(null)
  const mambaWsRef = useRef<WebSocket | null>(null)

  // Ref para saber qué modelos ya terminaron (evita stale closures)
  const doneRef = useRef<Set<ModelType>>(new Set())

  // Ref para acumular los últimos datos de frame recibidos por modelo
  // (fallback si el backend cierra sin enviar "done")
  const lastFrameDataRef = useRef<Record<ModelType, any>>({} as any)

  const inputRef = useRef<HTMLInputElement>(null)

  // ─────────────────────────────────────────────────────────────────────────
  // Helper: marca un modelo como terminado y apaga loading si ambos acabaron
  // ─────────────────────────────────────────────────────────────────────────
  const markDone = useCallback((model: ModelType) => {
    doneRef.current.add(model)
    if (doneRef.current.has("yolo") && doneRef.current.has("rtdetr") && doneRef.current.has("mamba")) {
      setLoading(false)
    }
  }, [])

  // ─────────────────────────────────────────────────────────────────────────
  // Selección de archivo
  // ─────────────────────────────────────────────────────────────────────────
  const handleFile = useCallback(async (f: File) => {
    setFile(f)
    setError(null)
    setComparisonResult({})
    setYoloFrame(null)
    setRtdetrFrame(null)
    setMambaFrame(null)
    setYoloProgress(0)
    setRtdetrProgress(0)
    setMambaProgress(0)
    setYoloStatus("")
    setRtdetrStatus("")
    setMambaStatus("")

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
    } else {
      setError("Formato no soportado. Usa JPG, PNG o MP4.")
    }
  }, [preview, fileType])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }, [handleFile])

  // ─────────────────────────────────────────────────────────────────────────
  // Comparación de IMAGEN
  // ─────────────────────────────────────────────────────────────────────────
  const runImageComparison = useCallback(async () => {
    if (!file || fileType !== "image") return
    setLoading(true)
    setError(null)
    setComparisonResult({})
    try {
      const [yoloRes, rtdetrRes, mambaRes] = await Promise.all([
        runSingleImageDetection("yolo"),
        runSingleImageDetection("rtdetr"),
        runSingleImageDetection("mamba"),
      ])
      setComparisonResult({ yolo: yoloRes, rtdetr: rtdetrRes, mamba: mambaRes })
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Error al realizar la comparación")
    } finally {
      setLoading(false)
    }
  }, [file, fileType])

  const runSingleImageDetection = async (
    model: ModelType
  ): Promise<ComparisonImageResponse> => {
    if (!file) throw new Error("No hay archivo")
    const formData = new FormData()
    formData.append("file", file)
    const res = await API.post<ComparisonImageResponse>(
      `/api/v1/compare/image?model=${model}`,
      formData
    )
    return res.data
  }

  const submitPlateFeedback = useCallback(async (
    plate: { detection_id?: number },
    isCorrect: boolean,
    correctedPlateText?: string
  ) => {
    if (!plate.detection_id) {
      throw new Error("No detection_id disponible para enviar feedback")
    }

    const payload = {
      detection_id: plate.detection_id,
      is_correct: isCorrect,
      corrected_plate_text: correctedPlateText || null,
    }

    await API.post(
      "/api/v1/compare/feedback",
      payload,
      { headers: { "Content-Type": "application/json" } }
    )
  }, [])

  // ─────────────────────────────────────────────────────────────────────────
  // Comparación de VIDEO
  // ─────────────────────────────────────────────────────────────────────────
  const runVideoComparison = useCallback(async () => {
    if (!file || fileType !== "video") return

    setLoading(true)
    setError(null)
    setComparisonResult({})
    setYoloFrame(null)
    setRtdetrFrame(null)
    setMambaFrame(null)
    setYoloProgress(0)
    setRtdetrProgress(0)
    setMambaProgress(0)
    setYoloStatus("")
    setRtdetrStatus("")
    setMambaStatus("")

    doneRef.current = new Set()
    lastFrameDataRef.current = {} as any
    setYoloTelemetry(null)
    setRtdetrTelemetry(null)
    setMambaTelemetry(null)

    try {
      const videoBytes = await file.arrayBuffer()
      startVideoWebSocket("yolo",   videoBytes)
      startVideoWebSocket("rtdetr", videoBytes)
      startVideoWebSocket("mamba", videoBytes)
    } catch {
      setError("Error al iniciar la comparación de video")
      setLoading(false)
    }
  }, [file, fileType])

  const startVideoWebSocket = (model: ModelType, videoBytes: ArrayBuffer): void => {
    const ws = new WebSocket(
      `ws://127.0.0.1:8000/api/v1/compare/video?model=${model}`
    )

    if (model === "yolo") yoloWsRef.current   = ws
    else if (model === "rtdetr") rtdetrWsRef.current = ws
    else mambaWsRef.current = ws

    const setFrame    = model === "yolo" ? setYoloFrame    : model === "rtdetr" ? setRtdetrFrame : setMambaFrame
    const setProgress = model === "yolo" ? setYoloProgress : model === "rtdetr" ? setRtdetrProgress : setMambaProgress
    const setStatus   = model === "yolo" ? setYoloStatus   : model === "rtdetr" ? setRtdetrStatus : setMambaStatus
    const setTelemetry = model === "yolo" ? setYoloTelemetry : model === "rtdetr" ? setRtdetrTelemetry : setMambaTelemetry

    ws.onopen = () => {
      setStatus(`[${model.toUpperCase()}] Enviando video...`)
      ws.send(videoBytes)
    }

    ws.onmessage = (event) => {
      let data: WebSocketMessage
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }

      if (data.type === "status") {
        setStatus(data.message)
      }

      if (data.type === "frame") {
        setFrame(`data:image/jpeg;base64,${data.frame}`)
        setProgress(data.progress ?? 0)

        // ── DEFENSIVO: evitar crash si el backend no manda estos campos ──
        const frameNum = data.frame_num ?? 0
        const infMs    = data.inference_ms ?? 0

        setStatus(
          `[${model.toUpperCase()}] Frame ${frameNum} · ${infMs.toFixed(1)}ms`
        )

        const telemetry = {
          vehicle_counter: data.vehicle_counter ?? {},
          plates_count:    data.plates_count ?? 0,
          inference_ms:    infMs,
          frame_num:       frameNum
        }

        setTelemetry(telemetry)

        lastFrameDataRef.current[model] = telemetry
      }

      if (data.type === "done") {
        setProgress(100)
        setStatus(`[${model.toUpperCase()}] ✓ Completado`)
        setComparisonResult((prev) => ({ ...prev, [model]: data.metrics }))
        ws.close()
        markDone(model)
      }

      if (data.type === "error") {
        setError(`[${model.toUpperCase()}] ${data.message}`)
        setStatus(`[${model.toUpperCase()}] ✗ Error`)
        ws.close()
        markDone(model)
      }
    }

    ws.onerror = () => {
      setError(`[${model.toUpperCase()}] Error en la conexión WebSocket`)
      setStatus(`[${model.toUpperCase()}] ✗ Error de conexión`)
      markDone(model)
    }

    // ── DEFENSA CLAVE: el backend cerró sin enviar "done" ──────────────────
    ws.onclose = () => {
      // Si ya fue marcado como done (cierre normal tras "done"), salir
      if (doneRef.current.has(model)) return

      // Recuperar desde el último frame recibido
      const lastFrame = lastFrameDataRef.current[model]

      if (lastFrame) {
        const fallbackMetrics = buildFallbackMetrics(model, lastFrame)
        setComparisonResult((prev) => ({ ...prev, [model]: fallbackMetrics }))
        setProgress(100)
        setStatus(`[${model.toUpperCase()}] ✓ Completado`)
      } else {
        setStatus(`[${model.toUpperCase()}] Sin datos recibidos`)
      }

      markDone(model)
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Métricas de fallback construidas desde datos de frames
  // ─────────────────────────────────────────────────────────────────────────
  const buildFallbackMetrics = (
    model: ModelType,
    lastFrame: {
      vehicle_counter: Record<string, number>
      plates_count:    number
      inference_ms:    number
    }
  ) => {
    const counter       = lastFrame.vehicle_counter ?? {}
    const totalVehicles = Object.values(counter).reduce(
      (sum: number, v) => sum + (v as number), 0
    )
    return {
      model,
      inference_ms:           lastFrame.inference_ms ?? 0,
      avg_inference_ms:       lastFrame.inference_ms ?? 0,
      vehicles_detected:      totalVehicles,
      total_unique_vehicles:  totalVehicles,
      avg_vehicle_confidence: 0,
      plates_detected:        lastFrame.plates_count ?? 0,
      total_plates_detected:  lastFrame.plates_count ?? 0,
      avg_plate_confidence:   0,
      plates_with_ocr:        0,
      vehicles_by_type:       counter,
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cancelar
  // ─────────────────────────────────────────────────────────────────────────
  const cancelComparison = useCallback(() => {
    yoloWsRef.current?.close()
    yoloWsRef.current = null
    rtdetrWsRef.current?.close()
    rtdetrWsRef.current = null
    mambaWsRef.current?.close()
    mambaWsRef.current = null
    doneRef.current = new Set()
    setLoading(false)
    setYoloStatus("Cancelado")
    setRtdetrStatus("Cancelado")
    setMambaStatus("Cancelado")
  }, [])

  // ─────────────────────────────────────────────────────────────────────────
  // Reset
  // ─────────────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    cancelComparison()
    if (preview && fileType === "video") URL.revokeObjectURL(preview)
    setFile(null)
    setFileType(null)
    setPreview(null)
    setError(null)
    setRealPlate("")
    setComparisonResult({})
    setYoloFrame(null)
    setRtdetrFrame(null)
    setMambaFrame(null)
    setYoloProgress(0)
    setRtdetrProgress(0)
    setMambaProgress(0)
    setYoloStatus("")
    setRtdetrStatus("")
    setMambaStatus("")
    setLoading(false)
  }, [cancelComparison, preview, fileType])

  return {
    file, fileType, preview, loading, activeModel, error,
    realPlate, comparisonResult,
    yoloFrame, rtdetrFrame, mambaFrame,
    yoloProgress, rtdetrProgress, mambaProgress,
    yoloStatus, rtdetrStatus, mambaStatus,
    yoloTelemetry, rtdetrTelemetry, mambaTelemetry,
    inputRef,
    handleFile, handleDrop,
    runImageComparison, runVideoComparison,
    submitPlateFeedback,
    cancelComparison, reset, setRealPlate,
  }
}