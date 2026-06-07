import type { PlateResult } from "../types/readplate.types"

// ── Canvas — dibuja bboxes de vehículos y placas ──────────────────────────────
export function drawBoxes(canvas: HTMLCanvasElement, img: HTMLImageElement, plates: PlateResult[]) {
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

// ── Extraer frames de video para previsualización ─────────────────────────────
export async function extractVideoFrames(videoFile: File, maxFrames = 10): Promise<string[]> {
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

// ── CSV export ─────────────────────────────────────────────────────────────────
export function exportReportCSV(report: import("../types/readplate.types").DetectionReport[]) {
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
