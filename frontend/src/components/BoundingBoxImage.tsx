// components/BoundingBoxImage.tsx
// Dibuja los bboxes reales sobre la imagen usando <canvas>
// Recibe las detecciones del resultado del modelo (vehículos + placas)

import { useEffect, useRef } from "react"
import type { PlateDetection, VehicleDetection } from "../types/comparison_types"

interface BoundingBoxImageProps {
  src:       string | null
  alt:       string
  modelName: string
  color:     string
  vehicles?: VehicleDetection[]
  plates?:   PlateDetection[]
}

// Colores internos fijos para vehículo vs placa
const VEHICLE_COLOR = "#3b82f6"   // azul — vehículo
const PLATE_COLOR_ALPHA = 0.9

export function BoundingBoxImage({
  src, alt, modelName, color, vehicles = [], plates = [],
}: BoundingBoxImageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!src || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx    = canvas.getContext("2d")
    if (!ctx) return

    const img  = new Image()
    img.onload = () => {
      // Ajustar canvas al tamaño real de la imagen
      canvas.width  = img.naturalWidth
      canvas.height = img.naturalHeight
      ctx.drawImage(img, 0, 0)

      const scale     = img.naturalWidth / 640  // factor si el backend normaliza a 640
      const lineWidth = Math.max(2, img.naturalWidth * 0.003)
      const fontSize  = Math.max(13, img.naturalWidth * 0.018)

      // ── Dibujar bboxes de vehículos ─────────────────────────────────
      vehicles.forEach((v) => {
        const [x1, y1, x2, y2] = v.bbox
        const w = x2 - x1
        const h = y2 - y1

        // Box punteado para vehículo
        ctx.strokeStyle   = VEHICLE_COLOR
        ctx.lineWidth     = lineWidth
        ctx.setLineDash([8, 4])
        ctx.strokeRect(x1, y1, w, h)
        ctx.setLineDash([])

        // Etiqueta vehículo
        const label  = `${v.type_es} ${(v.confidence * 100).toFixed(0)}%`
        ctx.font     = `bold ${fontSize}px monospace`
        const tw     = ctx.measureText(label).width
        const padX   = fontSize * 0.45
        const padY   = fontSize * 0.35
        const tagH   = fontSize + padY * 2
        ctx.fillStyle = VEHICLE_COLOR
        ctx.fillRect(x1, y1 - tagH, tw + padX * 2, tagH)
        ctx.fillStyle = "#ffffff"
        ctx.fillText(label, x1 + padX, y1 - padY)
      })

      // ── Dibujar bboxes de placas ────────────────────────────────────
      plates.forEach((p) => {
        const [x1, y1, x2, y2] = p.bbox
        const w = x2 - x1
        const h = y2 - y1

        // Box sólido para placa — usa el color del modelo
        ctx.strokeStyle = color
        ctx.lineWidth   = lineWidth * 1.2
        ctx.strokeRect(x1, y1, w, h)

        // Etiqueta placa (texto OCR + confianza)
        const text   = p.plate
          ? `${p.plate}  ${(p.ocr_confidence * 100).toFixed(0)}%`
          : `det ${(p.detector_confidence * 100).toFixed(0)}%`
        ctx.font     = `bold ${fontSize}px monospace`
        const tw     = ctx.measureText(text).width
        const padX   = fontSize * 0.45
        const padY   = fontSize * 0.35
        const tagH   = fontSize + padY * 2

        // Etiqueta debajo del box de placa
        ctx.fillStyle = color
        ctx.fillRect(x1, y2, tw + padX * 2, tagH)
        ctx.fillStyle = "#07090f"
        ctx.fillText(text, x1 + padX, y2 + tagH - padY)

        // Confianza detector en esquina superior derecha
        const detText = `YOLO ${(p.detector_confidence * 100).toFixed(0)}%`
        ctx.font      = `${fontSize * 0.85}px monospace`
        const dtw     = ctx.measureText(detText).width
        ctx.fillStyle = "rgba(0,0,0,0.6)"
        ctx.fillRect(x2 - dtw - padX * 2, y1, dtw + padX * 2, tagH * 0.85)
        ctx.fillStyle = color
        ctx.fillText(detText, x2 - dtw - padX, y1 + tagH * 0.6)
      })
    }
    img.src = src
  }, [src, vehicles, plates, color])

  if (!src) {
    return (
      <div className="bbox-image-empty">
        <div className="bbox-image-placeholder">
          <div className="bbox-placeholder-ring" style={{ borderColor: color }} />
          <span style={{ color, fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}>
            {modelName}
          </span>
          <p>Esperando procesamiento...</p>
        </div>
      </div>
    )
  }

  // Si no hay detecciones todavía pero hay imagen (imagen cruda preview)
  const hasDetections = vehicles.length > 0 || plates.length > 0

  return (
    <div className="bbox-image-container">
      <div className="bbox-image-header" style={{ borderColor: color + "55" }}>
        <span className="bbox-image-label" style={{ color }}>
          {modelName}
        </span>
        <span className="bbox-image-sub">
          {hasDetections
            ? `${vehicles.length} vehículo${vehicles.length !== 1 ? "s" : ""} · ${plates.length} placa${plates.length !== 1 ? "s" : ""}`
            : "Sin detecciones"}
        </span>
      </div>

      <div className="bbox-image-wrapper">
        {hasDetections ? (
          // Canvas con bboxes dibujados
          <canvas
            ref={canvasRef}
            className="bbox-canvas"
            role="img"
            aria-label={`${alt} con bounding boxes`}
          />
        ) : (
          // Imagen cruda sin anotaciones
          <img src={src} alt={alt} className="bbox-image" />
        )}
      </div>
    </div>
  )
}