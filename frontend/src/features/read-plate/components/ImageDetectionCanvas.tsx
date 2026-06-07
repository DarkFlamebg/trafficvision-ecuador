import type { RefObject } from "react"

interface ImageDetectionCanvasProps {
  canvasRef:  RefObject<HTMLCanvasElement | null>
  onDownload: () => void
}

export function ImageDetectionCanvas({ canvasRef, onDownload }: ImageDetectionCanvasProps) {
  return (
    <div className="rp-canvas-wrap">
      <div className="rp-section-label" style={{ marginBottom: "0.5rem" }}>
        DETECCIÓN VISUAL
      </div>
      <canvas
        ref={canvasRef}
        className="rp-canvas"
        role="img"
        aria-label="Imagen con detecciones superpuestas"
      />
      <button className="rp-btn-download" onClick={onDownload} aria-label="Descargar imagen con anotaciones">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <path d="M7 1V9M7 9L4 6M7 9L10 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M2 11H12" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
        </svg>
        Descargar imagen anotada
      </button>
    </div>
  )
}