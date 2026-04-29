import type { RefObject } from "react"

interface ImageDetectionCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onDownload:     () => void
}

export function ImageDetectionCanvas({ canvasRef, onDownload }: ImageDetectionCanvasProps) {
  return (
    <div className="rp-canvas-wrap">
      <div className="rp-section-label">DETECCIÓN VISUAL</div>
      <canvas ref={canvasRef} className="rp-canvas" />
      <button className="rp-btn-download" onClick={onDownload}>
        ⬇ Descargar imagen
      </button>
    </div>
  )
}
