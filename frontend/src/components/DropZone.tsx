import type { RefObject } from "react"

interface DropZoneProps {
  preview:    string | null
  fileType:   "image" | "video" | null
  file:       File | null
  videoRef:   RefObject<HTMLVideoElement | null>
  inputRef:   RefObject<HTMLInputElement | null>
  onDrop:     (e: React.DragEvent) => void
  onFile:     (f: File) => void
  onReset:    () => void
}

export function DropZone({ preview, fileType, file, videoRef, inputRef, onDrop, onFile, onReset }: DropZoneProps) {
  return (
    <>
      <div
        className={`rp-dropzone ${preview ? "rp-dropzone--has-image" : ""}`}
        onClick={() => !preview && inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        role={!preview ? "button" : undefined}
        tabIndex={!preview ? 0 : undefined}
        aria-label={!preview ? "Área para arrastrar o seleccionar archivo" : undefined}
        onKeyDown={!preview ? (e) => e.key === "Enter" && inputRef.current?.click() : undefined}
      >
        {preview ? (
          <>
            {fileType === "video" ? (
              <video
                ref={videoRef}
                src={preview}
                controls
                className="rp-preview"
                style={{ maxHeight: 360 }}
              />
            ) : (
              <img src={preview} alt="Vista previa del archivo seleccionado" className="rp-preview" />
            )}
            <button
              className="rp-clear"
              onClick={(e) => { e.stopPropagation(); onReset() }}
              aria-label="Eliminar archivo seleccionado"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
                <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </>
        ) : (
          <div className="rp-drop-hint">
            <div className="rp-drop-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 16V8M12 8L8 12M12 8L16 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M20 16.7A4 4 0 0 0 18 9h-1.26A7 7 0 1 0 4 15.7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <p>Arrastra imagen o video aquí</p>
            <span>o haz clic para seleccionar · JPG · PNG · MP4</span>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,video/mp4,video/avi,video/quicktime"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
        aria-hidden="true"
        tabIndex={-1}
      />

      {file && (
        <div className="rp-file-info" role="status" aria-live="polite">
          <span className="rp-file-name" title={file.name}>{file.name}</span>
          <span className="rp-file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          <span className="rp-file-type">
            {fileType === "video" ? "Video" : "Imagen"}
          </span>
        </div>
      )}
    </>
  )
}