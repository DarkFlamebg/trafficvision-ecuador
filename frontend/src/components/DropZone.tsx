import type { RefObject } from "react"

interface DropZoneProps {
  preview:    string | null
  fileType:   "image" | "video" | null
  file:       File | null
  videoRef: RefObject<HTMLVideoElement | null>;
  inputRef:   RefObject<HTMLInputElement | null>;
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
      >
        {preview ? (
          <>
            {fileType === "video" ? (
              <video ref={videoRef} src={preview} controls className="rp-preview" style={{ maxHeight: 400 }} />
            ) : (
              <img src={preview} alt="preview" className="rp-preview" />
            )}
            <button className="rp-clear" onClick={(e) => { e.stopPropagation(); onReset() }}>✕</button>
          </>
        ) : (
          <div className="rp-drop-hint">
            <div className="rp-drop-icon">⊕</div>
            <p>Arrastra imagen o video aquí</p>
            <span>o haz clic para seleccionar · JPG / PNG / MP4</span>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,video/mp4,video/avi,video/quicktime"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
      />

      {file && (
        <div className="rp-file-info">
          <span className="rp-file-name">{file.name}</span>
          <span className="rp-file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          <span className="rp-file-type">{fileType === "video" ? "🎬 Video" : "🖼️ Imagen"}</span>
        </div>
      )}
    </>
  )
}
