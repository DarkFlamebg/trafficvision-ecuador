interface VideoStreamPanelProps {
  wsFrameSrc:    string | null
  wsProgress:    number
  wsStatus:      string
  isRecording:   boolean
  isConverting?: boolean
  downloadUrl:   string | null
  loading:       boolean
  fileName?:     string
}

export function VideoStreamPanel({
  wsFrameSrc, wsProgress, wsStatus, isRecording, isConverting, downloadUrl, loading, fileName,
}: VideoStreamPanelProps) {
  const visible = wsFrameSrc || loading
  if (!visible) return null

  return (
    <div className="rp-canvas-wrap" style={{ marginTop: "1rem" }}>

      {/* Label + REC indicator */}
      <div className="rp-section-label" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        DETECCIÓN VISUAL — EN VIVO
        {isRecording && (
          <span style={{
            display: "inline-flex", alignItems: "center", gap: "0.3rem",
            fontSize: "0.65rem", color: "#ef4444", fontFamily: "DM Mono, monospace",
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%", background: "#ef4444",
              animation: "rp-pulse 1s ease-in-out infinite",
            }} />
            REC
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div style={{ width: "100%", height: 3, background: "#1e293b", borderRadius: 2, margin: "0.5rem 0", overflow: "hidden" }}>
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

      {/* Live frame or spinner */}
      {wsFrameSrc ? (
        <img src={wsFrameSrc} alt="frame en vivo" style={{ width: "100%", borderRadius: 8, border: "1px solid #1e293b", display: "block" }} />
      ) : (
        <div style={{
          width: "100%", minHeight: 180, borderRadius: 8, background: "#0e1420",
          border: "1px solid #1c2a3a", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 12,
        }}>
          <div className="rp-loading-dots"><span /><span /><span /></div>
          <span style={{ fontFamily: "DM Mono, monospace", fontSize: "0.68rem", color: "#2a3f55" }}>
            Iniciando stream...
          </span>
        </div>
      )}

      {/* Converting indicator */}
      {isConverting && (
        <div style={{
          marginTop: "0.75rem", padding: "0.5rem", borderRadius: 8,
          background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)",
          fontFamily: "DM Mono, monospace", fontSize: "0.7rem", color: "#f59e0b", textAlign: "center",
        }}>
          ⏳ Convirtiendo a MP4 con FFmpeg...
        </div>
      )}

      {/* Download link */}
      {downloadUrl && !isRecording && !isConverting && (
        <a
          href={downloadUrl}
          download={`deteccion-${fileName?.replace(/\.[^.]+$/, "") ?? "video"}.mp4`}
          style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: "0.4rem", marginTop: "0.75rem", padding: "0.55rem 1rem", borderRadius: 8,
            background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.25)",
            color: "#22d3ee", textDecoration: "none",
            fontFamily: "DM Mono, monospace", fontSize: "0.75rem", transition: "background 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(34,211,238,0.16)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(34,211,238,0.08)")}
        >
          ⬇ Descargar video anotado (.mp4)
        </a>
      )}
    </div>
  )
}