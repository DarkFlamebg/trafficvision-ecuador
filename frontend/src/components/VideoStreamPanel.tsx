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
    <div className="rp-canvas-wrap" role="region" aria-label="Stream de video en vivo">

      {/* Label + REC indicator */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.6rem" }}>
        <div className="rp-section-label">DETECCIÓN EN VIVO</div>
        {isRecording && (
          <span
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.35rem",
              fontSize: "0.6rem", color: "#ef4444",
              fontFamily: "var(--font-mono)", letterSpacing: "0.1em",
              fontWeight: 700,
            }}
            aria-label="Grabando"
          >
            <span style={{
              width: 7, height: 7, borderRadius: "50%", background: "#ef4444",
              boxShadow: "0 0 6px #ef4444",
              animation: "rp-pulse 1s ease-in-out infinite",
              flexShrink: 0,
            }} aria-hidden="true" />
            REC
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div
        style={{ width: "100%", height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 100, marginBottom: "0.5rem", overflow: "hidden" }}
        role="progressbar"
        aria-valuenow={wsProgress}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Progreso: ${wsProgress}%`}
      >
        <div style={{
          height: "100%",
          width: `${wsProgress}%`,
          background: wsProgress === 100
            ? "linear-gradient(90deg, #00FF88, #22D3EE)"
            : "linear-gradient(90deg, #22D3EE, #3B82F6)",
          transition: "width 0.3s ease",
          borderRadius: 100,
        }} />
      </div>

      {/* Status */}
      {wsStatus && (
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: "0.65rem",
          color: wsProgress === 100 ? "var(--accent-green)" : "var(--text-dim)",
          marginBottom: "0.6rem", letterSpacing: "0.03em",
          transition: "color 0.3s ease",
        }} aria-live="polite">
          {wsStatus}{wsProgress > 0 && wsProgress < 100 ? ` · ${wsProgress}%` : ""}
        </p>
      )}

      {/* Live frame or spinner */}
      {wsFrameSrc ? (
        <img
          src={wsFrameSrc}
          alt="Fotograma procesado en tiempo real"
          style={{ width: "100%", borderRadius: 8, border: "1px solid rgba(255,255,255,0.07)", display: "block" }}
        />
      ) : (
        <div style={{
          width: "100%", minHeight: 180, borderRadius: 8,
          background: "var(--bg-card)", border: "1px solid var(--border-sub)",
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: 12,
        }} aria-label="Iniciando stream">
          <div className="rp-loading-dots" aria-hidden="true"><span /><span /><span /></div>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-dim)" }}>
            Iniciando stream...
          </span>
        </div>
      )}

      {/* Converting */}
      {isConverting && (
        <div style={{
          marginTop: "0.75rem", padding: "0.6rem 0.85rem", borderRadius: 8,
          background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.2)",
          fontFamily: "var(--font-mono)", fontSize: "0.68rem",
          color: "var(--accent-amber)", display: "flex", alignItems: "center", gap: "0.5rem",
        }} aria-live="polite">
          <span aria-hidden="true">⏳</span> Convirtiendo a MP4 con FFmpeg...
        </div>
      )}

      {/* Download */}
      {downloadUrl && !isRecording && !isConverting && (
        <a
          href={downloadUrl}
          download={`deteccion-${fileName?.replace(/\.[^.]+$/, "") ?? "video"}.mp4`}
          className="rp-btn-download"
          aria-label="Descargar video anotado en formato MP4"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M7 1V9M7 9L4 6M7 9L10 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 11H12" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
          </svg>
          Descargar video anotado (.mp4)
        </a>
      )}
    </div>
  )
}