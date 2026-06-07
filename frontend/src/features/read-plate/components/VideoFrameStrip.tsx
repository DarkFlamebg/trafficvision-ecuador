interface VideoFrameStripProps {
  frames:       string[]
  currentFrame: number
  onSelect:     (i: number) => void
}

export function VideoFrameStrip({ frames, currentFrame, onSelect }: VideoFrameStripProps) {
  if (frames.length === 0) return null

  return (
    <div className="rp-video-frames" role="region" aria-label="Tira de fotogramas del video">
      <div className="rp-section-label" style={{ marginBottom: "0.5rem" }}>
        FRAMES EXTRAÍDOS · <span style={{ color: "var(--accent-cyan)" }}>{frames.length}</span>
      </div>
      <div
        style={{ display: "flex", gap: 6, overflowX: "auto", padding: "0.25rem 0 0.5rem" }}
        role="listbox"
        aria-label="Fotogramas del video"
      >
        {frames.map((frame, i) => (
          <img
            key={i}
            src={frame}
            alt={`Fotograma ${i + 1}`}
            role="option"
            aria-selected={i === currentFrame}
            tabIndex={0}
            style={{
              width: 76,
              height: 56,
              objectFit: "cover",
              borderRadius: 6,
              border: i === currentFrame
                ? "2px solid #22d3ee"
                : "2px solid transparent",
              cursor: "pointer",
              flexShrink: 0,
              opacity: i === currentFrame ? 1 : 0.65,
              transition: "all 150ms ease",
              boxShadow: i === currentFrame ? "0 0 12px rgba(34,211,238,0.35)" : "none",
            }}
            onClick={() => onSelect(i)}
            onKeyDown={(e) => e.key === "Enter" && onSelect(i)}
          />
        ))}
      </div>
    </div>
  )
}