interface VideoFrameStripProps {
  frames:       string[]
  currentFrame: number
  onSelect:     (i: number) => void
}

export function VideoFrameStrip({ frames, currentFrame, onSelect }: VideoFrameStripProps) {
  if (frames.length === 0) return null
  return (
    <div className="rp-video-frames">
      <div className="rp-section-label">FRAMES EXTRAÍDOS</div>
      <div style={{ display: "flex", gap: 8, overflowX: "auto", padding: "0.5rem 0" }}>
        {frames.map((frame, i) => (
          <img
            key={i}
            src={frame}
            alt={`frame-${i}`}
            style={{
              width: 80, height: 60, objectFit: "cover", borderRadius: 4,
              border: i === currentFrame ? "2px solid #22d3ee" : "2px solid transparent",
              cursor: "pointer",
            }}
            onClick={() => onSelect(i)}
          />
        ))}
      </div>
    </div>
  )
}
