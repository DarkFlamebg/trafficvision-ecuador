// ── LabelBadge ─────────────────────────────────────────────────────────────────
const LABEL_STYLES: Record<string, { bg: string; color: string }> = {
  "Legible":  { bg: "rgba(16,185,129,0.12)", color: "#10b981" },
  "Ilegible": { bg: "rgba(239,68,68,0.12)",  color: "#ef4444" },
  "No":       { bg: "rgba(16,185,129,0.12)", color: "#10b981" },
  "Parcial":  { bg: "rgba(245,158,11,0.12)", color: "#f59e0b" },
  "Severa":   { bg: "rgba(239,68,68,0.12)",  color: "#ef4444" },
  "Sí":       { bg: "rgba(245,158,11,0.12)", color: "#f59e0b" },
}

export function LabelBadge({ name, value }: { name: string; value: string }) {
  const style = LABEL_STYLES[value] ?? { bg: "rgba(255,255,255,0.06)", color: "#94a3b8" }
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 3,
      padding: "0.45rem 0.75rem", borderRadius: 8,
      background: style.bg, border: `1px solid ${style.color}30`, minWidth: 80,
    }}>
      <span style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>
        {name}
      </span>
      <span style={{ fontSize: 12, color: style.color, fontWeight: 700 }}>{value}</span>
    </div>
  )
}
