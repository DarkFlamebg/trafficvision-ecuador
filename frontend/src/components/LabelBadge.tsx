// ── LabelBadge ─────────────────────────────────────────────────────────────────
const LABEL_STYLES: Record<string, { bg: string; color: string; border: string }> = {
  "Legible":  { bg: "rgba(0,255,136,0.07)",   color: "#00FF88", border: "rgba(0,255,136,0.2)"   },
  "Ilegible": { bg: "rgba(239,68,68,0.07)",   color: "#F87171", border: "rgba(239,68,68,0.2)"   },
  "No":       { bg: "rgba(0,255,136,0.07)",   color: "#00FF88", border: "rgba(0,255,136,0.2)"   },
  "Parcial":  { bg: "rgba(251,191,36,0.07)",  color: "#FBBF24", border: "rgba(251,191,36,0.2)"  },
  "Severa":   { bg: "rgba(239,68,68,0.07)",   color: "#F87171", border: "rgba(239,68,68,0.2)"   },
  "Sí":       { bg: "rgba(251,191,36,0.07)",  color: "#FBBF24", border: "rgba(251,191,36,0.2)"  },
}

export function LabelBadge({ name, value }: { name: string; value: string }) {
  const style = LABEL_STYLES[value] ?? {
    bg: "rgba(255,255,255,0.04)",
    color: "#7A90A8",
    border: "rgba(255,255,255,0.08)",
  }

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      gap: 3,
      padding: "0.4rem 0.65rem",
      borderRadius: 6,
      background: style.bg,
      border: `1px solid ${style.border}`,
      minWidth: 74,
    }}>
      <span style={{
        fontSize: 9,
        color: "var(--text-primary, #d8dfe6)",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        fontWeight: 700,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
      }}>
        {name}
      </span>
      <span style={{
        fontSize: 11,
        color: style.color,
        fontWeight: 700,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
      }}>
        {value}
      </span>
    </div>
  )
}