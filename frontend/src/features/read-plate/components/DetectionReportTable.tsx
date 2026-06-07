import type { DetectionReport } from "../types/readplate.types"

interface DetectionReportTableProps {
  report:     DetectionReport[]
  onDownload: () => void
}

export function DetectionReportTable({ report, onDownload }: DetectionReportTableProps) {
  if (report.length === 0) return null

  return (
    <div className="rp-report-section" role="region" aria-label="Tabla de reporte de detección">
      <div className="rp-report-header">
        <div>
          <h3>Detección Actual</h3>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-dim)", marginTop: "0.25rem" }}>
            {report.length} registro{report.length !== 1 ? "s" : ""} encontrado{report.length !== 1 ? "s" : ""}
          </p>
        </div>
        <button
          className="rp-btn-download-csv"
          onClick={onDownload}
          aria-label="Descargar reporte como archivo CSV"
        >
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
            <path d="M6.5 1V8.5M6.5 8.5L3.5 5.5M6.5 8.5L9.5 5.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M1.5 10.5H11.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
          </svg>
          Descargar CSV
        </button>
      </div>

      <div className="rp-report-table-wrap">
        <table className="rp-report-table" aria-label="Registros de detección vehicular">
          <thead>
            <tr>
              <th scope="col">ID</th>
              <th scope="col">Archivo</th>
              <th scope="col">Ubicación</th>
              <th scope="col">Tipo Vehículo</th>
              <th scope="col">Confianza</th>
              <th scope="col">Fecha/Hora</th>
              <th scope="col">Proc. (s)</th>
              <th scope="col">Estado</th>
              <th scope="col">Coordenadas</th>
            </tr>
          </thead>
          <tbody>
            {report.map((r) => (
              <tr key={r.id}>
                <td style={{ color: "var(--text-dim)" }}>
                  <span style={{
                    fontFamily: "var(--font-mono)", fontSize: "0.62rem",
                    background: "rgba(255,255,255,0.04)", padding: "0.15rem 0.4rem",
                    borderRadius: 4, border: "1px solid var(--border-dim)",
                  }}>
                    #{r.id}
                  </span>
                </td>
                <td style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {r.filename}
                </td>
                <td>{r.location}</td>
                <td>
                  <span style={{
                    display: "inline-block", padding: "0.15rem 0.5rem",
                    background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.18)",
                    borderRadius: 4, color: "#93c5fd", fontSize: "0.65rem",
                  }}>
                    {r.vehicleType}
                  </span>
                </td>
                <td>
                  <span className={`rp-conf-badge ${r.confidence > 70 ? "high" : r.confidence > 40 ? "mid" : "low"}`}>
                    {r.confidence}%
                  </span>
                </td>
                <td style={{ color: "var(--text-dim)", whiteSpace: "nowrap" }}>{r.dateTime}</td>
                <td style={{ textAlign: "center" }}>{r.processingTime}s</td>
                <td style={{ textAlign: "center" }}>
                  {r.processed ? (
                    <span style={{ color: "var(--accent-green)", fontSize: "0.7rem", fontWeight: 700 }}>✓</span>
                  ) : (
                    <span style={{ color: "var(--accent-red)", fontSize: "0.7rem", fontWeight: 700 }}>✗</span>
                  )}
                </td>
                <td><code>{r.coordinates}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}