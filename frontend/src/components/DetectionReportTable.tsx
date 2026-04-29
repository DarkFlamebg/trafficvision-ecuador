import type { DetectionReport } from "../types/readplate.types"

interface DetectionReportTableProps {
  report:      DetectionReport[]
  onDownload:  () => void
}

export function DetectionReportTable({ report, onDownload }: DetectionReportTableProps) {
  if (report.length === 0) return null
  return (
    <div className="rp-report-section">
      <div className="rp-section-label">REPORTE DE DETECCIÓN</div>
      <div className="rp-report-header">
        <h3>Detección Actual</h3>
        <button className="rp-btn-download-csv" onClick={onDownload}>
          ⬇ Descargar CSV
        </button>
      </div>
      <div className="rp-report-table-wrap">
        <table className="rp-report-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Nombre de Archivo</th>
              <th>Ubicación</th>
              <th>Tipo de Vehículo</th>
              <th>Confianza (%)</th>
              <th>Fecha/Hora</th>
              <th>Tiempo Proc. (s)</th>
              <th>Procesado</th>
              <th>Coordenadas</th>
            </tr>
          </thead>
          <tbody>
            {report.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.filename}</td>
                <td>{r.location}</td>
                <td>{r.vehicleType}</td>
                <td>
                  <span className={`rp-conf-badge ${r.confidence > 70 ? "high" : r.confidence > 40 ? "mid" : "low"}`}>
                    {r.confidence}%
                  </span>
                </td>
                <td>{r.dateTime}</td>
                <td>{r.processingTime}s</td>
                <td>{r.processed ? "✅" : "❌"}</td>
                <td><code>{r.coordinates}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
