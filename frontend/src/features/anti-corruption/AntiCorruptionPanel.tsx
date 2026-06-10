import { useEffect, useState } from "react"
import {
  downloadDetectionReport,
  getDetectionRows,
  submitDetectionFeedback,
  type AntiCorruptionFilters,
  type DetectionRow,
  type ModelOption,
} from "../../services/antiCorruptionService"
import "./AntiCorruptionPanel.css"

const validationOptions = [
  { value: "", label: "Todos" },
  { value: "validated", label: "Validados" },
  { value: "pending", label: "Pendientes" },
]

const defaultSummary = {
  total_detections: 0,
  validated: 0,
  pending: 0,
  incorrect: 0,
  avg_confidence: 0,
}

export default function AntiCorruptionPanel() {
  const [filters, setFilters] = useState<AntiCorruptionFilters>({ limit: 10, offset: 0 })
  const [models, setModels] = useState<ModelOption[]>([])
  const [detections, setDetections] = useState<DetectionRow[]>([])
  const [summary, setSummary] = useState(defaultSummary)
  const [queryLabel, setQueryLabel] = useState("Sin filtros")
  const [editingDetectionId, setEditingDetectionId] = useState<number | null>(null)
  const [correctionText, setCorrectionText] = useState("")
  const [actionLoadingId, setActionLoadingId] = useState<number | null>(null)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalEntries, setTotalEntries] = useState(0)
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDetections()
  }, [])

  function buildQueryLabel(activeFilters: AntiCorruptionFilters) {
    const parts: string[] = []
    if (activeFilters.plate) parts.push(`placa: "${activeFilters.plate}"`)
    if (activeFilters.validated) parts.push(`estado: ${activeFilters.validated}`)
    if (activeFilters.model_id) {
      const model = models.find((item) => item.id === activeFilters.model_id)
      parts.push(`modelo: ${model?.name ?? activeFilters.model_id}`)
    }
    return parts.length > 0 ? parts.join(" · ") : "Sin filtros"
  }

  async function fetchDetections(newFilters: AntiCorruptionFilters = {}, nextPage = 1) {
    const limit = filters.limit ?? 10
    const offset = (nextPage - 1) * limit
    const activeFilters = { ...filters, ...newFilters, limit, offset }

    setLoading(true)
    setMessage(null)
    setError(null)

    try {
      const response = await getDetectionRows(activeFilters)
      setModels(response.models)
      setDetections(response.detections)
      setSummary(response.summary)
      setTotalEntries(response.total)
      setTotalPages(Math.max(1, Math.ceil(response.total / limit)))
      setPage(nextPage)
      setFilters(activeFilters)
      setQueryLabel(buildQueryLabel(activeFilters))
    } catch {
      setError("No se pudo cargar los datos. Verifica que el backend esté activo.")
    } finally {
      setLoading(false)
    }
  }

  async function handleDownloadReport() {
    setDownloading(true)
    setMessage(null)
    setError(null)
    try {
      await downloadDetectionReport(filters)
      setMessage("CSV descargado correctamente.")
    } catch {
      setError("No se pudo descargar el reporte. Verifica que el backend esté activo.")
    } finally {
      setDownloading(false)
    }
  }

  async function submitReview(
    detectionId: number,
    isCorrect: boolean,
    correctedPlateText?: string
  ) {
    setActionLoadingId(detectionId)
    setMessage(null)
    setError(null)
    try {
      await submitDetectionFeedback({
        detection_id: detectionId,
        is_correct: isCorrect,
        corrected_plate_text: correctedPlateText || null,
        comments: isCorrect ? "Validación auditoría: correcta" : "Validación auditoría: incorrecta",
      })
      setMessage("Validación guardada correctamente.")
      setEditingDetectionId(null)
      setCorrectionText("")
      await fetchDetections()
    } catch {
      setError("No se pudo guardar la validación. Verifica que el backend esté activo.")
    } finally {
      setActionLoadingId(null)
    }
  }

  const visibleDetections = detections

  return (
    <main className="ac-panel">
      {/* ── Hero ── */}
      <section className="ac-hero">
        <h1>Control anticorrupción</h1>
        <p>
          Trazabilidad, validaciones de usuario y registros de detección — TrafficVision
        </p>
      </section>

      {/* ── Action cards ── */}
      <p className="ac-section-label">Acciones rápidas</p>
      <section className="ac-actions" aria-label="Acciones de control anticorrupción">
        <article className="ac-action-card">
          <div>
            <span className="ac-action-tag">Reportes</span>
            <h2>Filtra y descarga detecciones</h2>
            <p>Descarga un CSV con los registros filtrados en tiempo real.</p>
          </div>
          <button
            className="ac-action-btn"
            type="button"
            onClick={handleDownloadReport}
            disabled={downloading || loading}
          >
            {downloading ? "Descargando…" : "Descargar CSV"}
          </button>
        </article>

        <article className="ac-action-card">
          <div>
            <span className="ac-action-tag">Auditoría</span>
            <h2>Revisión de validaciones</h2>
            <p>
              Marca detecciones como correctas o incorrectas y aplica correcciones
              directamente desde la base.
            </p>
          </div>
          <button
            className="ac-action-btn"
            type="button"
            onClick={() => fetchDetections({ validated: "pending" })}
            disabled={loading}
          >
            {loading ? "Cargando…" : "Ver pendientes"}
          </button>
        </article>
      </section>

      {/* ── Filters ── */}
      <section className="ac-filters" aria-label="Filtros">
        <div className="ac-filter-row">
          <label>
            Placa
            <input
              value={filters.plate ?? ""}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, plate: e.target.value || undefined }))
              }
              placeholder="Buscar por placa"
            />
          </label>

          <label>
            Estado
            <select
              value={filters.validated ?? ""}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, validated: e.target.value || undefined }))
              }
            >
              {validationOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Modelo
            <select
              value={filters.model_id ?? ""}
              onChange={(e) => {
                const id = Number(e.target.value)
                setFilters((prev) => ({
                  ...prev,
                  model_id: Number.isNaN(id) ? undefined : id,
                }))
              }}
            >
              <option value="">Todos</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          <button
            className="ac-action-btn ac-action-btn--secondary"
            type="button"
            onClick={() => fetchDetections()}
            disabled={loading}
          >
            {loading ? "Cargando…" : "Aplicar filtros"}
          </button>
        </div>
      </section>

      {/* ── Summary ── */}
      <section className="ac-summary" aria-label="Resumen">
        <div className="ac-summary-card">
          <strong>{summary.total_detections}</strong>
          <span>Total detecciones</span>
        </div>
        <div className="ac-summary-card">
          <strong>{summary.validated}</strong>
          <span>Validadas</span>
        </div>
        <div className="ac-summary-card">
          <strong>{summary.pending}</strong>
          <span>Pendientes</span>
        </div>
        <div className="ac-summary-card">
          <strong>{summary.incorrect}</strong>
          <span>Incorrectas</span>
        </div>
        <div className="ac-summary-card">
          <strong>{summary.avg_confidence.toFixed(1)}%</strong>
          <span>Confianza promedio</span>
        </div>
      </section>

      {/* ── Query meta ── */}
      <div className="ac-query-meta">
        <p>
          <strong>Consulta actual:</strong> {queryLabel}
        </p>
        <p>
          Mostrando <strong>{Math.min((filters.offset ?? 0) + detections.length, totalEntries)}</strong> de <strong>{totalEntries}</strong> registros
        </p>
      </div>

      <div className="ac-pagination" aria-label="Paginación de detecciones">
        <button
          className="ac-action-btn ac-action-btn--small"
          type="button"
          onClick={() => fetchDetections({}, Math.max(page - 1, 1))}
          disabled={page <= 1 || loading}
        >
          Anterior
        </button>
        <span className="ac-pagination-label">
          Página {page} de {totalPages}
        </span>
        <button
          className="ac-action-btn ac-action-btn--small"
          type="button"
          onClick={() => fetchDetections({}, Math.min(page + 1, totalPages))}
          disabled={page >= totalPages || loading}
        >
          Siguiente
        </button>
      </div>

      {/* ── Table ── */}
      <section className="ac-table-wrapper" aria-label="Registros de detección">
        <table className="ac-table">
          <colgroup>
            <col style={{ width: "14%" }} />
            <col style={{ width: "10%" }} />
            <col style={{ width: "13%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "11%" }} />
            <col style={{ width: "18%" }} />
            <col style={{ width: "16%" }} />
          </colgroup>
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Placa</th>
              <th>Modelo</th>
              <th>Confianza</th>
              <th>Validado</th>
              <th>Corrección</th>
              <th>Calidad / Auditoría</th>
              <th>Acciones</th>
            </tr>
          </thead>
          <tbody>
            {visibleDetections.map((detection) => (
              <tr key={detection.id}>
                <td className="ac-cell--muted">
                  {new Date(detection.detection_date).toLocaleString("es-EC", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })}
                </td>

                <td>
                  <strong>{detection.plate_text}</strong>
                </td>

                <td className="ac-cell--muted">{detection.model || "—"}</td>

                <td className="ac-cell--mono">
                  {(detection.confidence * 100).toFixed(1)}%
                </td>

                <td>
                  {detection.user_validated ? (
                    <span className="ac-tag ac-tag--success">Validado</span>
                  ) : (
                    <span className="ac-tag ac-tag--pending">Pendiente</span>
                  )}
                </td>

                <td className="ac-cell--muted">
                  {detection.user_corrected_text || "—"}
                </td>

                <td className="ac-cell--muted" style={{ fontSize: 12 }}>
                  {detection.quality_checks.length > 0
                    ? detection.quality_checks
                        .map((c) => `${c.label}: ${c.value}`)
                        .join("; ")
                    : "—"}
                  {detection.audit_logs.length > 0 &&
                    ` · ${detection.audit_logs[0].check_reason ?? "Auditoría"}`}
                </td>

                <td>
                  {detection.user_validated ? (
                    <span className="ac-tag ac-tag--success">Validado</span>
                  ) : editingDetectionId === detection.id ? (
                    <div className="ac-review-row">
                      <input
                        value={correctionText}
                        onChange={(e) => setCorrectionText(e.target.value)}
                        placeholder="Texto corregido"
                      />
                      <button
                        className="ac-action-btn ac-action-btn--small"
                        type="button"
                        onClick={() => submitReview(detection.id, false, correctionText)}
                        disabled={actionLoadingId === detection.id}
                      >
                        Guardar
                      </button>
                      <button
                        className="ac-action-btn ac-action-btn--ghost ac-action-btn--small"
                        type="button"
                        onClick={() => {
                          setEditingDetectionId(null)
                          setCorrectionText("")
                        }}
                      >
                        Cancelar
                      </button>
                    </div>
                  ) : (
                    <div className="ac-review-actions">
                      <button
                        className="ac-action-btn ac-action-btn--small"
                        type="button"
                        onClick={() => submitReview(detection.id, true)}
                        disabled={actionLoadingId === detection.id}
                      >
                        Correcta
                      </button>
                      <button
                        className="ac-action-btn ac-action-btn--ghost ac-action-btn--small"
                        type="button"
                        onClick={() => {
                          setEditingDetectionId(detection.id)
                          setCorrectionText(detection.plate_text)
                        }}
                      >
                        Incorrecta
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}

            {visibleDetections.length === 0 && (
              <tr>
                <td colSpan={8}>
                  No hay detecciones que coincidan con los filtros actuales.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      {/* ── Status ── */}
      {(message || error) && (
        <div
          className={`ac-status ${error ? "ac-status--error" : "ac-status--ok"}`}
          role="status"
        >
          {error ?? message}
        </div>
      )}
    </main>
  )
}