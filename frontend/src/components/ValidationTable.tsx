// components/ValidationTable.tsx
// Tabla de validación con veredicto automático por score combinado
// Score = conf_detector * 0.4 + conf_ocr * 0.4 + velocidad_normalizada * 0.2

import type { ComparisonResult, ComparisonImageResponse, ComparisonMetrics } from "../types/comparison_types"

interface ValidationTableProps {
  results:    ComparisonResult
  realPlate?: string
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function getMetrics(data?: ComparisonResult["yolo"] | ComparisonResult["rtdetr"]): ComparisonMetrics | null {
  if (!data) return null
  if ("metrics" in data) return (data as ComparisonImageResponse).metrics
  return data as ComparisonMetrics
}

function getBestPlate(data?: ComparisonResult["yolo"] | ComparisonResult["rtdetr"]): string {
  if (!data) return "—"
  if ("plates" in data) {
    const plates = (data as ComparisonImageResponse).plates
    if (plates && plates.length > 0) return plates[0].plate || "No Detectada Correctamente"
  }
  return "—"
}

function getBestOcrConf(data?: ComparisonResult["yolo"] | ComparisonResult["rtdetr"]): number {
  if (!data) return 0
  if ("plates" in data) {
    const plates = (data as ComparisonImageResponse).plates
    if (plates && plates.length > 0) return plates[0].ocr_confidence
  }
  return 0
}

// Score combinado: peso detector (40%) + peso OCR (40%) + velocidad (20%)
function computeScore(
  metrics: ComparisonMetrics | null,
  ocrConf: number,
  allInferenceTimes: number[],
): number {
  if (!metrics) return 0

  const detConf  = metrics.avg_plate_confidence ?? 0
  const maxTime  = Math.max(...allInferenceTimes, 1)
  const infTime  = metrics.avg_inference_ms ?? metrics.inference_ms ?? maxTime
  // velocidad normalizada: menor tiempo → mayor score
  const speedScore = maxTime > 0 ? (maxTime - infTime) / maxTime : 0

  return detConf * 0.4 + ocrConf * 0.4 + speedScore * 0.2
}

// ── Componente ─────────────────────────────────────────────────────────────────

export function ValidationTable({ results, realPlate }: ValidationTableProps) {
  const yoloMetrics   = getMetrics(results.yolo)
  const rtdetrMetrics = getMetrics(results.rtdetr)
  const yoloPlate     = getBestPlate(results.yolo)
  const rtdetrPlate   = getBestPlate(results.rtdetr)
  const yoloOcr       = getBestOcrConf(results.yolo)
  const rtdetrOcr     = getBestOcrConf(results.rtdetr)

  // Tiempos para normalizar velocidad
  const yoloTime   = yoloMetrics?.avg_inference_ms   ?? yoloMetrics?.inference_ms   ?? 0
  const rtdetrTime = rtdetrMetrics?.avg_inference_ms ?? rtdetrMetrics?.inference_ms ?? 0
  const allTimes   = [yoloTime, rtdetrTime].filter(Boolean)

  const yoloScore   = computeScore(yoloMetrics,   yoloOcr,   allTimes)
  const rtdetrScore = computeScore(rtdetrMetrics, rtdetrOcr, allTimes)

  // Veredicto
  const bothReady  = yoloMetrics && rtdetrMetrics
  const winnerKey  = bothReady
    ? (yoloScore >= rtdetrScore ? "yolo" : "rtdetr")
    : null
  const winnerLabel = winnerKey === "yolo" ? "YOLOv11n" : "RT-DETR"
  const winnerColor = winnerKey === "yolo" ? "var(--yolo)" : "var(--rtdetr)"

  const rows = [
    {
      key:       "yolo",
      label:     "YOLOv11n",
      color:     "var(--yolo)",
      metrics:   yoloMetrics,
      plate:     yoloPlate,
      ocrConf:   yoloOcr,
      score:     yoloScore,
      infTime:   yoloTime,
    },
    {
      key:       "rtdetr",
      label:     "RT-DETR",
      color:     "var(--rtdetr)",
      metrics:   rtdetrMetrics,
      plate:     rtdetrPlate,
      ocrConf:   rtdetrOcr,
      score:     rtdetrScore,
      infTime:   rtdetrTime,
    },
  ]

  return (
    <div className="validation-root">

      {/* ── Input placa real ───────────────────────────────────────────── */}
      {/* (renderizado por el padre — aquí solo lo mostramos si fue provisto) */}
      {realPlate && (
        <div className="validation-real-chip">
          <span className="validation-real-label">PLACA REAL</span>
          <span className="validation-real-value">{realPlate}</span>
        </div>
      )}

      {/* ── Tabla de resultados ────────────────────────────────────────── */}
      <div className="validation-table-wrap">
        <div className="validation-table-title">VALIDACIÓN DE RESULTADOS</div>

        <div className="validation-table-scroll">
          <table className="validation-table">
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Placa detectada</th>
                <th>Conf. detector</th>
                <th>Conf. OCR</th>
                <th>Inf. (ms)</th>
                <th>Score</th>
                {realPlate && <th>vs. placa real</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isWinner  = row.key === winnerKey
                const isCorrect = realPlate && row.plate === realPlate
                const detConf   = row.metrics?.avg_plate_confidence ?? 0

                return (
                  <tr
                    key={row.key}
                    className={isWinner ? "validation-row--winner" : ""}
                  >
                    <td>
                      <span className="validation-model-tag"
                        style={{ color: row.color, borderColor: row.color + "55", background: row.color + "0f" }}>
                        {isWinner && <span className="validation-crown">▲</span>}
                        {row.label}
                      </span>
                    </td>

                    <td>
                      <span className="validation-plate-text" style={{ color: row.color }}>
                        {row.plate}
                      </span>
                    </td>

                    <td>
                      <ConfBar value={detConf} color={row.color} />
                    </td>

                    <td>
                      <ConfBar value={row.ocrConf} color={row.color} />
                    </td>

                    <td>
                      <span className="validation-mono">
                        {row.infTime > 0 ? `${row.infTime.toFixed(1)}ms` : "—"}
                      </span>
                    </td>

                    <td>
                      <ScorePill score={row.score} isWinner={isWinner} color={row.color} />
                    </td>

                    {realPlate && (
                      <td>
                        <span className={`validation-match ${isCorrect ? "validation-match--ok" : "validation-match--fail"}`}>
                          {row.plate === "—" ? "—" : isCorrect ? "✓ Correcto" : "✗ Incorrecto"}
                        </span>
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Veredicto ──────────────────────────────────────────────────── */}
      {bothReady && (
        <div className="validation-verdict" style={{ borderColor: winnerColor + "33", flexDirection: 'column', alignItems: 'stretch' }}>
          <div className="validation-verdict-top">
            <div className="validation-verdict-left">
              <div className="validation-verdict-crown" style={{ color: winnerColor }}>
                <svg width="22" height="22" viewBox="0 0 18 18" fill="none">
                  <path
                    d="M2 13L4.5 6L8 10L9 4L10 10L13.5 6L16 13H2Z"
                    fill="currentColor"
                  />
                </svg>
              </div>
              <div>
                <div className="validation-verdict-title" style={{ color: winnerColor }}>
                  {winnerLabel}
                </div>
                <div className="validation-verdict-sub">
                  MEJOR MODELO · SCORE {(Math.max(yoloScore, rtdetrScore) * 100).toFixed(0)}/100
                </div>
              </div>
            </div>

            <div className="validation-verdict-breakdown">
              <ScoreBreakdown
                label="Conf. detector"
                yolo={yoloMetrics?.avg_plate_confidence ?? 0}
                rtdetr={rtdetrMetrics?.avg_plate_confidence ?? 0}
                format={(v) => `${(v * 100).toFixed(0)}%`}
                winnerKey={winnerKey}
              />
              <ScoreBreakdown
                label="Conf. OCR"
                yolo={yoloOcr}
                rtdetr={rtdetrOcr}
                format={(v) => `${(v * 100).toFixed(0)}%`}
                winnerKey={winnerKey}
              />
              <ScoreBreakdown
                label="Velocidad"
                yolo={yoloTime}
                rtdetr={rtdetrTime}
                format={(v) => `${v.toFixed(1)}ms`}
                winnerKey={winnerKey}
                lowerBetter
              />
            </div>
          </div>

          {/* ── Resumen Simplificado (Aesthetics Polish) ── */}
          <div className="validation-summary-box">
            <header className="validation-summary-header">
              <div className="validation-summary-icon">💡</div>
              <h4 className="validation-summary-title">Resumen</h4>
            </header>
            <div className="validation-summary-content">
              <p className="summary-intro">
                En esta prueba, <strong style={{ color: winnerColor }}>{winnerLabel}</strong> fue coronado como el mejor modelo general, obteniendo una calificación de <strong>{(Math.max(yoloScore, rtdetrScore) * 100).toFixed(0)} sobre 100</strong>. Aquí te desglosamos el porqué:
              </p>
              <div className="summary-grid">
                <div className="summary-item">
                  <div className="summary-item-label">Vista (Detector)</div>
                  <p className="summary-item-text">
                    {(() => {
                      const yoloDet = yoloMetrics?.avg_plate_confidence ?? 0;
                      const rtDet = rtdetrMetrics?.avg_plate_confidence ?? 0;
                      if (yoloDet === rtDet) return 'Ambos modelos tuvieron exactamente la misma seguridad al encontrar la placa.';
                      const winner = yoloDet > rtDet ? 'YOLOv11n' : 'RT-DETR';
                      const loser = yoloDet > rtDet ? 'RT-DETR' : 'YOLOv11n';
                      const wScore = Math.max(yoloDet, rtDet) * 100;
                      const lScore = Math.min(yoloDet, rtDet) * 100;
                      return `${winner} estuvo más seguro (${wScore.toFixed(0)}%) superando a ${loser} (${lScore.toFixed(0)}%) al señalar la placa.`;
                    })()}
                  </p>
                </div>
                <div className="summary-item">
                  <div className="summary-item-label">Lectura (OCR)</div>
                  <p className="summary-item-text">
                    {(() => {
                      if (yoloOcr === 0 && rtdetrOcr === 0) return 'Tarea difícil: ninguno logró leer nada (0%) en esta imagen.';
                      if (yoloOcr === rtdetrOcr) return `Empate técnico: ambos leyeron con ${(yoloOcr * 100).toFixed(0)}% de confianza.`;
                      const winner = yoloOcr > rtdetrOcr ? 'YOLOv11n' : 'RT-DETR';
                      const loser = yoloOcr > rtdetrOcr ? 'RT-DETR' : 'YOLOv11n';
                      const wScore = Math.max(yoloOcr, rtdetrOcr) * 100;
                      return `${winner} leyó mejor los caracteres (${wScore.toFixed(0)}%) que el modelo rival.`;
                    })()}
                  </p>
                </div>
                <div className="summary-item">
                  <div className="summary-item-label">Cerebro (Velocidad)</div>
                  <p className="summary-item-text">
                    {(() => {
                      if (yoloTime === rtdetrTime) return 'Ambos procesaron a la misma velocidad.';
                      const winner = yoloTime < rtdetrTime ? 'YOLOv11n' : 'RT-DETR';
                      const fast = Math.min(yoloTime, rtdetrTime);
                      const slow = Math.max(yoloTime, rtdetrTime);
                      const ratio = (slow / Math.max(fast, 1)).toFixed(1);
                      return `${winner} fue increíblemente rápido (${fast.toFixed(0)}ms), siendo ${ratio}x más veloz que la competencia.`;
                    })()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Átomos ─────────────────────────────────────────────────────────────────────

function ConfBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="validation-conf-wrap">
      <div className="validation-conf-track">
        <div className="validation-conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="validation-mono">{pct}%</span>
    </div>
  )
}

function ScorePill({ score, isWinner, color }: { score: number; isWinner: boolean; color: string }) {
  const pct = Math.round(score * 100)
  return (
    <span
      className={`validation-score-pill ${isWinner ? "validation-score-pill--winner" : ""}`}
      style={{ color, borderColor: color + "55", background: color + "0f" }}
    >
      {pct}
    </span>
  )
}

function ScoreBreakdown({
  label, yolo, rtdetr, format, winnerKey, lowerBetter = false,
}: {
  label: string
  yolo: number
  rtdetr: number
  format: (v: number) => string
  winnerKey: string | null
  lowerBetter?: boolean
}) {
  const yoloWins   = lowerBetter ? yolo < rtdetr : yolo > rtdetr
  const rtdetrWins = lowerBetter ? rtdetr < yolo  : rtdetr > yolo

  return (
    <div className="validation-breakdown-row">
      <span
        className={`validation-breakdown-val ${yoloWins ? "validation-breakdown-val--win" : ""}`}
        style={{ color: yoloWins ? "var(--yolo)" : "var(--text-hi)" }}
      >
        {format(yolo)}
        {yoloWins && <span className="validation-breakdown-arrow">▲</span>}
      </span>
      <span className="validation-breakdown-label">{label}</span>
      <span
        className={`validation-breakdown-val ${rtdetrWins ? "validation-breakdown-val--win" : ""}`}
        style={{ color: rtdetrWins ? "var(--rtdetr)" : "var(--text-hi)" }}
      >
        {format(rtdetr)}
        {rtdetrWins && <span className="validation-breakdown-arrow">▲</span>}
      </span>
    </div>
  )
}