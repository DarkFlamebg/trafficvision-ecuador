// components/ValidationTable.tsx
// Tabla de validación con veredicto automático por score combinado
// Score = conf_detector * 0.4 + conf_ocr * 0.4 + velocidad_normalizada * 0.2

import type { ComparisonResult, ComparisonImageResponse, ComparisonMetrics } from "../types/comparison_types"

interface ValidationTableProps {
  results:    ComparisonResult
  realPlate?: string
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function getMetrics(data?: any): ComparisonMetrics | null {
  if (!data) return null
  if ("metrics" in data) return (data as ComparisonImageResponse).metrics
  return data as ComparisonMetrics
}

function getBestPlate(data?: any): string {
  if (!data) return "—"
  if ("plates" in data) {
    const plates = (data as ComparisonImageResponse).plates
    if (plates && plates.length > 0) return plates[0].plate || "No Detectada Correctamente"
  }
  return "—"
}

function getBestOcrConf(data?: any): number {
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
  const mambaMetrics = getMetrics(results.mamba)
  
  const yoloPlate     = getBestPlate(results.yolo)
  const rtdetrPlate   = getBestPlate(results.rtdetr)
  const mambaPlate = getBestPlate(results.mamba)
  
  const yoloOcr       = getBestOcrConf(results.yolo)
  const rtdetrOcr     = getBestOcrConf(results.rtdetr)
  const mambaOcr = getBestOcrConf(results.mamba)

  // Tiempos para normalizar velocidad
  const yoloTime   = yoloMetrics?.avg_inference_ms   ?? yoloMetrics?.inference_ms   ?? 0
  const rtdetrTime = rtdetrMetrics?.avg_inference_ms ?? rtdetrMetrics?.inference_ms ?? 0
  const mambaTime = mambaMetrics?.avg_inference_ms ?? mambaMetrics?.inference_ms ?? 0
  const allTimes   = [yoloTime, rtdetrTime, mambaTime].filter(Boolean)

  const yoloScore   = computeScore(yoloMetrics,   yoloOcr,   allTimes)
  const rtdetrScore = computeScore(rtdetrMetrics, rtdetrOcr, allTimes)
  const mambaScore = computeScore(mambaMetrics, mambaOcr, allTimes)

  // Veredicto
  const allReady  = yoloMetrics && rtdetrMetrics && mambaMetrics
  
  let winnerKey: string | null = null
  if (allReady) {
    const maxScore = Math.max(yoloScore, rtdetrScore, mambaScore)
    if (maxScore === yoloScore) winnerKey = "yolo"
    else if (maxScore === rtdetrScore) winnerKey = "rtdetr"
    else winnerKey = "mamba"
  }
  const winnerLabel = winnerKey === "yolo" ? "YOLOv11n" : winnerKey === "rtdetr" ? "RT-DETR" : "Vision Mamba"
  const winnerColor = winnerKey === "yolo" ? "var(--yolo)" : winnerKey === "rtdetr" ? "var(--rtdetr)" : "var(--mamba)"

  const maxTotalScore = Math.max(yoloScore, rtdetrScore, mambaScore)

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
    {
      key:       "mamba",
      label:     "Vision Mamba",
      color:     "var(--mamba)",
      metrics:   mambaMetrics,
      plate:     mambaPlate,
      ocrConf:   mambaOcr,
      score:     mambaScore,
      infTime:   mambaTime,
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
      {allReady && (
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
                  MEJOR MODELO · SCORE {(maxTotalScore * 100).toFixed(0)}/100
                </div>
              </div>
            </div>

            <div className="validation-verdict-breakdown">
              <ScoreBreakdown
                label="Conf. detector"
                yolo={yoloMetrics?.avg_plate_confidence ?? 0}
                rtdetr={rtdetrMetrics?.avg_plate_confidence ?? 0}
                mamba={mambaMetrics?.avg_plate_confidence ?? 0}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <ScoreBreakdown
                label="Conf. OCR"
                yolo={yoloOcr}
                rtdetr={rtdetrOcr}
                mamba={mambaOcr}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <ScoreBreakdown
                label="Velocidad"
                yolo={yoloTime}
                rtdetr={rtdetrTime}
                mamba={mambaTime}
                format={(v) => `${v.toFixed(1)}ms`}
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
                En esta prueba, <strong style={{ color: winnerColor }}>{winnerLabel}</strong> fue coronado como el mejor modelo general, obteniendo una calificación de <strong>{(maxTotalScore * 100).toFixed(0)} sobre 100</strong>. Aquí te desglosamos el porqué:
              </p>
              <div className="summary-grid">
                <div className="summary-item">
                  <div className="summary-item-label">Vista (Detector)</div>
                  <p className="summary-item-text">
                    {(() => {
                      const yoloDet = yoloMetrics?.avg_plate_confidence ?? 0;
                      const rtDet = rtdetrMetrics?.avg_plate_confidence ?? 0;
                      const effDet = mambaMetrics?.avg_plate_confidence ?? 0;
                      const arr = [
                        { name: 'YOLOv11n', val: yoloDet },
                        { name: 'RT-DETR', val: rtDet },
                        { name: 'Vision Mamba', val: effDet }
                      ].sort((a,b) => b.val - a.val);
                      if (arr[0].val === arr[2].val && arr[0].val > 0) return 'Los modelos tuvieron exactamente la misma seguridad al encontrar la placa.';
                      return `${arr[0].name} estuvo más seguro (${(arr[0].val*100).toFixed(0)}%) superando a los demás al señalar la placa.`;
                    })()}
                  </p>
                </div>
                <div className="summary-item">
                  <div className="summary-item-label">Lectura (OCR)</div>
                  <p className="summary-item-text">
                    {(() => {
                      if (yoloOcr === 0 && rtdetrOcr === 0 && mambaOcr === 0) return 'Tarea difícil: ninguno logró leer nada (0%) en esta imagen.';
                      const arr = [
                        { name: 'YOLOv11n', val: yoloOcr },
                        { name: 'RT-DETR', val: rtdetrOcr },
                        { name: 'Vision Mamba', val: mambaOcr }
                      ].sort((a,b) => b.val - a.val);
                      return `${arr[0].name} lideró la lectura de los caracteres (${(arr[0].val*100).toFixed(0)}%) en esta imagen.`;
                    })()}
                  </p>
                </div>
                <div className="summary-item">
                  <div className="summary-item-label">Cerebro (Velocidad)</div>
                  <p className="summary-item-text">
                    {(() => {
                      const validTimes = [
                        { name: 'YOLOv11n', val: yoloTime },
                        { name: 'RT-DETR', val: rtdetrTime },
                        { name: 'Vision Mamba', val: mambaTime }
                      ].filter(t => t.val > 0).sort((a,b) => a.val - b.val);
                      if (validTimes.length === 0) return 'No se registraron tiempos de inferencia válidos.';
                      const fast = validTimes[0];
                      const slow = validTimes[validTimes.length - 1];
                      const ratio = (slow.val / Math.max(fast.val, 1)).toFixed(1);
                      return `${fast.name} fue el más rápido (${fast.val.toFixed(0)}ms), siendo ${ratio}x más veloz que el modelo más lento.`;
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
  label, yolo, rtdetr, mamba, format, lowerBetter = false,
}: {
  label: string
  yolo: number
  rtdetr: number
  mamba: number
  format: (v: number) => string
  lowerBetter?: boolean
}) {
  const validVals = [yolo, rtdetr, mamba].filter(v => v > 0 || !lowerBetter)
  const bestVal = validVals.length > 0 ? (lowerBetter ? Math.min(...validVals) : Math.max(...validVals)) : 0

  const yoloWins   = yolo === bestVal && yolo !== 0
  const rtdetrWins = rtdetr === bestVal && rtdetr !== 0
  const effWins    = mamba === bestVal && mamba !== 0

  return (
    <div className="validation-breakdown-row" style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-start', alignItems: 'center' }}>
      <span className="validation-breakdown-label" style={{ minWidth: '100px' }}>{label}</span>
      <div style={{ display: 'flex', gap: '0.75rem', flex: 1 }}>
        <span
          className={`validation-breakdown-val ${yoloWins ? "validation-breakdown-val--win" : ""}`}
          style={{ color: yoloWins ? "var(--yolo)" : "var(--text-hi)", flex: 1, textAlign: 'center' }}
        >
          {format(yolo)}
        </span>
        <span
          className={`validation-breakdown-val ${rtdetrWins ? "validation-breakdown-val--win" : ""}`}
          style={{ color: rtdetrWins ? "var(--rtdetr)" : "var(--text-hi)", flex: 1, textAlign: 'center' }}
        >
          {format(rtdetr)}
        </span>
        <span
          className={`validation-breakdown-val ${effWins ? "validation-breakdown-val--win" : ""}`}
          style={{ color: effWins ? "var(--mamba)" : "var(--text-hi)", flex: 1, textAlign: 'center' }}
        >
          {format(mamba)}
        </span>
      </div>
    </div>
  )
}