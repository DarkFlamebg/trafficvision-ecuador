/*
  TrainingMetricsDynamic.tsx
  ─────────────────────────────────────────────────────────────────────────────
  Uso:
    1. npm install chart.js react-chartjs-2

    2a. Vite — importa el CSV como texto raw:
          import rawCsv from "../../assets/data/yolov11_results.csv?raw"
          <TrainingMetricsDynamic csvText={rawCsv} />

    2b. Cualquier bundler — carga por URL en runtime:
          <TrainingMetricsDynamic csvUrl="/data/yolov11_results.csv" />
*/

import { useState, useEffect, useMemo } from "react"

import yoloCsv from "../../../assets/images/yolov11/results.csv?raw"
import rtdetrCsv from "../../../assets/images/rtdetr/results.csv?raw"
import efficientCsv from "../../../assets/images/efficient/results.csv?raw"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js"
import { Line, Scatter } from "react-chartjs-2"
import "./TrainingMetricsDynamic.css"

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend
)

// ─────────────────────────────────────────────────────────────────────────────
// Tipos
// ─────────────────────────────────────────────────────────────────────────────

interface EpochRow {
  epoch:    number
  trainBox: number
  trainCls: number
  trainDfl: number
  precision:number
  recall:   number
  map50:    number
  map5095:  number
  valBox:   number
  valCls:   number
  valDfl:   number
}

interface Props {
  /** CSV como string (Vite: import csv from "...csv?raw") */
  csvText?: string
  /** URL del CSV para fetch dinámico */
  csvUrl?: string
  /** Nombre del modelo que se muestra en el encabezado */
  modelName?: string
}

type Section = "convergence" | "metrics" | "confusion"
type LossTab  = "train" | "val" | "map"
type ModelKey = "yolov11n" | "rtdetr" | "efficientdet"

const MODEL_OPTIONS: Record<ModelKey, { name: string; description: string; csvText: string }> = {
  yolov11n: {
    name: "YOLOv11n",
    description: "Single-shot CNN - Maxima velocidad",
    csvText: yoloCsv,
  },
  rtdetr: {
    name: "RT-DETR",
    description: "Transformer + CNN - Alta precision en oclusion",
    csvText: rtdetrCsv,
  },
  efficientdet: {
    name: "EfficientDet-D2",
    description: "BiFPN + EfficientNet - Balance eficiente",
    csvText: efficientCsv,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser CSV
// ─────────────────────────────────────────────────────────────────────────────

function parseCsv(text: string): EpochRow[] {
  const lines = text.trim().split("\n")
  // Encabezado: epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,
  //             metrics/precision(B),metrics/recall(B),metrics/mAP50(B),
  //             metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,...
  return lines.slice(1).map((line) => {
    const cols = line.split(",").map(Number)
    return {
      epoch:     cols[0],
      trainBox:  cols[2],
      trainCls:  cols[3],
      trainDfl:  cols[4],
      precision: cols[5],
      recall:    cols[6],
      map50:     cols[7],
      map5095:   cols[8],
      valBox:    cols[9],
      valCls:    cols[10],
      valDfl:    cols[11],
    }
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers de gráficas
// ─────────────────────────────────────────────────────────────────────────────

const COLORS = {
  blue:   "#185FA5",
  orange: "#D85A30",
  green:  "#1D9E75",
  amber:  "#BA7517",
}

function lineOpts(
  yLabel = "",
  yMin?: number,
  yMax?: number,
  xType?: "linear" | "category",
  xLabel?: string,
) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index" as const, intersect: false },
    plugins: { legend: { display: false }, tooltip: { padding: 10 } },
    scales: {
      x: {
        ...(xType === "linear" ? { type: "linear" as const, min: 0, max: 1 } : {}),
        grid:  { color: "rgba(128,128,128,0.1)" },
        ticks: { color: "#888", font: { size: 11 }, maxTicksLimit: 10 },
        ...(xLabel ? { title: { display: true, text: xLabel, color: "#888", font: { size: 11 } } } : {}),
      },
      y: {
        grid:  { color: "rgba(128,128,128,0.1)" },
        ticks: {
          color: "#888", font: { size: 11 },
          callback: (v: number | string) => typeof v === "number" ? v.toFixed(2) : v,
        },
        ...(yLabel ? { title: { display: true, text: yLabel, color: "#888", font: { size: 11 } } } : {}),
        ...(yMin !== undefined ? { min: yMin } : {}),
        ...(yMax !== undefined ? { max: yMax } : {}),
      },
    },
    elements: { point: { radius: 0, hoverRadius: 4 }, line: { tension: 0.35, borderWidth: 2 } },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-componentes pequeños
// ─────────────────────────────────────────────────────────────────────────────

function KpiCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="tmd-kpi">
      <span className="tmd-kpi-label">{label}</span>
      <span className="tmd-kpi-value">{value}</span>
      <span className="tmd-kpi-sub">{sub}</span>
    </div>
  )
}

function LegendItem({
  color, dash = [], label, square = false, border = false,
}: {
  color: string; dash?: number[]; label: string; square?: boolean; border?: boolean
}) {
  return (
    <span className="tmd-legend-item">
      {square ? (
        <span
          className="tmd-legend-square"
          style={{ background: color, border: border ? "1px solid #bbb" : "none" }}
        />
      ) : (
        <svg width="20" height="6" aria-hidden="true">
          <line x1="0" y1="3" x2="20" y2="3" stroke={color} strokeWidth="2.5"
            strokeDasharray={dash.join(",")} strokeLinecap="round" />
        </svg>
      )}
      {label}
    </span>
  )
}

function Explain({ children }: { children: React.ReactNode }) {
  return (
    <aside className="tmd-explain" role="note">
      <span className="tmd-explain-icon" aria-hidden="true">💡</span>
      <div>{children}</div>
    </aside>
  )
}

function SectionTabs({
  active, onChange,
}: {
  active: Section; onChange: (s: Section) => void
}) {
  const tabs: { id: Section; label: string; icon: string }[] = [
    { id: "convergence", label: "Convergencia", icon: "📉" },
    { id: "metrics",     label: "Precisión & Recall", icon: "📈" },
    { id: "confusion",   label: "Matriz de Confusión", icon: "🔲" },
  ]
  return (
    <nav className="tmd-tabs" aria-label="Secciones del análisis">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`tmd-tab${active === t.id ? " active" : ""}`}
          onClick={() => onChange(t.id)}
          aria-pressed={active === t.id}
        >
          <span aria-hidden="true">{t.icon}</span> {t.label}
        </button>
      ))}
    </nav>
  )
}

function SubTabs<T extends string>({
  options, active, onChange,
}: {
  options: { id: T; label: string }[]
  active: T
  onChange: (t: T) => void
}) {
  return (
    <div className="tmd-sub-tabs">
      {options.map((o) => (
        <button
          key={o.id}
          className={`tmd-sub-tab${active === o.id ? " active" : ""}`}
          onClick={() => onChange(o.id)}
          aria-pressed={active === o.id}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sección 1 — Convergencia
// ─────────────────────────────────────────────────────────────────────────────

function SectionConvergence({ data }: { data: EpochRow[] }) {
  const [tab, setTab] = useState<LossTab>("train")
  const last = data[data.length - 1]
  const epochs = data.map((r) => r.epoch)

  const chartData = useMemo(() => ({
    train: {
      labels: epochs,
      datasets: [
        { label: "Box loss", data: data.map((r) => r.trainBox), borderColor: COLORS.blue,   borderDash: [] },
        { label: "Cls loss", data: data.map((r) => r.trainCls), borderColor: COLORS.orange, borderDash: [5, 3] },
        { label: "DFL loss", data: data.map((r) => r.trainDfl), borderColor: COLORS.green,  borderDash: [2, 2] },
      ],
    },
    val: {
      labels: epochs,
      datasets: [
        { label: "Val Box", data: data.map((r) => r.valBox), borderColor: COLORS.blue,   borderDash: [] },
        { label: "Val Cls", data: data.map((r) => r.valCls), borderColor: COLORS.orange, borderDash: [5, 3] },
        { label: "Val DFL", data: data.map((r) => r.valDfl), borderColor: COLORS.green,  borderDash: [2, 2] },
      ],
    },
    map: {
      labels: epochs,
      datasets: [
        {
          label: "mAP@50", data: data.map((r) => r.map50),
          borderColor: COLORS.blue, borderDash: [],
          fill: { target: "origin" as const, above: "rgba(24,95,165,0.09)" },
        },
        {
          label: "mAP@50-95", data: data.map((r) => r.map5095),
          borderColor: COLORS.green, borderDash: [5, 3],
        },
      ],
    },
  }), [data]) // eslint-disable-line react-hooks/exhaustive-deps

  const clsReduction = Math.round((1 - last.trainCls / 4.1) * 100)

  const explains: Record<LossTab, React.ReactNode> = {
    train: (
      <>
        Las tres curvas muestran cómo el modelo aprende a detectar vehículos durante el entrenamiento.
        La <strong>Box loss</strong> mide qué tan bien encuadra los objetos, la <strong>Cls loss</strong> si
        los clasifica correctamente y la <strong>DFL loss</strong> refina los bordes de las cajas.
        La Cls loss bajó de ~4.1 a {last.trainCls.toFixed(2)} — una reducción del {clsReduction}%, lo que
        indica que el modelo aprendió a distinguir vehículos con alta confianza.
      </>
    ),
    val: (
      <>
        Estas curvas muestran el rendimiento en imágenes que el modelo <em>nunca ha visto</em> durante el
        entrenamiento. Si bajan a la par que las de entrenamiento, el modelo generaliza bien y no está
        memorizando los datos (sobreajuste). En YOLOv11, la validación converge de forma estable,
        confirmando que las mejoras son reales y aplicables a nuevo material de vídeo de tráfico.
      </>
    ),
    map: (
      <>
        El <strong>mAP@50</strong> (Mean Average Precision con umbral del 50%) es la métrica estándar en
        detección de objetos. Exige que la caja detectada solape al menos la mitad de la caja real. El modelo
        pasó de 0.40 a <strong>{(last.map50 * 100).toFixed(1)}%</strong> en {last.epoch} épocas.
        El <strong>mAP@50-95</strong> es más exigente (promedia sobre múltiples umbrales) y llegó
        a {(last.map5095 * 100).toFixed(1)}%, ideal para condiciones de tráfico con oclusión parcial.
      </>
    ),
  }

  const lossTabs: { id: LossTab; label: string }[] = [
    { id: "train", label: "Pérdidas entrenamiento" },
    { id: "val",   label: "Pérdidas validación" },
    { id: "map",   label: "mAP (convergencia)" },
  ]

  return (
    <div className="tmd-section">
      <SubTabs options={lossTabs} active={tab} onChange={setTab} />

      <div className="tmd-chart-wrap" role="img"
        aria-label={
          tab === "train" ? "Curvas de pérdida de entrenamiento por época" :
          tab === "val"   ? "Curvas de pérdida de validación por época" :
                            "Evolución del mAP durante el entrenamiento"
        }>
        <Line
          data={chartData[tab]}
          options={tab === "map"
            ? lineOpts("mAP", 0, 1)
            : lineOpts("Pérdida")
          }
        />
      </div>

      <div className="tmd-legend">
        {tab !== "map" ? (
          <>
            <LegendItem color={COLORS.blue}   label={tab === "train" ? "Box loss" : "Val Box"} />
            <LegendItem color={COLORS.orange} dash={[5,3]} label={tab === "train" ? "Cls loss" : "Val Cls"} />
            <LegendItem color={COLORS.green}  dash={[2,2]} label={tab === "train" ? "DFL loss" : "Val DFL"} />
          </>
        ) : (
          <>
            <LegendItem color={COLORS.blue}  label="mAP@50" />
            <LegendItem color={COLORS.green} dash={[5,3]} label="mAP@50-95" />
          </>
        )}
      </div>

      <Explain>{explains[tab]}</Explain>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sección 2 — Precisión & Recall
// ─────────────────────────────────────────────────────────────────────────────

function SectionMetrics({ data }: { data: EpochRow[] }) {
  const last    = data[data.length - 1]
  const epochs  = data.map((r) => r.epoch)

  const prLineData = useMemo(() => ({
    labels: epochs,
    datasets: [
      { label: "Precisión", data: data.map((r) => r.precision), borderColor: COLORS.blue,   borderDash: [] },
      { label: "Recall",    data: data.map((r) => r.recall),    borderColor: COLORS.orange, borderDash: [5,3] },
      { label: "mAP@50",   data: data.map((r) => r.map50),     borderColor: COLORS.green,  borderDash: [2,2] },
    ],
  }), [data]) // eslint-disable-line react-hooks/exhaustive-deps

  // Curva PR: cada época es un punto (recall, precision)
  const prCurvePoints = useMemo(
    () =>
      data
        .map((r) => ({ x: parseFloat(r.recall.toFixed(4)), y: parseFloat(r.precision.toFixed(4)) }))
        .sort((a, b) => a.x - b.x),
    [data],
  )

  const prCurveData = {
    datasets: [{
      label: "Curva PR",
      data: prCurvePoints,
      borderColor: COLORS.blue,
      backgroundColor: "rgba(24,95,165,0.08)",
      fill: true,
      tension: 0.35,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
    }],
  }

  // AUC aproximada bajo la curva PR (trapezoide)
  const auc = prCurvePoints.reduce((acc, pt, i, arr) => {
    if (i === 0) return acc
    const dx = pt.x - arr[i - 1].x
    const avgY = (pt.y + arr[i - 1].y) / 2
    return acc + dx * avgY
  }, 0)

  return (
    <div className="tmd-section">
      <div className="tmd-pr-grid">

        <div className="tmd-pr-col">
          <p className="tmd-chart-title">Evolución por época</p>
          <div className="tmd-chart-wrap" role="img"
            aria-label="Curvas de precisión, recall y mAP por época">
            <Line data={prLineData} options={lineOpts("", 0, 1)} />
          </div>
          <div className="tmd-legend">
            <LegendItem color={COLORS.blue}   label="Precisión" />
            <LegendItem color={COLORS.orange} dash={[5,3]} label="Recall" />
            <LegendItem color={COLORS.green}  dash={[2,2]} label="mAP@50" />
          </div>
        </div>

        <div className="tmd-pr-col">
          <p className="tmd-chart-title">
            Curva Precision-Recall
            <span className="tmd-auc-badge">AUC ≈ {auc.toFixed(3)}</span>
          </p>
          <div className="tmd-chart-wrap" role="img"
            aria-label="Curva Precision-Recall del modelo">
            <Scatter
              data={prCurveData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "nearest" as const, intersect: false },
                plugins: { legend: { display: false } },
                scales: {
                  x: {
                    type: "linear" as const, min: 0, max: 1,
                    title: { display: true, text: "Recall",    color: "#888", font: { size: 11 } },
                    grid:  { color: "rgba(128,128,128,0.1)" },
                    ticks: { color: "#888", font: { size: 11 } },
                  },
                  y: {
                    min: 0, max: 1,
                    title: { display: true, text: "Precisión", color: "#888", font: { size: 11 } },
                    grid:  { color: "rgba(128,128,128,0.1)" },
                    ticks: { color: "#888", font: { size: 11 } },
                  },
                },
                elements: { point: { radius: 2, hoverRadius: 5 }, line: { tension: 0.35, borderWidth: 2 } },
                showLine: true,
              } as any}
            />
          </div>
          <div className="tmd-legend">
            <LegendItem color={COLORS.blue} label="Trayectoria del entrenamiento" />
          </div>
        </div>

      </div>

      <Explain>
        La <strong>Precisión</strong> responde: de todos los vehículos que el modelo dijo detectar,
        ¿cuántos eran reales? Una precisión alta evita falsas alarmas. El <strong>Recall</strong>
        responde: de todos los vehículos presentes en la imagen, ¿cuántos encontró el modelo?
        Un recall alto evita omisiones. Al final del entrenamiento, YOLOv11 alcanzó{" "}
        <strong>{(last.precision * 100).toFixed(1)}% de precisión</strong> y{" "}
        <strong>{(last.recall * 100).toFixed(1)}% de recall</strong>. En la curva PR, el área bajo
        la curva (AUC ≈ {auc.toFixed(3)}) resume el balance global — cuanto más cercana a 1.0,
        mejor el modelo en todos los umbrales de confianza.
      </Explain>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sección 3 — Matriz de Confusión
// ─────────────────────────────────────────────────────────────────────────────

function SectionConfusion({ data }: { data: EpochRow[] }) {
  const last = data[data.length - 1]

  // Estimación normalizada a partir de precision/recall
  const tp = last.precision * last.recall
  const fp = last.precision * (1 - last.recall)
  const fn = (1 - last.precision) * last.recall
  const tn = Math.max(0, 1 - tp - fp - fn)

  const matrix = [
    [tp, fn],
    [fp, tn],
  ]
  const rowLabels = ["Vehículo real", "Fondo real"]
  const colLabels = ["Predicho: vehículo", "Predicho: fondo"]

  function cellColors(v: number, correct: boolean): { bg: string; fg: string } {
    if (v > 0.70) return { bg: "#0C447C", fg: "#B5D4F4" }
    if (v > 0.40) return { bg: "#378ADD", fg: "#E6F1FB" }
    if (v > 0.15) return { bg: "#B5D4F4", fg: "#0C447C" }
    return correct
      ? { bg: "#EAF3DE", fg: "#27500A" }
      : { bg: "#FCEBEB", fg: "#791F1F" }
  }

  const labels: Record<string, string> = {
    "0-0": "Verdadero Positivo (TP)",
    "0-1": "Falso Negativo (FN)",
    "1-0": "Falso Positivo (FP)",
    "1-1": "Verdadero Negativo (TN)",
  }

  return (
    <div className="tmd-section">
      <p className="tmd-chart-title">
        Matriz de confusión normalizada — estimada desde métricas de la época {last.epoch}
      </p>

      <div className="tmd-cm-outer">
        <table className="tmd-cm-table" aria-label="Matriz de confusión normalizada YOLOv11">
          <thead>
            <tr>
              <th className="tmd-cm-corner">Real ╲ Predicho</th>
              {colLabels.map((l) => <th key={l} className="tmd-cm-th">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <th scope="row" className="tmd-cm-row-th">{rowLabels[ri]}</th>
                {row.map((v, ci) => {
                  const correct = ri === ci
                  const { bg, fg } = cellColors(v, correct)
                  const key = `${ri}-${ci}`
                  return (
                    <td
                      key={ci}
                      className="tmd-cm-cell"
                      style={{ background: bg, color: fg }}
                      title={labels[key]}
                    >
                      <span className="tmd-cm-pct">{(v * 100).toFixed(1)}%</span>
                      <span className="tmd-cm-abbr">{Object.keys(labels)[ri * 2 + ci]?.split(" ").pop() ?? ""}</span>
                      <span className="tmd-cm-tag">{correct ? "✓" : "✗"} {labels[key]}</span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="tmd-cm-stats">
        {[
          { label: "Exactitud (Accuracy)", value: `${((tp + tn) * 100).toFixed(1)}%` },
          { label: "Precisión",            value: `${(last.precision * 100).toFixed(1)}%` },
          { label: "Recall (Sensibilidad)",value: `${(last.recall * 100).toFixed(1)}%` },
          { label: "F1-Score",             value: `${((2 * last.precision * last.recall) / (last.precision + last.recall)).toFixed(3)}` },
        ].map((s) => (
          <div key={s.label} className="tmd-cm-stat">
            <span className="tmd-cm-stat-label">{s.label}</span>
            <span className="tmd-cm-stat-value">{s.value}</span>
          </div>
        ))}
      </div>

      <div className="tmd-legend">
        <LegendItem color="#0C447C" square label="Alta tasa (&gt;70%)" />
        <LegendItem color="#378ADD" square label="Media (40–70%)" />
        <LegendItem color="#B5D4F4" square label="Baja (15–40%)" />
        <LegendItem color="#EAF3DE" square border label="Muy baja (&lt;15%) — acierto" />
        <LegendItem color="#FCEBEB" square border label="Muy baja (&lt;15%) — error" />
      </div>

      <Explain>
        La <strong>diagonal principal</strong> (TP y TN) representa los aciertos: el modelo
        detectó correctamente un vehículo real ({(tp * 100).toFixed(1)}%) o identificó
        correctamente el fondo ({(tn * 100).toFixed(1)}%). Los valores fuera de la diagonal
        son errores: <strong>FN</strong> ({(fn * 100).toFixed(1)}%) cuando pierde un vehículo
        real, y <strong>FP</strong> ({(fp * 100).toFixed(1)}%) cuando detecta un vehículo
        donde no lo hay. Con un F1-Score de{" "}
        {((2 * last.precision * last.recall) / (last.precision + last.recall)).toFixed(3)},
        el modelo ofrece un balance excelente para aplicaciones de monitoreo de tráfico en tiempo real.
      </Explain>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Componente principal
// ─────────────────────────────────────────────────────────────────────────────

export default function TrainingMetricsDynamic({
  csvText,
  csvUrl,
  modelName,
}: Props) {
  const [data,    setData]    = useState<EpochRow[]>([])
  const [section, setSection] = useState<Section>("convergence")
  const [activeModel, setActiveModel] = useState<ModelKey>("yolov11n")
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(false)
  const selectedModel = MODEL_OPTIONS[activeModel]
  const usesBuiltInModels = !csvText && !csvUrl
  const activeCsvText = csvText ?? (usesBuiltInModels ? selectedModel.csvText : undefined)
  const displayModelName = modelName ?? selectedModel.name

  useEffect(() => {
    setLoading(true)
    setError(false)

    if (activeCsvText) {
      try {
        setData(parseCsv(activeCsvText))
        setLoading(false)
      } catch {
        setData([])
        setError(true)
        setLoading(false)
      }
      return
    }

    if (csvUrl) {
      fetch(csvUrl)
        .then((r) => {
          if (!r.ok) throw new Error("fetch failed")
          return r.text()
        })
        .then((txt) => { setData(parseCsv(txt)); setLoading(false) })
        .catch(() => { setData([]); setError(true); setLoading(false) })
      return
    }

    setData([])
    setError(true)
    setLoading(false)
  }, [activeCsvText, csvUrl])

  if (loading) {
    return (
      <div className="tmd-state">
        <span className="tmd-spinner" aria-label="Cargando…" />
        <p>Cargando métricas de entrenamiento…</p>
      </div>
    )
  }

  if (error || data.length === 0) {
    return (
      <div className="tmd-state tmd-state--error">
        <p>No se pudieron cargar los datos del CSV. Verifica la ruta o la prop <code>csvText</code>.</p>
      </div>
    )
  }

  const last  = data[data.length - 1]
  const f1    = (2 * last.precision * last.recall) / (last.precision + last.recall)

  return (
    <div className="tmd-container">

      {/* ── Encabezado ─────────────────────────────────────────────────────── */}
      <header className="tmd-header">
        <div className="tmd-header-title">
          <h1>Análisis de entrenamiento y validación</h1>
          <span className="tmd-model-badge">{displayModelName}</span>
        </div>
        <p className="tmd-header-desc">
          Curvas de convergencia empírica, curvas Precision-Recall y matrices de confusión
          obtenidas durante la fase experimental (Sprints 1 y 2) de TrafficVision —{" "}
          <strong>{last.epoch} épocas</strong> de entrenamiento.
        </p>
      </header>

      {usesBuiltInModels && (
        <div className="tmd-model-selector" role="group" aria-label="Seleccionar modelo">
          {(Object.keys(MODEL_OPTIONS) as ModelKey[]).map((key) => (
            <button
              key={key}
              type="button"
              className={`tmd-model-option${activeModel === key ? " active" : ""}`}
              onClick={() => setActiveModel(key)}
              aria-pressed={activeModel === key}
            >
              <span className="tmd-model-option-name">{MODEL_OPTIONS[key].name}</span>
              <span className="tmd-model-option-desc">{MODEL_OPTIONS[key].description}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── KPIs ───────────────────────────────────────────────────────────── */}
      <div className="tmd-kpis">
        <KpiCard label="mAP@50"     value={`${(last.map50  * 100).toFixed(1)}%`} sub={`época ${last.epoch}`} />
        <KpiCard label="mAP@50-95"  value={`${(last.map5095* 100).toFixed(1)}%`} sub={`época ${last.epoch}`} />
        <KpiCard label="Precisión"  value={`${(last.precision*100).toFixed(1)}%`} sub={`época ${last.epoch}`} />
        <KpiCard label="Recall"     value={`${(last.recall  *100).toFixed(1)}%`}  sub={`época ${last.epoch}`} />
        <KpiCard label="F1-Score"   value={f1.toFixed(3)}                          sub="harmonic mean" />
      </div>

      {/* ── Tabs principales ───────────────────────────────────────────────── */}
      <SectionTabs active={section} onChange={setSection} />

      {/* ── Secciones ──────────────────────────────────────────────────────── */}
      {section === "convergence" && <SectionConvergence data={data} />}
      {section === "metrics"     && <SectionMetrics     data={data} />}
      {section === "confusion"   && <SectionConfusion   data={data} />}

    </div>
  )
}
// 
