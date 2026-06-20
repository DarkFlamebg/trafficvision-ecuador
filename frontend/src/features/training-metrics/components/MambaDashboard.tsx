import { useMemo } from "react"
import { Line } from "react-chartjs-2"
import type { EpochRow } from "./Trainingmetricsdynamic"

const COLORS = {
  blue:   "#185FA5",
  orange: "#F28C28",
  green:  "#2E8B57",
  red:    "#D32F2F",
  navy:   "#1A237E",
  cyan:   "#0288D1",
}

function lineOpts(
  yLabel: string,
  yMin?: number,
  yMax?: number,
  logarithmic = false,
) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    interaction: { mode: "index" as const, intersect: false },
    plugins: {
      legend:  { display: true, position: "top" as const, labels: { color: "#bbb", boxWidth: 12, font: { size: 10 } } },
      tooltip: { padding: 8 },
    },
    scales: {
      x: {
        grid:  { color: "rgba(255,255,255,0.05)" },
        ticks: { color: "#888", font: { size: 9 }, maxTicksLimit: 10 },
        title: { display: true, text: "Época", color: "#888", font: { size: 9 } },
      },
      y: {
        type:  logarithmic ? ("logarithmic" as const) : ("linear" as const),
        grid:  { color: "rgba(255,255,255,0.05)" },
        ticks: { color: "#888", font: { size: 9 } },
        title: { display: true, text: yLabel, color: "#888", font: { size: 9 } },
        ...(yMin !== undefined ? { min: yMin } : {}),
        ...(yMax !== undefined ? { max: yMax } : {}),
      },
    },
    elements: {
      point: { radius: 0, hoverRadius: 4 },
      line:  { tension: 0.35, borderWidth: 1.8 },
    },
  }
}

function ds(label: string, color: string, values: (number | undefined)[]) {
  return {
    label,
    data: values.map(v => (v === undefined || isNaN(v as number) ? null : v)),
    borderColor: color,
    backgroundColor: color,
    spanGaps: true,
  }
}

function ChartCard({
  id, title, explain, children,
}: {
  id: string
  title: string
  explain: string
  children: React.ReactNode
}) {
  return (
    <div className="tmd-section" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      <h3 style={{ margin: 0, fontSize: "0.85rem", fontWeight: 600, textAlign: "center", color: "#ddd" }}>
        {title}
      </h3>
      <div style={{ height: "240px" }}>{children}</div>
      <p style={{ margin: 0, fontSize: "0.72rem", color: "#888", lineHeight: 1.5, textAlign: "justify" }}>
        {explain}
      </p>
    </div>
  )
}

export function MambaDashboard({ data }: { data: EpochRow[] }) {
  const epochs = useMemo(() => data.map(r => r.epoch), [data])

  const c1 = useMemo(() => ({ labels: epochs, datasets: [ds("Pérdida Total", COLORS.blue,  data.map(r => r.trainLossTotal))] }), [data, epochs])
  const c2 = useMemo(() => ({ labels: epochs, datasets: [
    ds("bbox",     COLORS.red,    data.map(r => r.trainBox)),
    ds("cls",      COLORS.blue,   data.map(r => r.trainCls)),
    ds("rpn_bbox", COLORS.green,  data.map(r => r.trainRpnBbox)),
    ds("rpn_cls",  COLORS.orange, data.map(r => r.trainRpnCls)),
  ]}), [data, epochs])
  const c3 = useMemo(() => ({ labels: epochs, datasets: [ds("LR", COLORS.green, data.map(r => r.trainLr))] }), [data, epochs])
  const c4 = useMemo(() => ({ labels: epochs, datasets: [
    ds("mAP",    COLORS.navy, data.map(r => r.map5095)),
    ds("mAP@50", COLORS.red,  data.map(r => r.map50)),
    ds("mAP@75", COLORS.cyan, data.map(r => r.map75)),
  ]}), [data, epochs])
  const c5 = useMemo(() => ({ labels: epochs, datasets: [
    ds("Large",  COLORS.green,  data.map(r => r.mapL)),
    ds("Medium", COLORS.orange, data.map(r => r.mapM)),
    ds("Small",  COLORS.red,    data.map(r => r.mapS)),
  ]}), [data, epochs])
  const c6 = useMemo(() => ({ labels: epochs, datasets: [
    ds("mAP@50", COLORS.red,  data.map(r => r.map50)),
    ds("mAP@75", COLORS.cyan, data.map(r => r.map75)),
  ]}), [data, epochs])

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(3, 1fr)",
      gap: "1.5rem",
      marginTop: "2rem",
    }}>
      <ChartCard id="mb-c1" title="Pérdida Total (Train)" explain="Reducción global del error durante el entrenamiento. Una curva decreciente indica convergencia estable del modelo.">
        <Line id="mb-c1" data={c1} options={lineOpts("Loss")} />
      </ChartCard>

      <ChartCard id="mb-c2" title="Pérdidas Detalladas" explain="Desglose de los componentes de pérdida: bbox y cls son las pérdidas principales del detector, rpn_bbox y rpn_cls corresponden a la región propuesta.">
        <Line id="mb-c2" data={c2} options={lineOpts("Loss")} />
      </ChartCard>

      <ChartCard id="mb-c3" title="Learning Rate" explain="Evolución escalonada de la tasa de aprendizaje. La escala logarítmica permite apreciar el decaimiento progresivo que estabiliza el entrenamiento.">
        <Line id="mb-c3" data={c3} options={lineOpts("LR", undefined, undefined, true)} />
      </ChartCard>

      <ChartCard id="mb-c4" title="mAP (Validación)" explain="Métricas de precisión media en validación. mAP@50 mide solapamientos del 50%; mAP@75 es más estricto y exige mayor exactitud en la caja delimitadora.">
        <Line id="mb-c4" data={c4} options={lineOpts("mAP", 0, 1)} />
      </ChartCard>

      <ChartCard id="mb-c5" title="mAP por Tamaño de Placa" explain="Rendimiento segmentado por escala de objeto. Las placas grandes (Large) obtienen el mejor mAP gracias a su área de cobertura en el detector.">
        <Line id="mb-c5" data={c5} options={lineOpts("mAP", 0, 1)} />
      </ChartCard>

      <ChartCard id="mb-c6" title="mAP@50 vs mAP@75" explain="Comparativa directa entre umbral normal (50%) y estricto (75%). La brecha entre ambas curvas refleja cuán preciso es el modelo al localizar la placa.">
        <Line id="mb-c6" data={c6} options={lineOpts("mAP", 0, 1)} />
      </ChartCard>
    </div>
  )
}
