import { useMemo } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { Line } from "react-chartjs-2";
import type { EpochData } from "../utils/parseMetrics";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Variables comunes de diseño oscuro
const darkOptions = {
  responsive: true,
  maintainAspectRatio: false,
  color: "#9ca3af", // texto gris claro
  plugins: {
    legend: {
      position: "bottom" as const,
      labels: {
        color: "#d1d5db",
        usePointStyle: true,
        boxWidth: 8,
      }
    },
    tooltip: {
      backgroundColor: "rgba(17, 17, 19, 0.9)",
      titleColor: "#ffffff",
      bodyColor: "#d1d5db",
      borderColor: "#222226",
      borderWidth: 1,
      padding: 10,
    }
  },
  scales: {
    x: {
      grid: {
        color: "rgba(255, 255, 255, 0.05)",
      },
      ticks: { color: "#9ca3af" }
    },
    y: {
      min: 0,
      max: 1.0,
      grid: {
        color: "rgba(255, 255, 255, 0.05)",
      },
      ticks: { color: "#9ca3af" }
    }
  }
};

interface ChartsProps {
  history: EpochData[];
}

export function ModelMetricsCharts({ history }: ChartsProps) {
  // Chart 1: Evolución de Precisión, Recall y mAP (eje X = época)
  const evolutionData = useMemo(() => {
    return {
      labels: history.map(d => d.epoch),
      datasets: [
        {
          label: "Precisión",
          data: history.map(d => d.precision),
          borderColor: "#3b82f6", // azul
          backgroundColor: "#3b82f6",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "Recall",
          data: history.map(d => d.recall),
          borderColor: "#ef4444", // rojo/naranja
          backgroundColor: "#ef4444",
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "mAP@50",
          data: history.map(d => d.map50),
          borderColor: "#10b981", // verde
          backgroundColor: "#10b981",
          borderWidth: 2,
          borderDash: [2, 2],
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3
        }
      ]
    };
  }, [history]);

  // Chart 2: Curva Precision-Recall (eje X = Recall, eje Y = Precisión)
  const prCurveData = useMemo(() => {
    // Para esta curva, conectamos los puntos P-R generados a través de las épocas
    return {
      // El eje X de Chart.js por defecto es categórico para líneas, 
      // pero podemos mapear x/y explícitamente si cambiamos el tipo de escala, 
      // o usar el Recall como label. Vamos a usar etiquetas de Recall limitadas a 2 decimales
      labels: history.map(d => d.recall.toFixed(2)),
      datasets: [
        {
          label: "Precisión vs Recall",
          data: history.map(d => d.precision),
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          pointRadius: 1,
          pointHoverRadius: 5,
          tension: 0.1
        }
      ]
    };
  }, [history]);

  return (
    <>
      <div className="tm-graph-card tm-graph-card--wide">
        <div className="tm-graph-overlay-top">
          <span>Evolución de Precisión y Recall</span>
        </div>
        <div style={{ padding: "3rem 1.5rem 1.5rem", width: "100%", height: "100%" }}>
          <Line options={darkOptions} data={evolutionData} />
        </div>
      </div>

      <div className="tm-graph-card">
        <div className="tm-graph-overlay-top">
          <span>Curva Precision-Recall (Histórica)</span>
        </div>
        <div style={{ padding: "3rem 1.5rem 1.5rem", width: "100%", height: "100%" }}>
          <Line options={{
             ...darkOptions,
             scales: {
               x: {
                 title: { display: true, text: "Recall", color: "#9ca3af" },
                 grid: { color: "rgba(255, 255, 255, 0.05)" },
                 ticks: { color: "#9ca3af", maxTicksLimit: 10 }
               },
               y: {
                 title: { display: true, text: "Precisión", color: "#9ca3af" },
                 min: 0, max: 1.0,
                 grid: { color: "rgba(255, 255, 255, 0.05)" },
                 ticks: { color: "#9ca3af" }
               }
             }
          }} data={prCurveData} />
        </div>
      </div>
    </>
  );
}
