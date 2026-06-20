import type { ParsedMetrics } from "../utils/parseMetrics";
import { ModelMetricsCharts } from "./ModelMetricsCharts";

interface ModelMetricsPanelProps {
  metrics: ParsedMetrics;
  images: {
    results: string;
    pr: string;
    confusion: string;
  };
}

export function ModelMetricsPanel({ metrics, images }: ModelMetricsPanelProps) {
  // Si las 3 imágenes son la misma, asumimos que es el dashboard estático consolidado (Mamba)
  const isConsolidated = images.results === images.confusion && images.results === images.pr;

  return (
    <div className="tm-panel-container">
      {/* KPIs Numericos (Convergencia) */}
      <div className="tm-kpi-grid">
        <div className="tm-kpi-card">
          <span className="tm-kpi-label">mAP@50</span>
          <span className="tm-kpi-value">{metrics.map50}</span>
        </div>
        <div className="tm-kpi-card">
          <span className="tm-kpi-label">mAP@50-95</span>
          <span className="tm-kpi-value">{metrics.map50_95}</span>
        </div>
        <div className="tm-kpi-card">
          <span className="tm-kpi-label">Precision</span>
          <span className="tm-kpi-value">{metrics.precision}</span>
        </div>
        <div className="tm-kpi-card">
          <span className="tm-kpi-label">Recall</span>
          <span className="tm-kpi-value">{metrics.recall}</span>
        </div>
        <div className="tm-kpi-card tm-kpi-card--highlight">
          <span className="tm-kpi-label">F1 Score</span>
          <span className="tm-kpi-value">{metrics.f1}</span>
        </div>
      </div>

      {/* Graficas */}
      {isConsolidated ? (
        <div className="tm-graphs-grid" style={{ display: 'block', marginTop: '2rem' }}>
          <div className="tm-graph-card" style={{ padding: '1rem' }}>
            <img 
              src={images.results} 
              alt="Resultados del Modelo" 
              style={{ width: '100%', height: 'auto', borderRadius: '8px', objectFit: 'contain' }} 
            />
          </div>
        </div>
      ) : (
        <div className="tm-graphs-grid">
          {/* Gráficos dinámicos interactivos (Chart.js) */}
          <ModelMetricsCharts history={metrics.history} />
          
          {/* Matriz de confusión estática */}
          <div className="tm-graph-card">
            <div className="tm-graph-overlay">
              <span>Matriz de Confusión</span>
            </div>
            <img src={images.confusion} alt="Matriz de Confusión" className="tm-graph-img" />
          </div>
        </div>
      )}
    </div>
  );
}
