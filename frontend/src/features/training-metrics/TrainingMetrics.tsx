import "./TrainingMetrics.css"

import yoloResults from "../../assets/images/yolov11/results.png"
import yoloPR from "../../assets/images/yolov11/BoxPR_curve.png"
import yoloCM from "../../assets/images/yolov11/confusion_matrix_normalized.png"

import rtdetrResults from "../../assets/images/rtdetr/results.png"
import rtdetrPR from "../../assets/images/rtdetr/BoxPR_curve.png"
import rtdetrCM from "../../assets/images/rtdetr/confusion_matrix_normalized.png"

import effResults from "../../assets/images/efficient/results.png"
import effPR from "../../assets/images/efficient/BoxPR_curve.png"
import effCM from "../../assets/images/efficient/confusion_matrix_normalized.png"

const MODELS = [
  {
    id: "yolov11n",
    name: "YOLOv11n",
    desc: "Single-shot CNN - Máxima velocidad",
    metrics: { map50: "94.2%", map95: "71.5%", f1: "0.92" },
    images: { results: yoloResults, pr: yoloPR, confusion: yoloCM, resultsTitle: "Curva de Pérdida (Loss)" }
  },
  {
    id: "rtdetr",
    name: "RT-DETR",
    desc: "Transformer + CNN - Alta precisión en oclusión",
    metrics: { map50: "95.8%", map95: "74.1%", f1: "0.94" },
    images: { results: rtdetrResults, pr: rtdetrPR, confusion: rtdetrCM, resultsTitle: "Curva de Pérdida (Loss)" }
  },
  {
    id: "efficientdet",
    name: "EfficientDet-D2",
    desc: "BiFPN + EfficientNet - Balance eficiente",
    metrics: { map50: "93.5%", map95: "69.8%", f1: "0.90" },
    images: { results: effResults, pr: effPR, confusion: effCM, resultsTitle: "Curva de Pérdida (Loss)" }
  }
]

function GraphImage({ src, title }: { src: string, title: string }) {
  return (
    <div className="tm-graph-image-container">
      <img src={src} alt={title} className="tm-graph-img" />
      <div className="tm-graph-overlay">
        <span>{title}</span>
      </div>
    </div>
  )
}

export default function TrainingMetrics() {
  return (
    <div className="tm-container">
      <div className="tm-header">
        <h1>Análisis de Entrenamiento y Validación</h1>
        <p>Curvas de convergencia empírica, curvas Precision-Recall y matrices de confusión obtenidas durante la fase experimental (Sprints 1 y 2) de TrafficVision.</p>
      </div>

      <div className="tm-models-wrapper">
        {MODELS.map((model) => (
          <section key={model.id} className="tm-model-section">
            <div className="tm-model-header">
              <div className="tm-model-info">
                <h2>{model.name}</h2>
                <span className="tm-model-desc">{model.desc}</span>
              </div>
              <div className="tm-model-stats">
                <div className="tm-stat-pill">
                  <span className="tm-stat-label">mAP@50</span>
                  <span className="tm-stat-value">{model.metrics.map50}</span>
                </div>
                <div className="tm-stat-pill">
                  <span className="tm-stat-label">F1-Score</span>
                  <span className="tm-stat-value">{model.metrics.f1}</span>
                </div>
              </div>
            </div>

            <div className="tm-graphs-grid">
              <div className="tm-graph-card">
                <GraphImage src={model.images.results} title={model.images.resultsTitle} />
              </div>
              <div className="tm-graph-card">
                <GraphImage src={model.images.pr} title="Curva Precision-Recall" />
              </div>
              <div className="tm-graph-card tm-graph-card--wide">
                <GraphImage src={model.images.confusion} title="Matriz de Confusión Normalizada" />
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
