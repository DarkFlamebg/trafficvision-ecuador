import { useState, useMemo } from "react"
import "./TrainingMetrics.css"
import { ModelMetricsPanel } from "./components/ModelMetricsPanel"
import { parseMetrics } from "./utils/parseMetrics"

// Importar imágenes (YOLOv11)
import yoloResults from "../../assets/images/yolov11/results.png"
import yoloPR from "../../assets/images/yolov11/BoxPR_curve.png"
import yoloCM from "../../assets/images/yolov11/confusion_matrix_normalized.png"
import yoloCsv from "../../assets/images/yolov11/results.csv?raw"

// Importar imágenes (RT-DETR)
import rtdetrResults from "../../assets/images/rtdetr/results.png"
import rtdetrPR from "../../assets/images/rtdetr/BoxPR_curve.png"
import rtdetrCM from "../../assets/images/rtdetr/confusion_matrix_normalized.png"
import rtdetrCsv from "../../assets/images/rtdetr/results.csv?raw"

// Importar imágenes (EfficientDet)
import effResults from "../../assets/images/efficient/results.png"
import effPR from "../../assets/images/efficient/BoxPR_curve.png"
import effCM from "../../assets/images/efficient/confusion_matrix_normalized.png"
import effCsv from "../../assets/images/efficient/results.csv?raw"

const MODELS_DATA = {
  yolov11n: {
    id: "yolov11n",
    name: "YOLOv11n",
    desc: "Single-shot CNN - Máxima velocidad",
    csvRaw: yoloCsv,
    images: { results: yoloResults, pr: yoloPR, confusion: yoloCM }
  },
  rtdetr: {
    id: "rtdetr",
    name: "RT-DETR",
    desc: "Transformer + CNN - Alta precisión en oclusión",
    csvRaw: rtdetrCsv,
    images: { results: rtdetrResults, pr: rtdetrPR, confusion: rtdetrCM }
  },
  efficientdet: {
    id: "efficientdet",
    name: "EfficientDet-D2",
    desc: "BiFPN + EfficientNet - Balance eficiente",
    csvRaw: effCsv,
    images: { results: effResults, pr: effPR, confusion: effCM }
  }
}

type ModelKey = keyof typeof MODELS_DATA

export default function TrainingMetrics() {
  const [activeTab, setActiveTab] = useState<ModelKey>("yolov11n")

  const currentModel = MODELS_DATA[activeTab]
  
  // Procesar CSV dinámicamente según la pestaña activa
  const metrics = useMemo(() => {
    return parseMetrics(currentModel.csvRaw)
  }, [currentModel.csvRaw])

  return (
    <div className="tm-container">
      <div className="tm-header">
        <h1>Análisis de Entrenamiento y Validación</h1>
        <p>Curvas de convergencia, métricas Precision/Recall y matrices de confusión calculadas dinámicamente desde el <code>results.csv</code> de cada modelo.</p>
      </div>

      <div className="tm-tabs">
        {(Object.keys(MODELS_DATA) as ModelKey[]).map((key) => (
          <button
            key={key}
            className={`tm-tab-btn ${activeTab === key ? "tm-tab-btn--active" : ""}`}
            onClick={() => setActiveTab(key)}
          >
            {MODELS_DATA[key].name}
          </button>
        ))}
      </div>

      <div className="tm-active-model-header">
        <h2>{currentModel.name}</h2>
        <span className="tm-model-desc">{currentModel.desc}</span>
      </div>

      <ModelMetricsPanel metrics={metrics} images={currentModel.images} />
    </div>
  )
}
