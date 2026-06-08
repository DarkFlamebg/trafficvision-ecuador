import "./Resources.css"

const MODELS = [
  {
    id: "yolov11n",
    name: "YOLOv11n",
    tag: "Más rápido",
    arch: "Single-shot CNN · 2.6M params · 6.3 GFLOPs",
    color: "#22d3ee",
    size: "15 MB",
    format: ".pt",
    map50: "96.2%",
    f1: "0.951",
    speed: "~3.7 ms",
    desc: "Modelo ultraliviano ideal para tiempo real y edge devices. Excelente para despliegue en cámaras de seguridad con recursos limitados.",
    files: [
      { label: "Pesos entrenados (.pt)", filename: "yolo11n_trafficvision.pt", size: "15 MB" },
    ],
  },
  {
    id: "rtdetr",
    name: "RT-DETR-L",
    tag: "Mejor mAP",
    arch: "Transformer + CNN · 32M params · 103 GFLOPs",
    color: "#f59e0b",
    size: "251 MB",
    format: ".pt",
    map50: "96.8%",
    f1: "0.959",
    speed: "~47 ms",
    desc: "Máxima generalización gracias a mecanismos de atención Vision Transformer. Recomendado cuando la precisión es prioritaria sobre la velocidad.",
    files: [
      { label: "Pesos entrenados (.pt)", filename: "rtdetr_l_trafficvision.pt", size: "251 MB" },
    ],
  },
  {
    id: "efficientdet",
    name: "EfficientDet-D2",
    tag: "Balance óptimo",
    arch: "BiFPN + EfficientNet-B2 · 3.5M params · 112 GFLOPs",
    color: "#10b981",
    size: "94.4 MB",
    format: ".pt",
    map50: "96.8%",
    f1: "0.959",
    speed: "~47 ms / ~5 ms GPU",
    desc: "El mejor equilibrio entre tamaño de modelo, costo computacional y precisión. Especialmente destacado en el dataset ecuatoriano.",
    files: [
      { label: "Pesos entrenados (.pt)", filename: "efficientdet_d2_trafficvision.pt", size: "94.4 MB" },
    ],
  },
]

const EXTRAS = [
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
      </svg>
    ),
    label: "Documento de Tesis",
    desc: "Investigación completa: metodología, resultados y conclusiones.",
    filename: "TESIS_INVESTIGATIVA_BAZAN_YAGUAL.pdf",
    size: "~4 MB",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6"/>
        <polyline points="8 6 2 12 8 18"/>
      </svg>
    ),
    label: "Scripts de Entrenamiento",
    desc: "Código fuente de entrenamiento para los tres modelos.",
    filename: "training_scripts.zip",
    size: "~120 KB",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
    label: "Dataset (muestra)",
    desc: "Subconjunto representativo del dataset de validación ecuatoriano.",
    filename: "trafficvision_sample_dataset.zip",
    size: "~200 MB",
  },
]

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  )
}

export default function Resources() {
  return (
    <div className="res-container">
      <div className="res-header">
        <span className="res-eyebrow">Entregables del Proyecto</span>
        <h1>Recursos y Descargas</h1>
        <p>
          Pesos de modelos entrenados, scripts de entrenamiento y documentación técnica
          producidos durante la fase experimental de TrafficVision.
        </p>
      </div>

      {/* Modelos */}
      <section className="res-section">
        <h2 className="res-section-title">
          <span className="res-section-num">01</span>
          Modelos Entrenados
        </h2>

        <div className="res-models-grid">
          {MODELS.map((m) => (
            <article key={m.id} className="res-model-card" style={{ "--model-color": m.color } as React.CSSProperties}>
              <div className="res-model-head">
                <div>
                  <span className="res-model-tag" style={{ color: m.color, borderColor: m.color + "44", background: m.color + "12" }}>
                    {m.tag}
                  </span>
                  <h3 className="res-model-name">{m.name}</h3>
                  <span className="res-model-arch">{m.arch}</span>
                </div>
              </div>

              <p className="res-model-desc">{m.desc}</p>

              <div className="res-model-pills">
                <div className="res-pill">
                  <span className="res-pill-lbl">mAP@50</span>
                  <span className="res-pill-val" style={{ color: m.color }}>{m.map50}</span>
                </div>
                <div className="res-pill">
                  <span className="res-pill-lbl">F1-Score</span>
                  <span className="res-pill-val" style={{ color: m.color }}>{m.f1}</span>
                </div>
                <div className="res-pill">
                  <span className="res-pill-lbl">Inferencia</span>
                  <span className="res-pill-val">{m.speed}</span>
                </div>
                <div className="res-pill">
                  <span className="res-pill-lbl">Tamaño</span>
                  <span className="res-pill-val">{m.size}</span>
                </div>
              </div>

              <div className="res-model-files">
                {m.files.map((f) => (
                  <div key={f.filename} className="res-file-row">
                    <div className="res-file-info">
                      <span className="res-file-name">{f.label}</span>
                      <span className="res-file-meta">{f.filename} · {f.size}</span>
                    </div>
                    <button className="res-download-btn" style={{ borderColor: m.color + "44", color: m.color }} disabled title="Próximamente disponible">
                      <DownloadIcon />
                      Descargar
                    </button>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Extras */}
      <section className="res-section">
        <h2 className="res-section-title">
          <span className="res-section-num">02</span>
          Documentación y Datos
        </h2>

        <div className="res-extras-grid">
          {EXTRAS.map((e) => (
            <div key={e.label} className="res-extra-card">
              <div className="res-extra-icon">{e.icon}</div>
              <div className="res-extra-body">
                <span className="res-extra-label">{e.label}</span>
                <span className="res-extra-desc">{e.desc}</span>
                <span className="res-extra-meta">{e.filename} · {e.size}</span>
              </div>
              <button className="res-download-btn res-download-btn--neutral" disabled title="Próximamente disponible">
                <DownloadIcon />
                Descargar
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Info footer */}
      <div className="res-notice">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <p>
          Los archivos estarán disponibles para descarga directa una vez el proyecto sea publicado. 
          Contacta al equipo de investigación para acceso anticipado.
        </p>
      </div>
    </div>
  )
}
