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

const DATASETS = [
  {
    id: "plates-ecuadorian-v4",
    name: "Plates Ecuadorian - v4",
    source: "Roboflow",
    summary: "2100 train · 216 valid · 115 test",
    note: "Dataset ecuatoriano cargado en Roboflow Universe.",
    url: "https://universe.roboflow.com/stevens-workspace-unaqf/plates-ecuadorian/dataset/4",
  },
  {
    id: "global-ecuador-combined",
    name: "Global Ecuador Combined",
    source: "Google Drive",
    summary: "~10,000 imágenes combinadas de Ecuador",
    note: "Dataset global combinado referenciado en Drive.",
    url: "https://drive.google.com/file/d/1UV2moaMn-B3zoQFv-5SwnZ39H7qfvcUp/view?usp=sharing",
  },
]

const EXTRAS = [
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

      {/* Datasets */}
      <section className="res-section">
        <h2 className="res-section-title">
          <span className="res-section-num">02</span>
          Datasets Remotos
        </h2>

        <div className="res-extras-grid">
          {DATASETS.map((dataset) => (
            <div key={dataset.id} className="res-extra-card">
              <div className="res-extra-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3v18" />
                  <path d="M5 8l7-5 7 5" />
                  <path d="M5 16l7 5 7-5" />
                </svg>
              </div>
              <div className="res-extra-body">
                <span className="res-extra-label">{dataset.name}</span>
                <span className="res-extra-desc">{dataset.note}</span>
                <span className="res-extra-meta">{dataset.summary} · {dataset.source}</span>
              </div>
              <a
                className="res-download-btn res-download-btn--neutral"
                href={dataset.url}
                target="_blank"
                rel="noreferrer"
              >
                <DownloadIcon />
                Ver enlace
              </a>
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
