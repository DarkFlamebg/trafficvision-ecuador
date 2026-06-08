import { useNavigate } from "react-router-dom"
import "./Home.css"

// --- Nuevas constantes explicativas -------------------------------------------------
// Descripción de cada métrica que se muestra en la tabla de modelos.
const METRIC_EXPLANATIONS = {
  "mAP@50": "Mean Average Precision a un umbral de IoU=0.5. Indica la proporción de detecciones correctas.",
  "mAP@50-95": "Promedio de mAP desde IoU 0.5 hasta 0.95 (paso 0.05). Mide la precisión global del detector.",
  "Precisión": "Porcentaje de detecciones correctas respecto al total detectado.",
  "Recall": "Porcentaje de objetos reales que fueron detectados.",
  "speed": "Tiempo medio de inferencia por imagen (CPU/GPU).",
  "fps": "Frames por segundo estimados a partir del tiempo de inferencia.",
};


const MODELS = [
  {
    id: "efficientdet-d2-combined",
    tag: "EfficientDet-D2",
    tagClass: "tag--efficientdet",
    name: "EfficientDet-D2 Combined",
    arch: "BiFPN + EfficientNet-B2 · 313 layers · 3.5M params · 112 GFLOPs",
    badge: "Alta Precisión",
    badgeClass: "badge--best",
    featured: true,
    metrics: [
      { key: "mAP@50",    val: "96.8%", bar: 96.8 },
      { key: "mAP@50-95", val: "78.4%", bar: 78.4 },
      { key: "Precisión", val: "97.1%", bar: 97.1 },
      { key: "Recall",    val: "94.7%", bar: 94.7 },
    ],
    speed: "~47 ms (CPU) / ~5 ms (GPU)",
    fps: "~21 FPS (CPU) / ~208 FPS (GPU)",
    size: "94.4 MB",
    note: "Detector de precisión equilibrada. Excelente generalización global + especialización en placas ecuatorianas.",
  },
  {
    id: "rtdetr-combined",
    tag: "RT-DETR-L",
    tagClass: "tag--rtdetr",
    name: "RT-DETR Combined",
    arch: "Transformer + CNN · 313 layers · 32M params · 103 GFLOPs",
    badge: "Mejor mAP",
    badgeClass: "badge--best",
    featured: true,
    metrics: [
      { key: "mAP@50",    val: "96.8%", bar: 96.8 },
      { key: "mAP@50-95", val: "68.6%", bar: 68.6 },
      { key: "Precisión", val: "97.1%", bar: 97.1 },
      { key: "Recall",    val: "94.7%", bar: 94.7 },
    ],
    speed: "~47 ms",
    fps: "~30 FPS",
    size: "251 MB",
    note: "Generalización superior. Entrenado con dataset global de 2 048 imágenes de validación.",
  },
  {
    id: "yolo11n",
    tag: "YOLOv11n",
    tagClass: "tag--yolo",
    name: "YOLO11n",
    arch: "Single-shot · 101 layers fusionadas · 2.6M params · 6.3 GFLOPs",
    badge: "Más rápido",
    badgeClass: "badge--fast",
    featured: false,
    metrics: [
      { key: "mAP@50",    val: "96.2%", bar: 96.2 },
      { key: "mAP@50-95", val: "65.7%", bar: 65.7 },
      { key: "Precisión", val: "98.2%", bar: 98.2 },
      { key: "Recall",    val: "92.3%", bar: 92.3 },
    ],
    speed: "~3.7 ms",
    fps: "~200 FPS",
    size: "15 MB",
    note: "Inferencia ultrarrápida. Ideal para tiempo real, edge devices y alta concurrencia.",
  },
]

const STACK = [
  {
    layer: "Frontend",
    tech: "React / Vite",
    desc: "UI · Componentes",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" height="28" width="28">
        <circle cx="12" cy="12" r="2.5" fill="#61DAFB"/>
        <ellipse cx="12" cy="12" rx="10" ry="4" stroke="#61DAFB" strokeWidth="1.3" fill="none"/>
        <ellipse cx="12" cy="12" rx="10" ry="4" stroke="#61DAFB" strokeWidth="1.3" fill="none" transform="rotate(60 12 12)"/>
        <ellipse cx="12" cy="12" rx="10" ry="4" stroke="#61DAFB" strokeWidth="1.3" fill="none" transform="rotate(120 12 12)"/>
      </svg>
    ),
  },
  {
    layer: "Backend",
    tech: "FastAPI",
    desc: "REST · Python 3.11",
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" height="28" width="28">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14l-3-3 1.41-1.41L11 13.17l5.59-5.59L18 9l-7 7z" fill="#05998b"/>
      </svg>
    ),
  },
  {
    layer: "Detección",
    tech: "Ultralytics",
    desc: "YOLOv11n · RT-DETR-L",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" height="28" width="28">
        <rect x="3" y="3" width="7" height="7" rx="1" stroke="#FF6B35" strokeWidth="1.5"/>
        <rect x="14" y="3" width="7" height="7" rx="1" stroke="#FF6B35" strokeWidth="1.5"/>
        <rect x="3" y="14" width="7" height="7" rx="1" stroke="#FF6B35" strokeWidth="1.5"/>
        <path d="M14 17.5h7M17.5 14v7" stroke="#FF6B35" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    layer: "Deep Learning",
    tech: "PyTorch 2.10",
    desc: "CUDA 12.8 · Tesla T4",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" height="28" width="28">
        <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.8-3.5 6L14 21H10l-1.5-6C6.5 13.8 5 11.5 5 9a7 7 0 0 1 7-7z" stroke="#EE4C2C" strokeWidth="1.5" fill="none"/>
        <circle cx="12" cy="9" r="2" fill="#EE4C2C"/>
      </svg>
    ),
  },
  {
    layer: "OCR",
    tech: "EasyOCR",
    desc: "Extracción de texto",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" height="28" width="28">
        <rect x="2" y="6" width="20" height="12" rx="2" stroke="#a78bfa" strokeWidth="1.5"/>
        <path d="M6 10h4M6 14h8M14 10h4" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    layer: "Visión",
    tech: "OpenCV",
    desc: "Pre y post procesado",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" height="28" width="28">
        <circle cx="12" cy="12" r="4" stroke="#5cb85c" strokeWidth="1.5"/>
        <circle cx="12" cy="12" r="1.5" fill="#5cb85c"/>
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3" stroke="#5cb85c" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" stroke="#5cb85c" strokeWidth="1.2" strokeLinecap="round"/>
      </svg>
    ),
  },
]

const WHY = [
  { icon: "01", title: "Control de tránsito",       desc: "Automatiza la identificación de vehículos en peajes, parqueaderos y zonas restringidas sin intervención humana." },
  { icon: "02", title: "Fiscalización y Anticorrupción", desc: "Detecta infracciones y registros pendientes en tiempo real, ofreciendo una herramienta precisa que minimiza la subjetividad y mejora la transparencia." },
  { icon: "03", title: "Optimización local",        desc: "Entrenado con capturas reales de Guayaquil (Ecuador), adaptado específicamente a las condiciones de iluminación, desgaste y formatos locales." },
  { icon: "04", title: "Alta velocidad",            desc: "Desde 3.7 ms por imagen. Apto para cámaras de vigilancia estándar sin requerir hardware sumamente costoso." },
]

const QUICK_STATS = [
  { val: "96.8%",  lbl: "mAP@50 mejor modelo" },
  { val: "3.7 ms", lbl: "Inferencia YOLO11n" },
  { val: "2 048",  lbl: "Imágenes validación" },
  { val: "32 M",   lbl: "Params RT-DETR-L" },
]

const ACTIONS = [
  { label: "Analizar placa",     route: "/read-plate",         primary: true  },
  { label: "Comparar Modelos",   route: "/model-comparison",   primary: false },
  { label: "Benchmark Live",     route: "/benchmark",          primary: false },
]

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="home-root">
      <div className="global-aurora-bg" aria-hidden="true" />

      <header className="home-hero">
        {/* Contenedor en dos columnas */}
        <div className="home-hero-content">
          
          {/* COLUMNA IZQUIERDA: Textos y CTAs */}
          <div className="home-hero-inner">
            <div className="home-eyebrow">
              <span className="home-dot" aria-hidden="true" />
              Sistema activo · Detección en tiempo real
            </div>

            <h1 className="home-title">
              Reconocimiento<br />
              automático de<br />
              <span>placas vehiculares</span>
            </h1>

            <p className="home-sub">
              Sistema de visión por computadora que detecta y lee placas en imágenes
              y video usando redes neuronales de última generación.
            </p>

            <div className="home-cta-row">
              {ACTIONS.map((a) => (
                <button
                  key={a.route}
                  className={a.primary ? "home-btn-primary" : "home-btn-secondary"}
                  onClick={() => navigate(a.route)}
                >
                  {a.label}
                  {a.primary && (
                    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                      <path d="M2.5 7h9M8 3.5L11.5 7 8 10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* COLUMNA DERECHA: Imagen de análisis o Mockup Técnico */}
          <div className="home-hero-visual">
            <div className="visual-preview-container">
              {/* Reemplaza la URL por la imagen o mockup real de tu sistema */}
              <img 
                src="../src\assets\images\Landing-page.png" 
                alt="Simulación de detección de placas" 
                className="visual-main-img"
              />
              {/* Superposición técnica simulando la inferencia de la IA */}
              <div className="visual-scan-line" />
            </div>
          </div>

        </div>

        {/* Barra de estadísticas inferior permanece intacta */}
        <div className="home-stats-strip">
          {QUICK_STATS.map((s) => (
            <div key={s.lbl} className="home-stat">
              <span className="home-stat-val">{s.val}</span>
              <span className="home-stat-lbl">{s.lbl}</span>
            </div>
          ))}
        </div>
      </header>

      <section className="home-section">
        <div className="home-section-inner">
          <header className="home-section-header">
            <span className="home-label">Antecedente</span>
            <h2 className="home-section-title">¿Por qué existe<br />este sistema?</h2>
            <p className="home-section-sub" style={{ maxWidth: "800px", marginTop: "1rem" }}>
              Este proyecto es el resultado del Diseño de un sistema de reconocimiento de placas vehiculares basado en múltiples modelos de inteligencia artificial preentrenados, con el objetivo de optimizar la seguridad vial y el control anticorrupción en la ciudad de Guayaquil, Ecuador.
              Al utilizar datos locales etiquetados, ofrece a las autoridades de tránsito una herramienta automatizada y precisa adaptada a la realidad del país.
            </p>
          </header>
          <div className="home-why-grid">
            {WHY.map((w) => (
              <div key={w.title} className="home-why-card">
                <span className="home-why-num">{w.icon}</span>
                <h3 className="home-why-title">{w.title}</h3>
                <p className="home-why-desc">{w.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="home-divider" />

      {/* METRICAS EXPLICADAS Y CASOS DE USO */}
      <section className="home-section">
        <div className="home-section-inner">
          <header className="home-section-header">
            <span className="home-label">Evaluación Técnica</span>
            <h2 className="home-section-title">Entendiendo los Modelos</h2>
            <p className="home-section-sub">
              Diferentes arquitecturas de Deep Learning adaptadas para distintos propósitos.
            </p>
          </header>

          <div className="home-info-grid">
            {/* Casos de Uso y Arquitecturas */}
            <div className="home-info-column">
              <h3 className="home-info-title">Arquitecturas y Casos de Uso</h3>
              <div className="home-arch-cards">
                <div className="home-arch-card">
                  <h4>YOLOv11n</h4>
                  <span className="home-arch-type">Single-shot CNN</span>
                  <p>Inferencia ultrarrápida con latencia estricta. Ideal para dispositivos Edge, cámaras de seguridad con recursos limitados y procesamiento de video en tiempo real masivo.</p>
                </div>
                <div className="home-arch-card">
                  <h4>RT-DETR</h4>
                  <span className="home-arch-type">Transformer + CNN</span>
                  <p>Aprovecha mecanismos de atención (Vision Transformers) para lograr máxima generalización y precisión global. Útil en condiciones difíciles de iluminación o placas desgastadas.</p>
                </div>
                <div className="home-arch-card">
                  <h4>EfficientDet-D2</h4>
                  <span className="home-arch-type">BiFPN + EfficientNet</span>
                  <p>Arquitectura alternativa que ofrece un excelente equilibrio entre eficiencia computacional y alta precisión, destacando en el conjunto de datos de Ecuador.</p>
                </div>
              </div>
            </div>

            {/* Explicación de Métricas */}
            <div className="home-info-column">
              <h3 className="home-info-title">Métricas de Rendimiento</h3>
              <div className="home-metrics-list">
                {Object.entries(METRIC_EXPLANATIONS).map(([key, desc]) => (
                  <div key={key} className="home-metric-desc">
                    <strong>{key}</strong>
                    <p>{desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="home-divider" />

      {/* STACK TECNOLOGICO */}
      <section className="home-section">
        <div className="home-section-inner">
          <header className="home-section-header">
            <span className="home-label">Herramientas</span>
            <h2 className="home-section-title">Stack Tecnológico</h2>
            <p className="home-section-sub">Las tecnologías detrás del motor de inferencia y la plataforma.</p>
          </header>
          
          <div className="home-stack-grid">
            {STACK.map((item) => (
              <div key={item.layer} className="home-stack-item">
                <div className="home-stack-icon">{item.icon}</div>
                <span className="home-stack-layer">{item.layer}</span>
                <span className="home-stack-tech">{item.tech}</span>
                <span className="home-stack-desc">{item.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="home-divider" />

      <section className="home-section">
        <div className="home-section-inner">
          <header className="home-section-header">
            <span className="home-label">Resultados reales</span>
            <h2 className="home-section-title">Benchmark comparativo</h2>
            <p className="home-section-sub">
              Evaluación sobre 10 048 imágenes · Hardware: Tesla T4 · CUDA 12.8 · Ultralytics 8.4.46
            </p>
          </header>

          <div className="home-models-grid">
            {MODELS.map((m) => (
              <article
                key={m.id}
                className={`home-model-card${m.featured ? " home-model-card--featured" : ""}`}
              >
                <div className="home-model-head">
                  <span className={`home-model-tag ${m.tagClass}`}>{m.tag}</span>
                  <span className="home-model-badge">{m.badge}</span>
                </div>

                <h3 className="home-model-name">{m.name}</h3>
                <p className="home-model-arch">{m.arch}</p>

                <div className="home-metrics">
                  {m.metrics.map((met) => (
                    <div key={met.key} className="home-metric">
                      <div className="home-metric-row">
                        <span className="home-metric-key">{met.key}</span>
                        <span className="home-metric-val">{met.val}</span>
                      </div>
                      <div className="home-bar-bg">
                        <div
                          className={`home-bar-fill ${m.tagClass}`}
                          style={{ width: `${met.bar}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="home-model-footer">
                  <div className="home-speed-row">
                    <span>⚡ {m.speed}</span>
                    <span>{m.fps}</span>
                    <span>{m.size}</span>
                  </div>
                  <p className="home-model-note">{m.note}</p>
                </div>
              </article>
            ))}
          </div>

          <div className="home-callout">
            <span className="home-callout-icon" aria-hidden="true">→</span>
            <p>
              <strong>RT-DETR Combined supera al modelo Ecuador en +73.5 pp de mAP@50.</strong>{" "}
              El dataset global aporta generalización crítica. RT-DETR y YOLO11n rinden de forma
              similar (+0.6 pp) — prefiere YOLO11n si la latencia es prioritaria.
            </p>
          </div>
        </div>
      </section>

      <div className="home-divider" />

      <footer className="home-footer">
        <span>Backend · <code>localhost:8000</code></span>
        <span>TrafficVision · 2026</span>
      </footer>

    </div>
  )
}