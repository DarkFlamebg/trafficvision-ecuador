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

// Descripción ampliada de cada capa tecnológica del stack.
const TOOL_EXPLANATIONS = {
  Frontend: "Interfaz reactiva con React + Vite, experiencia fluida y componentes reutilizables.",
  Backend: "API REST con FastAPI, manejo de peticiones de detección y streaming WebSocket.",
  Detección: "Modelos de detección de objetos basados en Ultralytics (YOLOv11n, RT‑DETR) ejecutados con PyTorch y CUDA.",
  "Deep learning": "Entrenamiento y afinado de modelos en PyTorch 2.10, aprovechando GPUs Tesla T4.",
  OCR: "Extracción de texto de placas usando EasyOCR, compatible con múltiples alfabetos.",
  Visión: "Pre‑y post‑procesado de imágenes con OpenCV, ajuste de contraste y normalización.",
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
  { layer: "Frontend",      tech: "React / Next.js",    desc: "UI + routing" },
  { layer: "Backend",       tech: "FastAPI",            desc: "REST · Python 3.11" },
  { layer: "Detección",     tech: "Ultralytics",        desc: "YOLOv11n · RT-DETR-L" },
  { layer: "Deep learning", tech: "PyTorch 2.10",       desc: "CUDA 12.8 · Tesla T4" },
  { layer: "OCR",           tech: "EasyOCR",            desc: "Lectura de caracteres" },
  { layer: "Visión",        tech: "OpenCV",             desc: "Pre y post proceso" },
]

const WHY = [
  { icon: "01", title: "Control de tránsito",       desc: "Automatiza la identificación de vehículos en peajes, parqueaderos y zonas restringidas sin intervención humana." },
  { icon: "02", title: "Fiscalización vial",        desc: "Detecta infracciones y vehículos con registros pendientes en tiempo real, integrándose con bases de datos oficiales." },
  { icon: "03", title: "Inteligencia de movilidad", desc: "Genera datos de flujo vehicular y patrones de tránsito para planificación urbana y optimización de vías." },
  { icon: "04", title: "Alta velocidad",            desc: "Desde 3.7 ms por imagen. Apto para cámaras de vigilancia sin hardware especializado adicional." },
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