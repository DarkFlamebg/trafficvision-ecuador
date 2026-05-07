import { useNavigate } from "react-router-dom"
import "./Home.css"

const MODELS = [
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
  {
    id: "rtdetr-ec",
    tag: "RT-DETR EC",
    tagClass: "tag--ec",
    name: "RT-DETR Ecuador",
    arch: "Fine-tune local · 310 layers · 32M params · 103 GFLOPs",
    badge: "Especializado",
    badgeClass: "badge--ec",
    featured: false,
    metrics: [
      { key: "mAP@50",    val: "23.3%", bar: 23.3 },
      { key: "mAP@50-95", val: "17.0%", bar: 17.0 },
      { key: "Precisión", val: "16.7%", bar: 16.7 },
      { key: "Recall",    val: "46.7%", bar: 46.7 },
    ],
    speed: "~70 ms",
    fps: "~14 FPS",
    size: "63 MB",
    note: "Fine-tune sobre placas ecuatorianas. Dataset reducido (21 imgs) limita generalización actual.",
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
  { title: "Control de tránsito",       desc: "Automatiza la identificación de vehículos en peajes, parqueaderos y zonas restringidas sin intervención humana." },
  { title: "Fiscalización vial",        desc: "Detecta infracciones y vehículos con registros pendientes en tiempo real, integrándose con bases de datos oficiales." },
  { title: "Inteligencia de movilidad", desc: "Genera datos de flujo vehicular y patrones de tránsito para planificación urbana y optimización de vías." },
  { title: "Alta velocidad",            desc: "Desde 3.7 ms por imagen. Apto para cámaras de vigilancia sin hardware especializado adicional." },
]

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="home-root">

      {/* NAV */}
      <nav className="home-nav">
        <span className="home-logo">
          Traffic<span className="home-logo-accent">Vision</span>
        </span>
        <a
          className="home-nav-link"
          
          target="_blank"
          rel="noreferrer"
        >
          docs ↗
        </a>
      </nav>

      {/* HERO */}
      <header className="home-hero">
        <div className="home-hero-inner">
          <div className="home-eyebrow">
            <span className="home-dot" aria-hidden="true" />
            Sistema activo · Detección en tiempo real
          </div>

          <h1 className="home-title">
            Reconocimiento automático<br />
            de <span>placas vehiculares</span>
          </h1>

          <p className="home-sub">
            Sistema de visión por computadora que detecta y lee placas en imágenes
            y video usando redes neuronales de última generación. Diseñado para
            control de acceso, fiscalización vial y monitoreo de tránsito urbano.
          </p>

          <div className="home-cta-row">
            <button
              className="home-btn-primary"
              onClick={() => navigate("/read-plate")}
            >
              Analizar placa
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <path d="M2.5 7h9M8 3.5L11.5 7 8 10.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <button
              className="home-btn-primary"
              onClick={() => navigate("/model-comparison")}
            >
              Comparar Modelos IA
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <path d="M2.5 7h9M8 3.5L11.5 7 8 10.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <a
              className="home-btn-ghost"
              
              target="_blank"
              rel="noreferrer"
            >
              Ver docs
            </a>
          </div>

          {/* quick stats */}
          <div className="home-stats">
            {[
              { val: "96.8%",  lbl: "mAP@50 mejor modelo" },
              { val: "3.7 ms", lbl: "Inferencia YOLO11n" },
              { val: "2 048",  lbl: "Imágenes validación" },
              { val: "32 M",   lbl: "Params RT-DETR-L" },
            ].map((s) => (
              <div key={s.lbl} className="home-stat">
                <span className="home-stat-val">{s.val}</span>
                <span className="home-stat-lbl">{s.lbl}</span>
              </div>
            ))}
          </div>
        </div>
      </header>

      <div className="home-divider" />

      {/* WHY */}
      <section className="home-section">
        <div className="home-section-inner">
          <p className="home-section-label">Antecedente</p>
          <h2 className="home-section-title">¿Por qué existe este sistema?</h2>
          <div className="home-why-grid">
            {WHY.map((w) => (
              <div key={w.title} className="home-why-card">
                <h3 className="home-why-title">{w.title}</h3>
                <p className="home-why-desc">{w.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="home-divider" />

      {/* MODELS */}
      <section className="home-section">
        <div className="home-section-inner">
          <p className="home-section-label">Modelos · Resultados reales</p>
          <h2 className="home-section-title">Benchmark comparativo</h2>
          <p className="home-section-sub">
            Evaluación sobre 2 048 imágenes. Hardware: Tesla T4 · CUDA 12.8 · Ultralytics 8.4.46.
          </p>

          <div className="home-models-grid">
            {MODELS.map((m) => (
              <article
                key={m.id}
                className={`home-model-card${m.featured ? " home-model-card--featured" : ""}`}
              >
                <div className="home-model-head">
                  <span className={`home-model-tag ${m.tagClass}`}>{m.tag}</span>
                  <span className={`home-model-badge ${m.badgeClass}`}>{m.badge}</span>
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
                    <span>⚡ {m.speed} · GPU T4</span>
                    <span>{m.fps}</span>
                    <span>{m.size}</span>
                  </div>
                  <p className="home-model-note">{m.note}</p>
                </div>
              </article>
            ))}
          </div>

          <div className="home-callout">
            <span className="home-callout-icon" aria-hidden="true"></span>
            <p>
              <strong>RT-DETR Combined supera al modelo Ecuador en +73.5 pp de mAP@50.</strong>{" "}
              El dataset global aporta generalización crítica. RT-DETR y YOLO11n rinden de forma
              similar (+0.6 pp) — prefiere YOLO11n si la latencia es prioritaria.
            </p>
          </div>
        </div>
      </section>

      <div className="home-divider" />

      {/* STACK */}
      <section className="home-section">
        <div className="home-section-inner">
          <p className="home-section-label">Arquitectura</p>
          <h2 className="home-section-title">Stack tecnológico</h2>
          <div className="home-stack-grid">
            {STACK.map((s) => (
              <div key={s.layer} className="home-stack-item">
                <span className="home-stack-layer">{s.layer}</span>
                <span className="home-stack-tech">{s.tech}</span>
                <span className="home-stack-desc">{s.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="home-footer">
        <span>Backend · <code>localhost:8000</code></span>
        <span>TrafficVision · 2026</span>
      </footer>
    </div>
  )
}