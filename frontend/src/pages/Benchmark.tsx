// src/pages/Benchmark.tsx
// Página de Benchmark en Tiempo Real — TrafficVision

import { useState, useRef, useCallback, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import "./Benchmark.css"

// ── Tipos ──────────────────────────────────────────────────────────────────

interface SingleResult {
  detections: number
  inference_time_ms: number
  memory_mb: number
  error?: string
}

interface ProgressMsg {
  type: "progress"
  current: number
  total: number
  image: string
  device: string
  results: {
    yolo: SingleResult
    rtdetr: SingleResult
    efficientdet: SingleResult
  }
}

interface DoneMsg {
  type: "done"
  summary: {
    yolo: Summary
    rtdetr: Summary
    efficientdet: Summary
    device: string
  }
}

interface ErrorMsg {
  type: "error"
  message: string
}

interface Summary {
  avg_time_ms: number
  avg_memory_mb: number
  total_detections: number
}

type WsMsg = ProgressMsg | DoneMsg | ErrorMsg

interface FeedItem {
  image: string
  yolo_ms: number
  rtdetr_ms: number
  effdet_ms: number
}

// ── Helpers ────────────────────────────────────────────────────────────────

const NUM_IMAGES = 10

const WS_URL = `ws://localhost:8000/api/v1/benchmark/ws?num_images=${NUM_IMAGES}`

function getFastest(s: { yolo: Summary; rtdetr: Summary; efficientdet: Summary }) {
  const entries = [
    { key: "YOLOv11n",       ms: s.yolo.avg_time_ms },
    { key: "RT-DETR",        ms: s.rtdetr.avg_time_ms },
    { key: "EfficientDet-D2", ms: s.efficientdet.avg_time_ms },
  ]
  return entries.reduce((a, b) => (a.ms < b.ms ? a : b))
}

function fmtMs(ms: number) {
  if (ms <= 0) return "—"
  return ms > 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`
}

// ── Component ──────────────────────────────────────────────────────────────

export default function Benchmark() {
  const navigate = useNavigate()

  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [current, setCurrent] = useState(0)
  const [total, setTotal]   = useState(NUM_IMAGES)
  const [feed, setFeed]     = useState<FeedItem[]>([])
  const [summary, setSummary] = useState<DoneMsg["summary"] | null>(null)
  const [error, setError]   = useState<string | null>(null)
  const [device, setDevice] = useState<string>("Detectando...")

  const wsRef = useRef<WebSocket | null>(null)
  const feedEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll feed
  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [feed])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  const startBenchmark = useCallback(() => {
    // Reset state
    setRunning(true)
    setProgress(0)
    setCurrent(0)
    setTotal(NUM_IMAGES)
    setFeed([])
    setSummary(null)
    setError(null)

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      console.log("[Benchmark] WebSocket conectado")
    }

    ws.onmessage = (evt) => {
      const msg: WsMsg = JSON.parse(evt.data)

      if (msg.type === "progress") {
        const p = msg as ProgressMsg
        setCurrent(p.current)
        setTotal(p.total)
        setProgress(Math.round((p.current / p.total) * 100))
        if (p.device) setDevice(p.device)

        setFeed((prev) => [
          ...prev,
          {
            image:    p.image,
            yolo_ms:  p.results.yolo.inference_time_ms,
            rtdetr_ms: p.results.rtdetr.inference_time_ms,
            effdet_ms: p.results.efficientdet.inference_time_ms,
          },
        ])
      } else if (msg.type === "done") {
        const d = msg as DoneMsg
        setSummary(d.summary)
        if (d.summary.device) setDevice(d.summary.device)
        setRunning(false)
        setProgress(100)
        ws.close()
      } else if (msg.type === "error") {
        const e = msg as ErrorMsg
        setError(e.message)
        setRunning(false)
        ws.close()
      }
    }

    ws.onerror = () => {
      setError("No se pudo conectar con el servidor. Verifica que el backend esté activo en localhost:8000.")
      setRunning(false)
    }

    ws.onclose = () => {
      if (running) {
        setRunning(false)
      }
    }
  }, [running])

  // ── Bar chart max values ──────────────────────────────────────────────
  const maxTime = summary
    ? Math.max(summary.yolo.avg_time_ms, summary.rtdetr.avg_time_ms, summary.efficientdet.avg_time_ms, 1)
    : 1

  const maxDet = summary
    ? Math.max(summary.yolo.total_detections, summary.rtdetr.total_detections, summary.efficientdet.total_detections, 1)
    : 1

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="bm-root">

      {/* NAV */}
      <nav className="bm-nav">
        <span className="bm-logo">
          Traffic<span className="bm-logo-accent">Vision</span>
        </span>
        <button className="bm-nav-back" onClick={() => navigate("/")}>
          ← Volver al inicio
        </button>
      </nav>

      {/* MAIN */}
      <main className="bm-main">

        {/* Header */}
        <div className="bm-header">
          <div className="bm-eyebrow">
            <span className="bm-dot" aria-hidden="true" />
            Evaluación cuantitativa · Sprint 5
          </div>
          <h1 className="bm-title">Benchmark de Modelos</h1>
          <p className="bm-sub">
            Ejecuta los tres detectores sobre <strong>{NUM_IMAGES} imágenes</strong> de prueba
            y compara tiempos de inferencia y detecciones en tiempo real.
          </p>
        </div>

        {/* Control Panel */}
        <div className="bm-panel">
          <div className="bm-panel-info">
            <span className="bm-panel-label">Configuración</span>
            <span className="bm-panel-value">
              <strong>{NUM_IMAGES} imágenes</strong> &nbsp;·&nbsp; YOLOv11n · RT-DETR · EfficientDet-D2
            </span>
          </div>
          <div className="bm-panel-info">
            <span className="bm-panel-label">Dataset</span>
            <span className="bm-panel-value">license-plates-ec-combined / test</span>
          </div>
          <div className="bm-panel-info">
            <span className="bm-panel-label">Hardware actual</span>
            <span className="bm-panel-value" style={{ color: device.includes("GPU") ? "#34d399" : "#fbbf24" }}>
              {device}
            </span>
          </div>
          <button
            id="btn-run-benchmark"
            className="bm-btn-run"
            onClick={startBenchmark}
            disabled={running}
          >
            {running ? (
              <>
                <span className="bm-btn-spinner" />
                Procesando…
              </>
            ) : (
              <>
                ▶ Ejecutar Benchmark
              </>
            )}
          </button>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bm-error" role="alert">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* Progress */}
        {(running || (feed.length > 0 && !summary)) && (
          <div className="bm-progress-wrap">
            <div className="bm-progress-header">
              <span className="bm-progress-title">
                {running ? "Procesando imágenes…" : "Completado"}
              </span>
              <span className="bm-progress-counter">{current} / {total}</span>
            </div>

            <div className="bm-progress-bar-bg">
              <div
                className="bm-progress-bar-fill"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* Feed */}
            <div className="bm-feed" aria-live="polite">
              {feed.map((item, i) => (
                <div className="bm-feed-item" key={i}>
                  <span className="bm-feed-name">{item.image}</span>
                  <span className="bm-feed-chip bm-feed-chip--yolo">
                    YOLO · {fmtMs(item.yolo_ms)}
                  </span>
                  <span className="bm-feed-chip bm-feed-chip--rtdetr">
                    RT-DETR · {fmtMs(item.rtdetr_ms)}
                  </span>
                  <span className="bm-feed-chip bm-feed-chip--effdet">
                    EffDet · {fmtMs(item.effdet_ms)}
                  </span>
                </div>
              ))}
              <div ref={feedEndRef} />
            </div>
          </div>
        )}

        {/* Idle state */}
        {!running && feed.length === 0 && !error && !summary && (
          <div className="bm-idle">
            <span className="bm-idle-icon">📊</span>
            <p className="bm-idle-text">
              Presiona <strong>Ejecutar Benchmark</strong> para iniciar la evaluación.
            </p>
          </div>
        )}

        {/* Results */}
        {summary && (
          <div className="bm-results">

            <div>
              <h2 className="bm-results-title">Resultados del Benchmark</h2>
              <p className="bm-results-sub">
                Promedio calculado sobre {NUM_IMAGES} imágenes de prueba del conjunto EC-Combined.
              </p>
            </div>

            {/* Summary Cards */}
            <div className="bm-cards">
              {(
                [
                  { key: "yolo",        label: "YOLOv11n",        cls: "yolo",   badge: "Más rápido", s: summary.yolo },
                  { key: "rtdetr",      label: "RT-DETR",         cls: "rtdetr", badge: "Mejor mAP",  s: summary.rtdetr },
                  { key: "efficientdet", label: "EfficientDet-D2", cls: "effdet", badge: "Alta precisión", s: summary.efficientdet },
                ] as const
              ).map(({ label, cls, badge, s }) => (
                <div key={cls} className={`bm-card bm-card--${cls}`}>
                  <div className="bm-card-header">
                    <span className="bm-card-model">{label}</span>
                    <span className="bm-card-badge">{badge}</span>
                  </div>
                  <div className="bm-card-metrics">
                    <div className="bm-card-metric">
                      <span className="bm-card-metric-key">Tiempo promedio</span>
                      <span className="bm-card-metric-val">{fmtMs(s.avg_time_ms)}</span>
                    </div>
                    <div className="bm-card-metric">
                      <span className="bm-card-metric-key">FPS equivalente</span>
                      <span className="bm-card-metric-val">
                        {s.avg_time_ms > 0 ? `${(1000 / s.avg_time_ms).toFixed(1)} FPS` : "—"}
                      </span>
                    </div>
                    <div className="bm-card-metric">
                      <span className="bm-card-metric-key">Total detecciones</span>
                      <span className="bm-card-metric-val">{s.total_detections}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Bar Charts */}
            <div className="bm-charts">

              {/* Tiempo */}
              <div className="bm-chart-box">
                <p className="bm-chart-title">⏱ Tiempo promedio (ms) — menor es mejor</p>
                <div className="bm-bar-group">
                  {[
                    { label: "YOLOv11n",        val: summary.yolo.avg_time_ms,        cls: "yolo" },
                    { label: "RT-DETR",         val: summary.rtdetr.avg_time_ms,      cls: "rtdetr" },
                    { label: "EfficientDet-D2", val: summary.efficientdet.avg_time_ms, cls: "effdet" },
                  ].map(({ label, val, cls }) => (
                    <div className="bm-bar-row" key={label}>
                      <span className="bm-bar-label">{label}</span>
                      <div className="bm-bar-track">
                        <div
                          className={`bm-bar-fill bm-bar-fill--${cls}`}
                          style={{ width: `${(val / maxTime) * 100}%` }}
                        />
                      </div>
                      <span className="bm-bar-val">{fmtMs(val)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Detecciones */}
              <div className="bm-chart-box">
                <p className="bm-chart-title">🎯 Total detecciones — mayor cobertura</p>
                <div className="bm-bar-group">
                  {[
                    { label: "YOLOv11n",        val: summary.yolo.total_detections,        cls: "yolo" },
                    { label: "RT-DETR",         val: summary.rtdetr.total_detections,      cls: "rtdetr" },
                    { label: "EfficientDet-D2", val: summary.efficientdet.total_detections, cls: "effdet" },
                  ].map(({ label, val, cls }) => (
                    <div className="bm-bar-row" key={label}>
                      <span className="bm-bar-label">{label}</span>
                      <div className="bm-bar-track">
                        <div
                          className={`bm-bar-fill bm-bar-fill--${cls}`}
                          style={{ width: `${(val / maxDet) * 100}%` }}
                        />
                      </div>
                      <span className="bm-bar-val">{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Winner Banner */}
            {(() => {
              const fastest = getFastest(summary)
              return (
                <div className="bm-winner-banner">
                  <span className="bm-winner-icon">🏆</span>
                  <span>
                    <strong>{fastest.key}</strong> fue el modelo más rápido con un promedio de{" "}
                    <strong>{fmtMs(fastest.ms)}</strong> por imagen.
                  </span>
                </div>
              )
            })()}

          </div>
        )}

      </main>

      {/* FOOTER */}
      <footer className="bm-footer">
        <span>Backend · <code>localhost:8000</code></span>
        <span>TrafficVision · 2026</span>
      </footer>

    </div>
  )
}
