import React, { Suspense } from "react"
import { BrowserRouter, Routes, Route } from "react-router-dom"

import { Layout } from "./components/Layout"

// Route-level Code Splitting for optimal initial load time
const Home = React.lazy(() => import("./features/home/Home"))
const ReadPlate = React.lazy(() => import("./features/read-plate/ReadPlate"))
const ModelComparison = React.lazy(() => import("./features/model-comparison/ModelComparison"))
const Benchmark = React.lazy(() => import("./features/benchmark/Benchmark"))
const TrainingMetrics = React.lazy(() => import("./features/training-metrics/TrainingMetrics"))
const Resources = React.lazy(() => import("./features/resources/Resources"))

const PageLoader = () => (
  <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh", background: "var(--bg, #0a0a0b)" }}>
    <div style={{ width: 24, height: 24, border: "2px solid rgba(59, 130, 246, 0.2)", borderTopColor: "#3b82f6", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
  </div>
)

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/read-plate" element={<ReadPlate />} />
            <Route path="/model-comparison" element={<ModelComparison />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/training-metrics" element={<TrainingMetrics />} />
            <Route path="/resources" element={<Resources />} />
          </Routes>
        </Suspense>
      </Layout>
    </BrowserRouter>
  )
}

export default App