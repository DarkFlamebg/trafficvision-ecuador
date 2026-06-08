import { BrowserRouter, Routes, Route } from "react-router-dom"

import Home from "./features/home/Home"
import ReadPlate from "./features/read-plate/ReadPlate"
import ModelComparison from "./features/model-comparison/ModelComparison"
import Benchmark from "./features/benchmark/Benchmark"
import TrainingMetrics from "./features/training-metrics/TrainingMetrics"
import { Layout } from "./components/Layout"

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/read-plate" element={<ReadPlate />} />
          <Route path="/model-comparison" element={<ModelComparison />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/training-metrics" element={<TrainingMetrics />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App