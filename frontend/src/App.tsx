import { BrowserRouter, Routes, Route } from "react-router-dom"

import Home from "./pages/Home"
import ReadPlate from "./pages/ReadPlate"
import ModelComparison from "./pages/ModelComparison"
import Benchmark from "./pages/Benchmark"
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
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App