import { BrowserRouter, Routes, Route } from "react-router-dom"

import Home from "./pages/Home"
import ReadPlate from "./pages/ReadPlate"
import ModelComparison from "./pages/ModelComparison"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/read-plate" element={<ReadPlate />} />
        <Route path="/model-comparison" element={<ModelComparison />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App