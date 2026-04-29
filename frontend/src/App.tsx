import { BrowserRouter, Routes, Route } from "react-router-dom"

import Home from "./pages/Home"
import ReadPlate from "./pages/ReadPlate"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/read-plate" element={<ReadPlate />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App