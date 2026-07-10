import { useState } from "react"
import { NavLink, useNavigate } from "react-router-dom"
import "./Navbar.css"

export function Navbar() {
  const navigate = useNavigate()
  const [isOpen, setIsOpen] = useState(false)

  const toggleMenu = () => setIsOpen(!isOpen)
  const closeMenu = () => setIsOpen(false)

  return (
    <nav className="global-nav">
      <div className="global-nav-inner">
        <span className="global-nav-logo" onClick={() => { navigate("/"); closeMenu(); }}>
          Traffic<span className="global-nav-accent">Vision</span>
        </span>
        
        {/* Hamburger Icon */}
        <div className={`hamburger ${isOpen ? "open" : ""}`} onClick={toggleMenu}>
          <span></span>
          <span></span>
          <span></span>
        </div>

        <div className={`global-nav-links ${isOpen ? "open" : ""}`}>
          <NavLink to="/" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Inicio
          </NavLink>
          <NavLink to="/read-plate" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Analizar Placa
          </NavLink>
          <NavLink to="/model-comparison" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Comparativa
          </NavLink>
          <NavLink to="/benchmark" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Benchmark
          </NavLink>
          <NavLink to="/control-anticorrupcion" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Registros
          </NavLink>
          <NavLink to="/training-metrics-dynamic" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Metricas
          </NavLink>
          <NavLink to="/resources" className={({ isActive }) => `global-nav-link global-nav-link--download ${isActive ? "active" : ""}`} onClick={closeMenu}>
            Recursos
          </NavLink>
        </div>
      </div>
    </nav>
  )
}

