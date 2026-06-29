import { NavLink, useNavigate } from "react-router-dom"
import "./Navbar.css"

export function Navbar() {
  const navigate = useNavigate()

  return (
    <nav className="global-nav">
      <div className="global-nav-inner">
        <span className="global-nav-logo" onClick={() => navigate("/")}>
          Traffic<span className="global-nav-accent">Vision</span>
        </span>
        <div className="global-nav-links">
          <NavLink to="/" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Inicio
          </NavLink>
          <NavLink to="/read-plate" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Analizar Placa
          </NavLink>
          <NavLink to="/model-comparison" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Comparativa
          </NavLink>
          <NavLink to="/benchmark" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Benchmark
          </NavLink>
          <NavLink to="/control-anticorrupcion" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Registros
          </NavLink>
          <NavLink to="/training-metrics-dynamic" className={({ isActive }) => `global-nav-link ${isActive ? "active" : ""}`}>
            Metricas
          </NavLink>
          <NavLink to="/resources" className={({ isActive }) => `global-nav-link global-nav-link--download ${isActive ? "active" : ""}`}>
            Recursos
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
