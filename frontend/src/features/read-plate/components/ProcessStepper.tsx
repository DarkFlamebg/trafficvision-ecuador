interface ProcessStepperProps {
  currentStep: "upload" | "analyze" | "results"
  hasFile: boolean
  loading: boolean
  hasResults: boolean
}

export function ProcessStepper({ currentStep, hasFile, loading, hasResults }: ProcessStepperProps) {
  const steps = [
    { id: "upload", label: "Carga el archivo", icon: "📁" },
    { id: "analyze", label: "Analiza", icon: "🔍" },
    { id: "results", label: "Resultados", icon: "✓" },
  ]

  const getStepStatus = (stepId: string) => {
    if (stepId === "upload") return hasFile ? "complete" : currentStep === "upload" ? "active" : "pending"
    if (stepId === "analyze") return hasResults ? "complete" : loading ? "active" : hasFile ? "pending" : "disabled"
    if (stepId === "results") return hasResults ? "active" : "disabled"
    return "pending"
  }

  return (
    <div className="rp-stepper" role="navigation" aria-label="Progreso del análisis">
      {steps.map((step, index) => {
        const status = getStepStatus(step.id)
        return (
          <div key={step.id} className="rp-stepper-item">
            <div className={`rp-stepper-step rp-stepper-step--${status}`}>
              <div className="rp-stepper-icon">
                {status === "complete" ? (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8L6.5 11.5L13 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : status === "active" && loading ? (
                  <span className="rp-stepper-spinner" />
                ) : (
                  <span style={{ fontSize: "1.1rem" }}>{step.icon}</span>
                )}
              </div>
              <span className="rp-stepper-label">{step.label}</span>
            </div>
            {index < steps.length - 1 && (
              <div className={`rp-stepper-line rp-stepper-line--${status === "complete" ? "complete" : "pending"}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}