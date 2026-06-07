import { ReactNode } from "react"
import { Navbar } from "./Navbar"
import "./Layout.css"

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="global-layout">
      <Navbar />
      <main className="global-main">
        {children}
      </main>
    </div>
  )
}
