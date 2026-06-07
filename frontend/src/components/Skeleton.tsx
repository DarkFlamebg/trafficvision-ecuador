import type { CSSProperties } from "react"
import "./Skeleton.css"

interface SkeletonProps {
  width?: string | number
  height?: string | number
  borderRadius?: string | number
  className?: string
  style?: CSSProperties
}

export function Skeleton({ width = "100%", height = "1rem", borderRadius = "6px", className = "", style }: SkeletonProps) {
  return (
    <div 
      className={`skeleton-base ${className}`} 
      style={{ width, height, borderRadius, ...style }}
    />
  )
}

export function SkeletonBlock({ lines = 3 }: { lines?: number }) {
  return (
    <div className="skeleton-block">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton 
          key={i} 
          height="1rem" 
          width={i === lines - 1 ? "60%" : "100%"} 
          className="skeleton-mb"
        />
      ))}
    </div>
  )
}
