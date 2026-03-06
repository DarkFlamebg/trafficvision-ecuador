import { useEffect, useState } from "react"
import API from "../services/api"
import type { Dataset } from "../types/dataset"


export function useDatasets() {

  const [datasets, setDatasets] = useState<Dataset[]>([])

  useEffect(() => {
    API.get("/datasets")
      .then(res => setDatasets(res.data))
  }, [])

  return datasets
}