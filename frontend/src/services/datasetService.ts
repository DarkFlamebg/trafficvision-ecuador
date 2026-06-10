import API from './api'

export type DatasetFile = {
  id: number
  file_name: string
  file_type?: string | null
  file_path: string
}

export type Dataset = {
  id: number
  name: string
  description?: string | null
  version?: string | null
  created_by?: string | null
  created_at: string
  files: DatasetFile[]
}

export async function getDatasets(): Promise<Dataset[]> {
  const response = await API.get<Dataset[]>('/api/v1/datasets')
  return response.data
}
