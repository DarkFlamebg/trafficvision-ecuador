import axios from 'axios'

export const getBaseUrl = () => {
  return import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
}

export const getWsUrl = () => {
  const baseUrl = getBaseUrl();
  return baseUrl.startsWith('https') 
    ? baseUrl.replace(/^https/, 'wss') 
    : baseUrl.replace(/^http/, 'ws');
}

const API = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

export default API