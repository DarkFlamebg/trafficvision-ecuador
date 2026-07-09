# TrafficVision - Frontend 🖥️📊

Interfaz de usuario interactiva y panel de control (dashboard) para el sistema **TrafficVision**. Permite a los usuarios visualizar en tiempo real o bajo demanda el análisis de tráfico, estadísticas, y flujos de vehículos.

## 🛠️ Stack Tecnológico
- **Core:** React 19, TypeScript
- **Build Tool:** Vite
- **UI / Estilos:** Material UI (MUI), Emotion
- **Gráficos y Visualización:** Chart.js, React-Chartjs-2, XYFlow
- **Procesamiento Multimedia:** FFmpeg (WebAssembly)

## ✨ Características Principales
- **Dashboard Estadístico:** Visualización de métricas clave (tipos de vehículos, densidad de tráfico) mediante gráficos interactivos.
- **Procesamiento de Video en Cliente:** Uso de FFmpeg en el navegador para optimizar y pre-procesar medios antes de subirlos.
- **Diagramas de Flujo interactivos:** Integración de XYFlow para representar reglas lógicas o flujos de tráfico.
- **Diseño Responsivo y Moderno:** Componentes pulidos utilizando Material UI (MUI) garantizando la accesibilidad.
- **Conexión en Tiempo Real:** Recepción de eventos desde el backend para actualizar conteos en vivo.

## 🚀 Instalación y Configuración Local

1. **Entrar al directorio del frontend:**
   ```bash
   cd TrafficVision/frontend
   ```

2. **Instalar dependencias:**
   *Asegúrate de tener Node.js instalado.*
   ```bash
   npm install
   ```

3. **Variables de entorno:**
   Crea un archivo `.env` (si aplica) para conectar con el backend local:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Levantar el entorno de desarrollo:**
   ```bash
   npm run dev
   ```
   *La aplicación estará disponible típicamente en `http://localhost:5173`.*

## 📜 Scripts Disponibles

- `npm run dev`: Inicia el servidor de desarrollo local con Vite (Hot Module Replacement).
- `npm run build`: Transpila TypeScript y empaqueta la aplicación para producción.
- `npm run preview`: Sirve localmente los archivos generados en la carpeta dist (para probar el build).
- `npm run lint`: Ejecuta ESLint para buscar y corregir problemas en el código fuente.
