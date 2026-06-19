import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

/* ─── Override estilos internos de React Flow para dark mode ─── */
const darkOverride = `
  /* Botones de controles */
  .react-flow__controls {
    background: #10101e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    overflow: hidden;
  }
  .react-flow__controls-button {
    background: #10101e !important;
    border: none !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.7) !important;
    fill: rgba(255,255,255,0.7) !important;
    width: 28px !important;
    height: 28px !important;
    padding: 5px !important;
    transition: background 0.15s, fill 0.15s !important;
  }
  .react-flow__controls-button:hover {
    background: rgba(255,255,255,0.08) !important;
    fill: #fff !important;
  }
  .react-flow__controls-button:last-child {
    border-bottom: none !important;
  }
  .react-flow__controls-button svg {
    fill: inherit !important;
  }

  /* Handles (bolitas de conexión): ocultar en modo lectura */
  .react-flow__handle {
    opacity: 0 !important;
    pointer-events: none !important;
  }

  /* Edge labels - fondo oscuro */
  .react-flow__edge-text {
    fill: rgba(220,225,255,0.85) !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
  }
  .react-flow__edge-textbg {
    fill: rgba(10,10,25,0.82) !important;
  }

  /* Nodos del grupo - sin sombra blanca */
  .react-flow__node-group {
    box-shadow: none !important;
  }

  /* Mini mapa si lo habilitamos */
  .react-flow__minimap {
    background: #0a0a18 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
  }
`;

/* ─── Estilos de nodos hijos ────────────────────────────────────── */
const base = {
  borderRadius: '8px',
  padding: '9px 11px',
  fontSize: '12px',
  textAlign: 'center' as const,
  lineHeight: '1.45',
  fontFamily: "'Inter', system-ui, sans-serif",
  width: 138,
};

const styleRF = { ...base, background: '#2e0a50', color: '#e8d5ff', border: '1.5px solid #a855f7' };
const styleCL = { ...base, background: '#2d1800', color: '#fde68a', border: '1.5px solid #f59e0b' };
const styleBE = { ...base, background: '#041a0c', color: '#a7f3d0', border: '1.5px solid #10b981' };
const styleFE = { ...base, background: '#020d22', color: '#bfdbfe', border: '1.5px solid #3b82f6' };
const styleDB = { ...base, background: '#0e1520', color: '#e2e8f0', border: '1.5px solid #64748b' };

/* ─── Estilos de grupos ─────────────────────────────────────────── */
const gBase = {
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: '700' as const,
  letterSpacing: '0.08em',
  textTransform: 'uppercase' as const,
  fontFamily: "'Inter', system-ui, sans-serif",
};

const gRF = { ...gBase, background: 'rgba(110,20,180,0.2)',  border: '1.5px solid rgba(168,85,247,0.6)',  color: '#d8b4fe' };
const gCL = { ...gBase, background: 'rgba(160,70,0,0.2)',    border: '1.5px solid rgba(245,158,11,0.6)',  color: '#fcd34d' };
const gBE = { ...gBase, background: 'rgba(5,80,35,0.22)',    border: '1.5px solid rgba(16,185,129,0.6)',  color: '#6ee7b7' };
const gFE = { ...gBase, background: 'rgba(5,30,100,0.22)',   border: '1.5px solid rgba(59,130,246,0.6)',  color: '#93c5fd' };

/* ─── Nodos con posiciones de Handles configuradas ──────────────── */
const initialNodes: Node[] = [
  /* ══ ① ROBOFLOW ═══════════════════════════════════════════════ */
  {
    id: 'g-rf', type: 'group',
    position: { x: 50, y: 20 },
    style: { ...gRF, width: 985, height: 112 },
    data: { label: '① Procesamiento de Datos — Roboflow' },
  },
  { id: 'rf1', parentId: 'g-rf', extent: 'parent', position: { x: 20,  y: 34 }, style: styleRF, data: { label: 'Recolección\nImágenes' }, sourcePosition: 'right' as any },
  { id: 'rf2', parentId: 'g-rf', extent: 'parent', position: { x: 215, y: 34 }, style: styleRF, data: { label: 'Limpieza\ny Curación' }, sourcePosition: 'right' as any, targetPosition: 'left' as any },
  { id: 'rf3', parentId: 'g-rf', extent: 'parent', position: { x: 410, y: 34 }, style: styleRF, data: { label: 'Anotación\nBounding Boxes' }, sourcePosition: 'right' as any, targetPosition: 'left' as any },
  { id: 'rf4', parentId: 'g-rf', extent: 'parent', position: { x: 615, y: 34 }, style: styleRF, data: { label: 'Aumentación\nde Datos' }, sourcePosition: 'right' as any, targetPosition: 'left' as any },
  { id: 'rf5', parentId: 'g-rf', extent: 'parent', position: { x: 820, y: 34 }, style: styleRF, data: { label: 'Exportación\nYOLO / COCO' }, sourcePosition: 'bottom' as any, targetPosition: 'left' as any },

  /* ══ ② GOOGLE COLAB ═══════════════════════════════════════════ */
  {
    id: 'g-cl', type: 'group',
    position: { x: 580, y: 200 },
    style: { ...gCL, width: 455, height: 380 },
    data: { label: '② Entrenamiento — Google Colab · GPU T4' },
  },
  { id: 'ds',  parentId: 'g-cl', extent: 'parent', position: { x: 158, y: 40 },  style: styleCL, data: { label: 'Descarga\nDataset API' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'm1',  parentId: 'g-cl', extent: 'parent', position: { x: 14,  y: 135 }, style: styleCL, data: { label: 'YOLOv11n\nVelocidad' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'm2',  parentId: 'g-cl', extent: 'parent', position: { x: 158, y: 135 }, style: styleCL, data: { label: 'RT-DETR-L\nAlta Precisión' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'm3',  parentId: 'g-cl', extent: 'parent', position: { x: 302, y: 135 }, style: styleCL, data: { label: 'Vision Mamba\nVim-Tiny' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'ev',  parentId: 'g-cl', extent: 'parent', position: { x: 158, y: 235 }, style: styleCL, data: { label: 'Evaluación\ny Métricas' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'ex',  parentId: 'g-cl', extent: 'parent', position: { x: 158, y: 310 }, style: styleCL, data: { label: 'Exportar\nPesos .pt/.pth' }, sourcePosition: 'left' as any, targetPosition: 'top' as any },

  /* ══ ④ FRONTEND UI ════════════════════════════════════════════ */
  {
    id: 'g-fe', type: 'group',
    position: { x: 50, y: 200 },
    style: { ...gFE, width: 480, height: 230 },
    data: { label: '④ Frontend — React / Vite' },
  },
  { id: 'fe1', parentId: 'g-fe', extent: 'parent', position: { x: 171, y: 45 },  style: styleFE, data: { label: 'Dashboard\nPrincipal' }, sourcePosition: 'bottom' as any },
  { id: 'fe2', parentId: 'g-fe', extent: 'parent', position: { x: 20,  y: 145 }, style: styleFE, data: { label: 'Carga de\nArchivos' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'fe3', parentId: 'g-fe', extent: 'parent', position: { x: 171, y: 145 }, style: styleFE, data: { label: 'Tabla de\nDetecciones' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },
  { id: 'fe4', parentId: 'g-fe', extent: 'parent', position: { x: 322, y: 145 }, style: styleFE, data: { label: 'Descarga\nReportes' }, sourcePosition: 'bottom' as any, targetPosition: 'top' as any },

  /* ══ ③ BACKEND API ════════════════════════════════════════════ */
  {
    id: 'g-be', type: 'group',
    position: { x: 300, y: 640 },
    style: { ...gBE, width: 490, height: 230 },
    data: { label: '③ Backend — FastAPI + PyTorch' },
  },
  { id: 'api', parentId: 'g-be', extent: 'parent', position: { x: 25,  y: 45 },  style: styleBE, data: { label: 'API REST\nEndpoints' }, sourcePosition: 'right' as any, targetPosition: 'top' as any },
  { id: 'sel', parentId: 'g-be', extent: 'parent', position: { x: 175, y: 45 },  style: styleBE, data: { label: 'Selección\nde Modelo' }, sourcePosition: 'right' as any, targetPosition: 'left' as any },
  { id: 'inf', parentId: 'g-be', extent: 'parent', position: { x: 325, y: 45 },  style: styleBE, data: { label: 'Detección\nOpenCV+PyTorch' }, sourcePosition: 'bottom' as any, targetPosition: 'left' as any },
  { id: 'ocr', parentId: 'g-be', extent: 'parent', position: { x: 325, y: 145 }, style: styleBE, data: { label: 'EasyOCR\nExtracción' }, sourcePosition: 'left' as any, targetPosition: 'top' as any },
  { id: 'dbs', parentId: 'g-be', extent: 'parent', position: { x: 175, y: 145 }, style: styleDB, data: { label: 'PostgreSQL\nBase de Datos' }, targetPosition: 'right' as any },
];

/* ─── Helper aristas ───────────────────────────────────────────── */
const mkEdge = (
  id: string,
  source: string,
  target: string,
  color: string,
  opts: Partial<Edge> = {},
): Edge => ({
  id,
  source,
  target,
  type: 'default',
  markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
  style: { stroke: color, strokeWidth: 1.6 },
  labelStyle: {
    fontSize: '12px',
    fill: 'rgba(220,225,255,0.85)',
    fontFamily: "'Inter',sans-serif",
    fontWeight: 600,
  },
  labelBgStyle: { fill: 'rgba(8,8,20,0.85)', stroke: color, strokeWidth: 0.6 },
  labelBgPadding: [4, 7] as [number, number],
  labelBgBorderRadius: 4,
  ...opts,
});

const initialEdges: Edge[] = [
  /* Roboflow lineal */
  mkEdge('e-r1-r2', 'rf1', 'rf2', '#a855f7'),
  mkEdge('e-r2-r3', 'rf2', 'rf3', '#a855f7'),
  mkEdge('e-r3-r4', 'rf3', 'rf4', '#a855f7'),
  mkEdge('e-r4-r5', 'rf4', 'rf5', '#a855f7'),

  /* Roboflow → Colab */
  mkEdge('e-r5-ds', 'rf5', 'ds', '#f59e0b', { animated: true, label: 'API Key' }),

  /* Colab interno */
  mkEdge('e-ds-m1', 'ds', 'm1', '#f59e0b'),
  mkEdge('e-ds-m2', 'ds', 'm2', '#f59e0b'),
  mkEdge('e-ds-m3', 'ds', 'm3', '#f59e0b'),
  mkEdge('e-m1-ev', 'm1', 'ev', '#f59e0b'),
  mkEdge('e-m2-ev', 'm2', 'ev', '#f59e0b'),
  mkEdge('e-m3-ev', 'm3', 'ev', '#f59e0b'),
  mkEdge('e-ev-ex', 'ev', 'ex', '#f59e0b'),

  /* Colab → Backend (Inyección lateral de pesos entrenados) */
  mkEdge('e-ex-api', 'ex', 'api', '#10b981', { animated: true, label: 'Integración de Pesos' }),

  /* Backend interno */
  mkEdge('e-api-sel', 'api', 'sel', '#10b981'),
  mkEdge('e-sel-inf', 'sel', 'inf', '#10b981'),
  mkEdge('e-inf-ocr', 'inf', 'ocr', '#10b981'),
  mkEdge('e-ocr-db',  'ocr', 'dbs', '#64748b', { animated: true, label: 'Almacena' }),
  mkEdge('e-api-db',  'api', 'dbs', '#64748b', {
    style: { stroke: '#475569', strokeWidth: 1.4, strokeDasharray: '5,4' },
    label: 'Historial',
    markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b', width: 14, height: 14 },
  }),

  /* Frontend interno */
  mkEdge('e-fe1-fe2', 'fe1', 'fe2', '#3b82f6'),
  mkEdge('e-fe1-fe3', 'fe1', 'fe3', '#3b82f6'),
  mkEdge('e-fe1-fe4', 'fe1', 'fe4', '#3b82f6'),

  /* Frontend → Backend */
  mkEdge('e-fe2-api', 'fe2', 'api', '#3b82f6', { animated: true, label: '/upload' }),
  mkEdge('e-fe3-api', 'fe3', 'api', '#3b82f6', { animated: true, label: '/detections' }),
  mkEdge('e-fe4-api', 'fe4', 'api', '#3b82f6', { animated: true, label: '/report' }),
];

/* ─── Componente ─────────────────────────────────────────────────── */
export default function ArchitectureDiagram() {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  return (
    <>
      <style>{darkOverride}</style>
      <div
        style={{
          height: '520px',
          width: '100%',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '16px',
          background: '#05050f',
          overflow: 'hidden',
          boxShadow: '0 24px 80px rgba(100,60,200,0.14)',
        }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={{ padding: 0.1 }}
          minZoom={0.2}
          maxZoom={2.5}
          nodesDraggable={true}
          nodesConnectable={false}
          elementsSelectable={true}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={24}
            size={1}
            color="rgba(255,255,255,0.03)"
          />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </>
  );
}