# app/routes/benchmark.py
# Endpoint para ejecutar el benchmark en tiempo real vía WebSocket

import os
import asyncio
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.ai.benchmark_detectors import BenchmarkRunner

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])

# Construir ruta absoluta al dataset (raíz del proyecto -> ml/...)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "assets", "license-plates-ec-combined", "test", "images")


@router.websocket("/ws")
async def benchmark_ws(websocket: WebSocket, num_images: int = 10):
    """
    Ejecuta el benchmark en tiempo real.
    El cliente se conecta y opcionalmente pasa num_images (default 10).
    Emite el progreso por cada imagen procesada y un resumen al final.
    """
    await websocket.accept()

    try:
        # Verificar que el directorio existe
        if not os.path.exists(TEST_IMAGES_DIR):
            await websocket.send_json({
                "type": "error",
                "message": f"Directorio de test no encontrado: {TEST_IMAGES_DIR}"
            })
            await websocket.close()
            return

        runner = BenchmarkRunner(images_dir=TEST_IMAGES_DIR, ground_truth_file=None)
        
        # Cola para comunicar el hilo de procesamiento con el websocket
        queue = asyncio.Queue()

        # Capturar el event loop de la petición principal
        loop = asyncio.get_running_loop()

        def progress_callback(current, total, img_name, results):
            # Limpiar resultados eliminando datos no serializables como las imágenes recortadas
            clean_results = {}
            for model, r in results.items():
                out_key = "mamba" if model == "vm" else model
                clean_results[out_key] = {
                    "detections": r.get("detections", 0),
                    "inference_time_ms": r.get("inference_time_ms", 0.0),
                    "memory_mb": r.get("memory_mb", 0.0)
                }
                if "error" in r:
                    clean_results[out_key]["error"] = r["error"]

            # Enviar a la cola usando el loop de la petición principal
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "image": img_name,
                    "results": clean_results,
                    "device": "GPU (CUDA)" if torch.cuda.is_available() else "CPU Local"
                }),
                loop
            )

        # Iniciar el procesamiento pesado en un hilo separado
        task = loop.run_in_executor(None, runner.run_batch, num_images, progress_callback)

        # Consumir la cola y enviar al websocket mientras la tarea corre
        while not task.done():
            try:
                # Esperar mensajes de la cola con timeout para chequear si task terminó
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                await websocket.send_json(msg)
                queue.task_done()
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                print("[Benchmark] Cliente desconectado prematuramente.")
                return

        # Procesar los últimos mensajes que pudieran quedar en la cola
        while not queue.empty():
            msg = await queue.get()
            await websocket.send_json(msg)
            queue.task_done()

        # Obtener los resultados finales
        final_results = task.result()
        
        # Calcular los promedios para el frontend
        summary = {
            "yolo": _calculate_averages(final_results["yolo"]),
            "rtdetr": _calculate_averages(final_results["rtdetr"]),
            "mamba": _calculate_averages(final_results["vm"]),
            "device": "GPU (CUDA)" if torch.cuda.is_available() else "CPU Local"
        }

        await websocket.send_json({
            "type": "done",
            "summary": summary
        })

    except WebSocketDisconnect:
        print("[Benchmark] Cliente desconectado.")
    except Exception as e:
        print(f"[Benchmark] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


def _calculate_averages(results_list):
    """Calcula promedios de tiempo, memoria y detecciones totales."""
    if not results_list:
        return {"avg_time_ms": 0, "avg_memory_mb": 0, "total_detections": 0}
    
    valid = [r for r in results_list if "error" not in r]
    if not valid:
        return {"avg_time_ms": 0, "avg_memory_mb": 0, "total_detections": 0}

    times = [r["inference_time_ms"] for r in valid]
    mems = [r["memory_mb"] for r in valid]
    dets = [r["detections"] for r in valid]

    return {
        "avg_time_ms": round(sum(times) / len(times), 2),
        "avg_memory_mb": round(sum(mems) / len(mems), 2),
        "total_detections": sum(dets)
    }
