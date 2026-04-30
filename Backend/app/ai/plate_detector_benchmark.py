# app/ai/plate_detector_benchmark.py
# Script para comparar rendimiento entre YOLOv11n y RT-DETR

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from plate_detector import detect_plate as detect_yolo
except ImportError:
    print("⚠️  No se pudo importar plate_detector.py (YOLO)")
    detect_yolo = None

try:
    from plate_detector_rtdetr import detect_plate_rtdetr, detect_plate_ensemble
except ImportError:
    print("⚠️  No se pudo importar plate_detector_rtdetr.py")
    detect_plate_rtdetr = None
    detect_plate_ensemble = None


class BenchmarkResults:
    """Almacena y analiza resultados de benchmarking."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.detections: List[int] = []
        self.inference_times: List[float] = []
        self.confidences: List[float] = []
        self.errors: int = 0
    
    def add_result(self, num_detections: int, inference_time: float, confidences: List[float]):
        """Registra resultado de una inferencia."""
        self.detections.append(num_detections)
        self.inference_times.append(inference_time)
        self.confidences.extend(confidences)
    
    def add_error(self):
        """Registra un error de inferencia."""
        self.errors += 1
    
    def get_summary(self) -> Dict:
        """Retorna resumen estadístico."""
        if not self.inference_times:
            return {
                "model": self.model_name,
                "total_images": 0,
                "errors": self.errors,
                "status": "No data"
            }
        
        return {
            "model": self.model_name,
            "total_images": len(self.inference_times),
            "errors": self.errors,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "total_detections": sum(self.detections),
            "avg_detections_per_image": np.mean(self.detections),
            "avg_confidence": np.mean(self.confidences) if self.confidences else 0,
            "min_confidence": np.min(self.confidences) if self.confidences else 0,
            "max_confidence": np.max(self.confidences) if self.confidences else 0,
        }
    
    def print_summary(self):
        """Imprime resumen formateado."""
        summary = self.get_summary()
        
        print(f"\n{'='*70}")
        print(f"📊 Resultados: {summary['model']}")
        print(f"{'='*70}")
        
        if summary.get('status') == 'No data':
            print("   ⚠️  No hay datos para mostrar")
            return
        
        print(f"   Imágenes procesadas: {summary['total_images']}")
        print(f"   Errores: {summary['errors']}")
        print(f"\n   🕐 Tiempo de Inferencia:")
        print(f"      Promedio: {summary['avg_inference_time']*1000:.2f} ms")
        print(f"      Mínimo:   {summary['min_inference_time']*1000:.2f} ms")
        print(f"      Máximo:   {summary['max_inference_time']*1000:.2f} ms")
        print(f"      Std Dev:  {summary['std_inference_time']*1000:.2f} ms")
        print(f"\n   🎯 Detecciones:")
        print(f"      Total: {summary['total_detections']}")
        print(f"      Promedio por imagen: {summary['avg_detections_per_image']:.2f}")
        print(f"\n   📈 Confianza:")
        print(f"      Promedio: {summary['avg_confidence']:.2%}")
        print(f"      Mínima:   {summary['min_confidence']:.2%}")
        print(f"      Máxima:   {summary['max_confidence']:.2%}")
        print(f"{'='*70}\n")


def benchmark_detector(detector_func, images_dir: str, model_name: str) -> BenchmarkResults:
    """
    Ejecuta benchmark de un detector sobre un directorio de imágenes.
    
    Args:
        detector_func: función de detección (detect_yolo o detect_plate_rtdetr)
        images_dir: directorio con imágenes de prueba
        model_name: nombre del modelo para identificación
        
    Returns:
        BenchmarkResults con estadísticas
    """
    results = BenchmarkResults(model_name)
    
    # Buscar imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(images_dir).glob(f'*{ext}'))
        image_paths.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"⚠️  No se encontraron imágenes en: {images_dir}")
        return results
    
    print(f"\n🔄 Procesando {len(image_paths)} imágenes con {model_name}...")
    
    for i, img_path in enumerate(image_paths, 1):
        try:
            # Medir tiempo de inferencia
            start_time = time.time()
            detections = detector_func(str(img_path))
            inference_time = time.time() - start_time
            
            # Extraer confianzas
            confidences = [d['confidence'] for d in detections]
            
            # Registrar resultado
            results.add_result(len(detections), inference_time, confidences)
            
            # Progreso
            if i % 10 == 0 or i == len(image_paths):
                print(f"   Progreso: {i}/{len(image_paths)} imágenes", end='\r')
        
        except Exception as e:
            results.add_error()
            print(f"\n   ⚠️  Error en {img_path.name}: {e}")
    
    print()  # Nueva línea después del progreso
    return results


def compare_models(images_dir: str):
    """
    Compara YOLOv11n vs RT-DETR vs Ensemble en el mismo dataset.
    
    Args:
        images_dir: directorio con imágenes de prueba
    """
    print(f"\n{'#'*70}")
    print(f"🔬 BENCHMARK: Comparación de Detectores de Placas")
    print(f"{'#'*70}")
    print(f"\n📁 Dataset: {images_dir}")
    
    all_results = []
    
    # Test 1: YOLOv11n
    if detect_yolo:
        print("\n" + "─"*70)
        print("🟦 Test 1/3: YOLOv11n")
        print("─"*70)
        yolo_results = benchmark_detector(detect_yolo, images_dir, "YOLOv11n")
        yolo_results.print_summary()
        all_results.append(yolo_results)
    else:
        print("\n⚠️  YOLOv11n no disponible (saltar test)")
    
    # Test 2: RT-DETR
    if detect_plate_rtdetr:
        print("\n" + "─"*70)
        print("🟩 Test 2/3: RT-DETR")
        print("─"*70)
        rtdetr_results = benchmark_detector(detect_plate_rtdetr, images_dir, "RT-DETR")
        rtdetr_results.print_summary()
        all_results.append(rtdetr_results)
    else:
        print("\n⚠️  RT-DETR no disponible (saltar test)")
    
    # Test 3: Ensemble
    if detect_plate_ensemble:
        print("\n" + "─"*70)
        print("🟪 Test 3/3: Ensemble (YOLO + RT-DETR)")
        print("─"*70)
        ensemble_results = benchmark_detector(detect_plate_ensemble, images_dir, "Ensemble")
        ensemble_results.print_summary()
        all_results.append(ensemble_results)
    else:
        print("\n⚠️  Ensemble no disponible (saltar test)")
    
    # Comparación final
    if len(all_results) > 1:
        print_comparison_table(all_results)


def print_comparison_table(results_list: List[BenchmarkResults]):
    """Imprime tabla comparativa de todos los modelos."""
    print(f"\n{'='*70}")
    print(f"📊 TABLA COMPARATIVA")
    print(f"{'='*70}\n")
    
    # Headers
    print(f"{'Métrica':<30} | {'YOLOv11n':>12} | {'RT-DETR':>12} | {'Ensemble':>12}")
    print(f"{'-'*30} | {'-'*12} | {'-'*12} | {'-'*12}")
    
    # Extraer summaries
    summaries = [r.get_summary() for r in results_list]
    
    # Comparar métricas clave
    metrics = [
        ("Tiempo inferencia (ms)", "avg_inference_time", lambda x: f"{x*1000:.2f}"),
        ("Detecciones totales", "total_detections", lambda x: f"{x:.0f}"),
        ("Detecciones/imagen", "avg_detections_per_image", lambda x: f"{x:.2f}"),
        ("Confianza promedio", "avg_confidence", lambda x: f"{x:.2%}"),
        ("Errores", "errors", lambda x: f"{x:.0f}"),
    ]
    
    for metric_name, metric_key, formatter in metrics:
        row = f"{metric_name:<30} |"
        for summary in summaries:
            value = summary.get(metric_key, 0)
            row += f" {formatter(value):>12} |"
        print(row)
    
    print(f"{'='*70}\n")
    
    # Recomendación
    print("💡 RECOMENDACIÓN:")
    
    # Encontrar mejor modelo por velocidad
    fastest = min(summaries, key=lambda s: s.get('avg_inference_time', float('inf')))
    print(f"   🏃 Más rápido: {fastest['model']} ({fastest['avg_inference_time']*1000:.2f} ms)")
    
    # Encontrar mejor modelo por confianza
    most_confident = max(summaries, key=lambda s: s.get('avg_confidence', 0))
    print(f"   🎯 Más confiable: {most_confident['model']} ({most_confident['avg_confidence']:.2%})")
    
    # Encontrar mejor modelo por detecciones
    most_detections = max(summaries, key=lambda s: s.get('total_detections', 0))
    print(f"   📈 Más detecciones: {most_detections['model']} ({most_detections['total_detections']} placas)\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Script de benchmarking.
    
    Uso:
        python plate_detector_benchmark.py <directorio_imagenes>
    
    Ejemplo:
        python plate_detector_benchmark.py ../../test_images
    """
    import sys
    
    if len(sys.argv) < 2:
        print("\n❌ Error: Debes proporcionar un directorio de imágenes")
        print("\nUso:")
        print("   python plate_detector_benchmark.py <directorio_imagenes>")
        print("\nEjemplo:")
        print("   python plate_detector_benchmark.py ../../test_images")
        sys.exit(1)
    
    images_directory = sys.argv[1]
    
    if not os.path.exists(images_directory):
        print(f"\n❌ Error: El directorio no existe: {images_directory}")
        sys.exit(1)
    
    if not os.path.isdir(images_directory):
        print(f"\n❌ Error: La ruta no es un directorio: {images_directory}")
        sys.exit(1)
    
    # Ejecutar comparación
    compare_models(images_directory)
    
    print(f"\n✅ Benchmark completado\n")
