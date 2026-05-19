#!/usr/bin/env python3
"""
Script de benchmark para comparar rendimiento de los tres detectores:
- YOLOv11n
- RT-DETR
- EfficientDet-D2

Mide precisión, velocidad y uso de memoria.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

# Importar detectores
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.ai.plate_detector import detect_plate as detect_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr
from app.ai.plate_detector_efficientdet import detect_plate_efficientdet
from app.ai.plate_reader import read_plate


class BenchmarkRunner:
    """Ejecutor de benchmarks para detectores de placas."""
    
    def __init__(self, images_dir: str, ground_truth_file: str = None):
        self.images_dir = Path(images_dir)
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        
        # Resultados
        self.results = {
            'yolo': [],
            'rtdetr': [],
            'efficientdet': []
        }
    
    def _load_ground_truth(self, gt_file: str) -> Dict:
        """Carga ground truth desde archivo JSON (formato COCO)."""
        if gt_file is None or not os.path.exists(gt_file):
            print("[WARN] No ground truth provided, solo se medirá velocidad")
            return None
        
        with open(gt_file, 'r') as f:
            return json.load(f)
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria GPU en MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def benchmark_detector(self, detector_name: str, detector_fn, image_path: str) -> Dict:
        """
        Ejecuta benchmark en un detector específico.
        
        Returns:
            {
                'detections': int,
                'inference_time_ms': float,
                'memory_mb': float,
                'plates': list
            }
        """
        # Warm-up (primera inferencia es más lenta)
        if not hasattr(self, f'_warmed_up_{detector_name}'):
            _ = detector_fn(image_path)
            setattr(self, f'_warmed_up_{detector_name}', True)
        
        # Limpiar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        mem_before = self._get_memory_usage()
        
        # Benchmark
        start = time.perf_counter()
        detections = detector_fn(image_path)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start) * 1000  # ms
        
        mem_after = self._get_memory_usage()
        mem_used = mem_after - mem_before
        
        return {
            'detections': len(detections),
            'inference_time_ms': round(inference_time, 2),
            'memory_mb': round(max(0, mem_used), 2),
            'plates': detections
        }
    
    def run_single_image(self, image_path: str, verbose: bool = True):
        """Ejecuta benchmark en una imagen con los tres detectores."""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Imagen: {Path(image_path).name}")
            print(f"{'='*70}\n")
        
        results = {}
        
        # YOLOv11n
        if verbose:
            print("  [1/3] YOLOv11n...", end=' ')
        results['yolo'] = self.benchmark_detector('yolo', detect_yolo, image_path)
        if verbose:
            print(f"✓ {results['yolo']['detections']} placas | "
                  f"{results['yolo']['inference_time_ms']:.1f}ms")
        
        # RT-DETR
        if verbose:
            print("  [2/3] RT-DETR...", end=' ')
        results['rtdetr'] = self.benchmark_detector('rtdetr', detect_plate_rtdetr, image_path)
        if verbose:
            print(f"✓ {results['rtdetr']['detections']} placas | "
                  f"{results['rtdetr']['inference_time_ms']:.1f}ms")
        
        # EfficientDet-D2
        if verbose:
            print("  [3/3] EfficientDet-D2...", end=' ')
        results['efficientdet'] = self.benchmark_detector('efficientdet', detect_plate_efficientdet, image_path)
        if verbose:
            print(f"✓ {results['efficientdet']['detections']} placas | "
                  f"{results['efficientdet']['inference_time_ms']:.1f}ms")
        
        return results
    
    def run_batch(self, num_images: int = None):
        """Ejecuta benchmark en múltiples imágenes."""
        image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                           list(self.images_dir.glob('*.png')))
        
        if num_images:
            image_files = image_files[:num_images]
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {len(image_files)} imágenes")
        print(f"{'='*70}")
        
        all_results = {
            'yolo': [],
            'rtdetr': [],
            'efficientdet': []
        }
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {img_path.name}")
            
            results = self.run_single_image(str(img_path), verbose=False)
            
            for model in ['yolo', 'rtdetr', 'efficientdet']:
                all_results[model].append(results[model])
            
            # Resumen de esta imagen
            print(f"  YOLOv11n:       {results['yolo']['detections']} placas | "
                  f"{results['yolo']['inference_time_ms']:6.1f}ms")
            print(f"  RT-DETR:        {results['rtdetr']['detections']} placas | "
                  f"{results['rtdetr']['inference_time_ms']:6.1f}ms")
            print(f"  EfficientDet:   {results['efficientdet']['detections']} placas | "
                  f"{results['efficientdet']['inference_time_ms']:6.1f}ms")
        
        # Estadísticas agregadas
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Imprime resumen estadístico."""
        print(f"\n{'='*70}")
        print(f"RESUMEN")
        print(f"{'='*70}\n")
        
        for model in ['yolo', 'rtdetr', 'efficientdet']:
            times = [r['inference_time_ms'] for r in results[model]]
            detections = [r['detections'] for r in results[model]]
            
            model_name = {
                'yolo': 'YOLOv11n',
                'rtdetr': 'RT-DETR',
                'efficientdet': 'EfficientDet-D2'
            }[model]
            
            print(f"{model_name}:")
            print(f"  Tiempo promedio:    {np.mean(times):.2f}ms (±{np.std(times):.2f})")
            print(f"  Tiempo mínimo:      {np.min(times):.2f}ms")
            print(f"  Tiempo máximo:      {np.max(times):.2f}ms")
            print(f"  FPS promedio:       {1000/np.mean(times):.1f}")
            print(f"  Detecciones total:  {sum(detections)}")
            print(f"  Promedio por imagen: {np.mean(detections):.2f}")
            print()
        
        # Comparación relativa
        yolo_avg = np.mean([r['inference_time_ms'] for r in results['yolo']])
        rtdetr_avg = np.mean([r['inference_time_ms'] for r in results['rtdetr']])
        eff_avg = np.mean([r['inference_time_ms'] for r in results['efficientdet']])
        
        print("Velocidad relativa (YOLOv11n = 1.0x):")
        print(f"  YOLOv11n:       1.0x")
        print(f"  RT-DETR:        {rtdetr_avg/yolo_avg:.2f}x")
        print(f"  EfficientDet:   {eff_avg/yolo_avg:.2f}x")
        print()
    
    def save_results(self, results: Dict, output_file: str):
        """Guarda resultados en JSON."""
        timestamp = datetime.now().isoformat()
        
        output = {
            'timestamp': timestamp,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Resultados guardados en: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark de detectores de placas")
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directorio con imágenes de prueba')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Número máximo de imágenes a procesar')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Archivo de salida para resultados')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Archivo JSON con ground truth (formato COCO)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Directorio no encontrado: {args.images_dir}")
        sys.exit(1)
    
    # Ejecutar benchmark
    runner = BenchmarkRunner(args.images_dir, args.ground_truth)
    results = runner.run_batch(args.num_images)
    
    # Guardar resultados
    runner.save_results(results, args.output)
    
    print(f"\n{'='*70}")
    print("Benchmark completado ✓")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
