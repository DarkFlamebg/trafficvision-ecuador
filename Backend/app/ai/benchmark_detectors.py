#!/usr/bin/env python3
"""
Script de benchmark para comparar rendimiento de los tres detectores:
- YOLOv11n
- RT-DETR
- EfficientDet-D2 (motor interno: YOLOv8n — misma arquitectura base de detección)

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
from app.ai.plate_detector_vm import detect_plate_vision_mamba
from app.ai.plate_reader import read_plate


class BenchmarkRunner:
    """Ejecutor de benchmarks para detectores de placas."""

    def __init__(self, images_dir: str, ground_truth_file: str = None):
        self.images_dir   = Path(images_dir)
        self.ground_truth = self._load_ground_truth(ground_truth_file)

        self.results = {
            'yolo':         [],
            'rtdetr':       [],
            'vm':           [],
        }

    # ── Ground truth ──────────────────────────────────────────────────────────
    def _load_ground_truth(self, gt_file: str) -> Dict:
        """Carga ground truth desde archivo JSON (formato COCO)."""
        if gt_file is None or not os.path.exists(gt_file):
            print("[WARN] No ground truth provided — solo se medirá velocidad")
            return None
        with open(gt_file, 'r') as f:
            return json.load(f)

    # ── Memoria GPU ───────────────────────────────────────────────────────────
    def _get_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    # ── Benchmark por imagen y detector ───────────────────────────────────────
    def benchmark_detector(self, detector_name: str, detector_fn, image_path: str) -> Dict:
        """
        Ejecuta benchmark en un detector específico.

        Returns:
            {
                'detections':        int,
                'inference_time_ms': float,
                'memory_mb':         float,
                'plates':            list,
            }
        """
        # Warm-up — primera inferencia siempre es más lenta por JIT / carga de pesos
        if not hasattr(self, f'_warmed_up_{detector_name}'):
            try:
                _ = detector_fn(image_path)
            except Exception:
                pass
            setattr(self, f'_warmed_up_{detector_name}', True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        mem_before = self._get_memory_usage()

        start      = time.perf_counter()
        detections = detector_fn(image_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = (time.perf_counter() - start) * 1000   # ms
        mem_used       = max(0, self._get_memory_usage() - mem_before)

        return {
            'detections':        len(detections),
            'inference_time_ms': round(inference_time, 2),
            'memory_mb':         round(mem_used, 2),
            'plates':            detections,
        }

    # ── Imagen única ──────────────────────────────────────────────────────────
    def run_single_image(self, image_path: str, verbose: bool = True) -> Dict:
        """Ejecuta benchmark en una imagen con los tres detectores."""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Imagen: {Path(image_path).name}")
            print(f"{'='*70}\n")

        results = {}

        detectors = [
            ('yolo',         'YOLOv11n       ', detect_yolo),
            ('rtdetr',       'RT-DETR        ', detect_plate_rtdetr),
            ('vm',           'Vision Mamba   ', detect_plate_vision_mamba),
        ]

        for idx, (key, label, fn) in enumerate(detectors, 1):
            if verbose:
                print(f"  [{idx}/3] {label}...", end=' ', flush=True)
            try:
                results[key] = self.benchmark_detector(key, fn, image_path)
                if verbose:
                    print(f"✓  {results[key]['detections']} placas | "
                          f"{results[key]['inference_time_ms']:.1f}ms")
            except Exception as e:
                print(f"✗  ERROR: {e}")
                results[key] = {
                    'detections': 0,
                    'inference_time_ms': 0.0,
                    'memory_mb': 0.0,
                    'plates': [],
                    'error': str(e),
                }

        return results

    # ── Batch ─────────────────────────────────────────────────────────────────
    def run_batch(self, num_images: int = None, progress_callback=None) -> Dict:
        """Ejecuta benchmark en múltiples imágenes."""
        image_files = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.jpeg')) +
            list(self.images_dir.glob('*.png'))
        )

        if num_images:
            image_files = image_files[:num_images]

        verbose = progress_callback is None

        if verbose:
            print(f"\n{'='*70}")
            print(f"BENCHMARK — {len(image_files)} imágenes | "
                  f"device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            print(f"{'='*70}")

        all_results = {'yolo': [], 'rtdetr': [], 'vm': []}

        for i, img_path in enumerate(image_files, 1):
            if verbose:
                print(f"\n[{i}/{len(image_files)}] {img_path.name}")
            results = self.run_single_image(str(img_path), verbose=False)

            for model in all_results:
                all_results[model].append(results[model])
            
            if progress_callback:
                progress_callback(i, len(image_files), img_path.name, results)

            # Resumen compacto por imagen
            if verbose:
                for key, label in [('yolo',         'YOLOv11n      '),
                                    ('rtdetr',       'RT-DETR       '),
                                    ('vm',           'Vision Mamba  ')]:
                    r = results[key]
                    err = f" ⚠ {r['error']}" if 'error' in r else ''
                    print(f"  {label}  {r['detections']} placas | "
                          f"{r['inference_time_ms']:6.1f}ms{err}")

        if verbose:
            self._print_summary(all_results)
        return all_results

    # ── Resumen ───────────────────────────────────────────────────────────────
    def _print_summary(self, results: Dict):
        print(f"\n{'='*70}")
        print("RESUMEN ESTADÍSTICO")
        print(f"{'='*70}\n")

        labels = {
            'yolo':         'YOLOv11n',
            'rtdetr':       'RT-DETR',
            'vm':           'Vision Mamba (Swin+SSM)',
        }

        averages = {}
        for key, label in labels.items():
            entries = [r for r in results[key] if 'error' not in r]
            if not entries:
                print(f"{label}: sin resultados válidos\n")
                averages[key] = float('inf')
                continue

            times      = [r['inference_time_ms'] for r in entries]
            detections = [r['detections']        for r in entries]

            averages[key] = np.mean(times)

            print(f"{label}:")
            print(f"  Tiempo promedio     : {np.mean(times):.2f} ms  (±{np.std(times):.2f})")
            print(f"  Tiempo mínimo       : {np.min(times):.2f} ms")
            print(f"  Tiempo máximo       : {np.max(times):.2f} ms")
            print(f"  FPS promedio        : {1000/np.mean(times):.1f}")
            print(f"  Total detecciones   : {sum(detections)}")
            print(f"  Promedio por imagen : {np.mean(detections):.2f}")
            print()

        # Velocidad relativa tomando YOLOv11n como base
        base = averages.get('yolo', 1.0)
        if base and base != float('inf'):
            print("Velocidad relativa (YOLOv11n = 1.0x):")
            for key, label in labels.items():
                avg = averages.get(key, float('inf'))
                rel = avg / base if avg != float('inf') else float('inf')
                faster = "más lento" if rel > 1 else "más rápido"
                print(f"  {label:<20}: {rel:.2f}x  {faster}")
            print()

    # ── Guardar resultados ────────────────────────────────────────────────────
    def save_results(self, results: Dict, output_file: str):
        """Guarda resultados en JSON (serializable)."""

        def _sanitize(obj):
            """Convierte tipos numpy a Python nativo para JSON."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()
                        if k != 'plates'}          # omitir crops (no serializable)
            if isinstance(obj, list):
                return [_sanitize(i) for i in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        output = {
            'timestamp': datetime.now().isoformat(),
            'device':    'cuda' if torch.cuda.is_available() else 'cpu',
            'results':   _sanitize(results),
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Resultados guardados en: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark de detectores de placas")
    parser.add_argument('--images-dir',    type=str, required=True,
                        help='Directorio con imágenes de prueba')
    parser.add_argument('--num-images',    type=int, default=None,
                        help='Número máximo de imágenes a procesar')
    parser.add_argument('--output',        type=str, default='benchmark_results.json',
                        help='Archivo de salida para resultados')
    parser.add_argument('--ground-truth',  type=str, default=None,
                        help='Archivo JSON con ground truth (formato COCO)')

    args = parser.parse_args()

    if not os.path.exists(args.images_dir):
        print(f"Error: Directorio no encontrado: {args.images_dir}")
        sys.exit(1)

    runner  = BenchmarkRunner(args.images_dir, args.ground_truth)
    results = runner.run_batch(args.num_images)
    runner.save_results(results, args.output)

    print(f"\n{'='*70}")
    print("Benchmark completado ✓")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()