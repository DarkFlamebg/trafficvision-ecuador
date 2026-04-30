# app/ai/plate_detector_config.py
# Configuración centralizada para seleccionar y gestionar detectores de placas

import os
from enum import Enum
from typing import Callable, Dict, Any

# ── Tipos de Detectores ────────────────────────────────────────────────────────
class DetectorType(str, Enum):
    """Tipos de detectores disponibles."""
    YOLO = "yolo"
    RTDETR = "rtdetr"
    ENSEMBLE = "ensemble"


# ── Configuración de Modelos ───────────────────────────────────────────────────
class DetectorConfig:
    """Configuración de parámetros para cada detector."""
    
    # Rutas de modelos
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
    MODELS_DIR = os.path.join(ROOT_DIR, "ml", "models", "trained")
    
    # YOLOv11n
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n_combined_all", "best.pt")
    YOLO_CONFIDENCE_THRESHOLD = 0.45
    
    # RT-DETR
    RTDETR_MODEL_PATH = os.path.join(MODELS_DIR, "rtdetr_combined_all", "best.pt")
    RTDETR_CONFIDENCE_THRESHOLD = 0.45
    
    # Ensemble
    ENSEMBLE_NMS_THRESHOLD = 0.5  # IoU threshold para NMS
    
    # Filtros comunes
    ASPECT_RATIO_MIN = 1.5
    ASPECT_RATIO_MAX = 6.0
    
    # Configuración de uso (puede modificarse en runtime)
    DEFAULT_DETECTOR = DetectorType.ENSEMBLE
    
    @classmethod
    def get_model_path(cls, detector_type: DetectorType) -> str:
        """Retorna la ruta del modelo según el tipo de detector."""
        if detector_type == DetectorType.YOLO:
            return cls.YOLO_MODEL_PATH
        elif detector_type == DetectorType.RTDETR:
            return cls.RTDETR_MODEL_PATH
        else:
            return None  # Ensemble no tiene ruta única
    
    @classmethod
    def get_confidence_threshold(cls, detector_type: DetectorType) -> float:
        """Retorna el umbral de confianza según el tipo de detector."""
        if detector_type == DetectorType.YOLO:
            return cls.YOLO_CONFIDENCE_THRESHOLD
        elif detector_type == DetectorType.RTDETR:
            return cls.RTDETR_CONFIDENCE_THRESHOLD
        else:
            # Ensemble usa el mínimo de ambos
            return min(cls.YOLO_CONFIDENCE_THRESHOLD, cls.RTDETR_CONFIDENCE_THRESHOLD)
    
    @classmethod
    def is_model_available(cls, detector_type: DetectorType) -> bool:
        """Verifica si el modelo está disponible en el sistema."""
        model_path = cls.get_model_path(detector_type)
        
        if detector_type == DetectorType.ENSEMBLE:
            # Ensemble requiere ambos modelos
            return (os.path.exists(cls.YOLO_MODEL_PATH) and 
                    os.path.exists(cls.RTDETR_MODEL_PATH))
        
        return model_path and os.path.exists(model_path)
    
    @classmethod
    def list_available_detectors(cls) -> list:
        """Lista todos los detectores disponibles en el sistema."""
        available = []
        for detector in DetectorType:
            if cls.is_model_available(detector):
                available.append(detector.value)
        return available


# ── Factory de Detectores ──────────────────────────────────────────────────────
class DetectorFactory:
    """Factory para obtener funciones de detección según configuración."""
    
    _detectors_cache: Dict[str, Callable] = {}
    
    @classmethod
    def get_detector(cls, detector_type: DetectorType = None) -> Callable:
        """
        Retorna la función de detección según el tipo especificado.
        
        Args:
            detector_type: Tipo de detector (YOLO, RTDETR, ENSEMBLE).
                          Si es None, usa DEFAULT_DETECTOR de la config.
        
        Returns:
            Función de detección con firma: detect(input_image) -> list
        
        Raises:
            ValueError: Si el detector no está disponible
            ImportError: Si no se puede importar el módulo
        """
        # Usar detector por defecto si no se especifica
        if detector_type is None:
            detector_type = DetectorConfig.DEFAULT_DETECTOR
        
        # Validar disponibilidad
        if not DetectorConfig.is_model_available(detector_type):
            available = DetectorConfig.list_available_detectors()
            raise ValueError(
                f"Detector '{detector_type.value}' no disponible. "
                f"Detectores disponibles: {available}"
            )
        
        # Retornar desde cache si ya está cargado
        cache_key = detector_type.value
        if cache_key in cls._detectors_cache:
            return cls._detectors_cache[cache_key]
        
        # Importar y cachear el detector correspondiente
        try:
            if detector_type == DetectorType.YOLO:
                from plate_detector import detect_plate
                cls._detectors_cache[cache_key] = detect_plate
                return detect_plate
            
            elif detector_type == DetectorType.RTDETR:
                from plate_detector_rtdetr import detect_plate_rtdetr
                cls._detectors_cache[cache_key] = detect_plate_rtdetr
                return detect_plate_rtdetr
            
            elif detector_type == DetectorType.ENSEMBLE:
                from plate_detector_rtdetr import detect_plate_ensemble
                cls._detectors_cache[cache_key] = detect_plate_ensemble
                return detect_plate_ensemble
            
        except ImportError as e:
            raise ImportError(
                f"No se pudo importar el detector '{detector_type.value}': {e}"
            )
    
    @classmethod
    def detect_with_fallback(cls, input_image, preferred_detector: DetectorType = None) -> list:
        """
        Detecta placas con fallback automático si el detector preferido falla.
        
        Orden de fallback:
        1. Detector preferido
        2. Ensemble (si está disponible)
        3. RT-DETR (si está disponible)
        4. YOLO (si está disponible)
        
        Args:
            input_image: ruta o array NumPy
            preferred_detector: detector a intentar primero
        
        Returns:
            Lista de detecciones o lista vacía si todos fallan
        """
        # Orden de intentos
        fallback_order = []
        
        # 1. Agregar detector preferido si se especificó
        if preferred_detector:
            fallback_order.append(preferred_detector)
        
        # 2. Agregar detectores en orden de preferencia
        for detector in [DetectorType.ENSEMBLE, DetectorType.RTDETR, DetectorType.YOLO]:
            if detector not in fallback_order:
                fallback_order.append(detector)
        
        # Intentar cada detector en orden
        last_error = None
        for detector_type in fallback_order:
            if not DetectorConfig.is_model_available(detector_type):
                continue
            
            try:
                detector_func = cls.get_detector(detector_type)
                results = detector_func(input_image)
                
                # Si obtuvimos resultados, retornar
                if results:
                    print(f"✅ Detección exitosa con: {detector_type.value}")
                    return results
                
            except Exception as e:
                last_error = e
                print(f"⚠️  Error con {detector_type.value}: {e}")
                continue
        
        # Si llegamos aquí, todos los detectores fallaron
        if last_error:
            print(f"❌ Todos los detectores fallaron. Último error: {last_error}")
        
        return []
    
    @classmethod
    def clear_cache(cls):
        """Limpia el cache de detectores cargados."""
        cls._detectors_cache.clear()


# ── Funciones de conveniencia ──────────────────────────────────────────────────
def detect_plate_auto(input_image, detector: str = None) -> list:
    """
    Función de conveniencia para detección automática.
    
    Args:
        input_image: ruta o array NumPy
        detector: "yolo", "rtdetr", "ensemble", o None para usar default
    
    Returns:
        Lista de detecciones
    
    Example:
        >>> from app.ai.plate_detector_config import detect_plate_auto
        >>> 
        >>> # Usar detector por defecto (ensemble)
        >>> plates = detect_plate_auto("image.jpg")
        >>> 
        >>> # Forzar uso de YOLO
        >>> plates = detect_plate_auto("image.jpg", detector="yolo")
    """
    if detector:
        detector_type = DetectorType(detector.lower())
    else:
        detector_type = None
    
    detector_func = DetectorFactory.get_detector(detector_type)
    return detector_func(input_image)


def get_best_available_detector() -> DetectorType:
    """
    Retorna el mejor detector disponible en el sistema.
    
    Prioridad:
    1. Ensemble (mayor precisión)
    2. RT-DETR (balance precisión/velocidad)
    3. YOLO (fallback)
    
    Returns:
        DetectorType del mejor detector disponible
    
    Raises:
        RuntimeError: Si no hay ningún detector disponible
    """
    for detector in [DetectorType.ENSEMBLE, DetectorType.RTDETR, DetectorType.YOLO]:
        if DetectorConfig.is_model_available(detector):
            return detector
    
    raise RuntimeError("No hay detectores de placas disponibles en el sistema")


# ── Información del Sistema ────────────────────────────────────────────────────
def print_system_info():
    """Imprime información sobre los detectores disponibles."""
    print("\n" + "="*70)
    print("🔍 SISTEMA DE DETECCIÓN DE PLACAS")
    print("="*70 + "\n")
    
    print("📋 Detectores Disponibles:")
    for detector in DetectorType:
        is_available = DetectorConfig.is_model_available(detector)
        status = "✅" if is_available else "❌"
        
        print(f"   {status} {detector.value.upper()}")
        
        if is_available and detector != DetectorType.ENSEMBLE:
            model_path = DetectorConfig.get_model_path(detector)
            confidence = DetectorConfig.get_confidence_threshold(detector)
            print(f"      Modelo: {model_path}")
            print(f"      Umbral: {confidence}")
    
    print(f"\n🎯 Detector por Defecto: {DetectorConfig.DEFAULT_DETECTOR.value}")
    
    try:
        best = get_best_available_detector()
        print(f"🏆 Mejor Detector Disponible: {best.value}")
    except RuntimeError as e:
        print(f"⚠️  {e}")
    
    print("\n" + "="*70 + "\n")


# ── Main para Testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Script para verificar configuración del sistema.
    
    Uso:
        python plate_detector_config.py
    """
    import sys
    
    # Mostrar información del sistema
    print_system_info()
    
    # Test básico si se proporciona imagen
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Imagen no encontrada: {image_path}")
            sys.exit(1)
        
        print(f"🧪 Probando detección en: {image_path}\n")
        
        # Probar con fallback automático
        plates = DetectorFactory.detect_with_fallback(image_path)
        
        print(f"\n📊 Resultados:")
        print(f"   Placas detectadas: {len(plates)}")
        
        for i, plate in enumerate(plates, 1):
            detector_used = plate.get('detector', 'unknown')
            print(f"\n   Placa #{i}:")
            print(f"      Detector: {detector_used}")
            print(f"      Confianza: {plate['confidence']:.2%}")
            print(f"      Bbox: {plate['bbox']}")
    
    else:
        print("💡 Tip: Ejecuta con una imagen para probar:")
        print("   python plate_detector_config.py imagen.jpg")
