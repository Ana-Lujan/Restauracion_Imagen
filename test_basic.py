#!/usr/bin/env python3
"""
Script de prueba básico para verificar que el sistema funciona sin dependencias pesadas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Prueba las importaciones básicas."""
    try:
        print("🔍 Probando importaciones...")

        # Importaciones básicas que no requieren dependencias pesadas
        from src.utils.imagen import normalize_image, verify_image
        print("✅ Utilidades de imagen: OK")

        from src.utils.preprocessing import apply_white_balance
        print("✅ Preprocesamiento: OK")

        from src.utils.postprocessing import apply_sharpening
        print("✅ Postprocesamiento: OK")

        from src.utils.metrics import calculate_psnr
        print("✅ Métricas: OK")

        from src.models import SRCNN
        print("✅ Modelos: OK")

        print("\n🎉 ¡Todas las importaciones funcionan correctamente!")
        return True

    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def test_basic_functionality():
    """Prueba funcionalidad básica sin dependencias externas."""
    try:
        print("\n🔧 Probando funcionalidad básica...")

        import numpy as np

        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Probar normalización
        from src.utils.imagen import normalize_image
        normalized = normalize_image(test_image)
        print("✅ Normalización de imagen: OK")

        # Probar métricas con imagen sintética
        from src.utils.metrics import calculate_psnr
        psnr = calculate_psnr(test_image.astype(np.float64), normalized.astype(np.float64))
        print(f"✅ Cálculo PSNR: {psnr:.2f} dB")

        print("\n🎯 ¡Funcionalidad básica verificada!")
        return True

    except Exception as e:
        print(f"❌ Error en funcionalidad básica: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Probando sistema de restauración y enhancement...\n")

    success = True
    success &= test_imports()
    success &= test_basic_functionality()

    if success:
        print("\n✅ ¡Sistema listo! Ahora instala las dependencias para la interfaz:")
        print("   pip install gradio opencv-python-headless pillow numpy")
        print("   python3 app_gradio.py")
    else:
        print("\n❌ Hay problemas con el sistema. Revisa las dependencias.")