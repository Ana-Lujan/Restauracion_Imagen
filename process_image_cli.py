#!/usr/bin/env python3
"""
Procesador de imágenes por línea de comandos.
Demuestra la funcionalidad del sistema sin interfaz web.
"""

import argparse
import os
import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Procesador de imágenes - Restauración y Enhancement')
    parser.add_argument('input_image', help='Ruta a la imagen de entrada')
    parser.add_argument('--output', '-o', help='Ruta de salida (opcional)')
    parser.add_argument('--type', '-t', choices=['restauracion', 'enhancement'],
                       default='restauracion', help='Tipo de procesamiento')
    parser.add_argument('--method', '-m', choices=['opencv', 'srcnn', 'realesrgan'],
                       default='opencv', help='Método para enhancement')
    parser.add_argument('--scale', '-s', type=int, choices=[2, 4], default=2,
                       help='Factor de escala para enhancement')

    args = parser.parse_args()

    # Verificar que la imagen existe
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"❌ Error: La imagen '{input_path}' no existe.")
        return 1

    # Determinar ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_processed{suffix}"

    print("🎨 Sistema de Restauración y Enhancement")
    print(f"   📥 Input: {input_path}")
    print(f"   📤 Output: {output_path}")
    print(f"   🎯 Tipo: {args.type}")
    if args.type == 'enhancement':
        print(f"   🔧 Método: {args.method}")
        print(f"   📏 Escala: {args.scale}x")
    print()

    try:
        # Importar el pipeline
        from src.pipeline import image_enhancement_pipeline

        print("🚀 Procesando imagen...")

        # Procesar la imagen
        processed_image, report = image_enhancement_pipeline(
            str(input_path),
            enhancement_type=args.type,
            enhancement_method=args.method,
            scale_factor=args.scale
        )

        # Guardar resultado
        import cv2
        cv2.imwrite(str(output_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

        print("✅ ¡Procesamiento completado!")
        print()
        print("📊 Reporte de procesamiento:")
        print(report)
        print()
        print(f"💾 Imagen guardada en: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        return 1

if __name__ == "__main__":
    exit(main())