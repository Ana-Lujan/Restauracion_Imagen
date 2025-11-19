#!/usr/bin/env python3
"""
Aplicación web simple con Flask para restauración y enhancement de imágenes.
Alternativa ligera a Gradio para evitar problemas de dependencias.
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import sys

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import process_image_for_gradio

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Crear directorio para uploads temporales
UPLOAD_FOLDER = Path('temp_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/')
def index():
    """Página principal con interfaz de usuario."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Procesa la imagen subida."""
    try:
        # Verificar si hay archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se encontró archivo de imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400

        # Verificar tipo de archivo
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return jsonify({'error': 'Formato no soportado. Use PNG, JPG o BMP'}), 400

        # Obtener parámetros
        enhancement_type = request.form.get('enhancement_type', 'restauracion')
        enhancement_method = request.form.get('enhancement_method', 'opencv')
        scale_factor = int(request.form.get('scale_factor', 2))

        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename, dir=UPLOAD_FOLDER) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Leer imagen
            image = Image.open(temp_path)
            image_array = np.array(image)

            # Procesar imagen
            processed_array, report = process_image_for_gradio(
                image_array,
                enhancement_type=enhancement_type,
                enhancement_method=enhancement_method,
                scale_factor=scale_factor
            )

            # Convertir a PIL Image
            processed_image = Image.fromarray(processed_array)

            # Convertir a base64 para enviar al frontend
            buffered = BytesIO()
            processed_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'report': report
            })

        finally:
            # Limpiar archivo temporal
            os.unlink(temp_path)

    except Exception as e:
        return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500

@app.route('/health')
def health():
    """Endpoint de salud para verificar que la app funciona."""
    return jsonify({'status': 'healthy', 'message': 'Sistema de restauración y enhancement operativo'})

if __name__ == '__main__':
    print("🚀 Iniciando aplicación web simple...")
    print("📱 Accede en: http://127.0.0.1:5000")
    print("💡 Usa Ctrl+C para detener")

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Cambiar a True para desarrollo
        threaded=True
    )