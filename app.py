#!/usr/bin/env python3
"""
Aplicación web Flask para restauración y enhancement de imágenes.
Alternativa ligera a Gradio para evitar problemas de dependencias.
# Force rebuild commit
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import logging
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB para imágenes HD/4K

# Crear directorio para uploads si no existe
os.makedirs('temp_uploads', exist_ok=True)

@app.route('/')
def index():
    """Página principal con interfaz de usuario completa."""
    logger.info("Acceso a página principal")
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 Sistema de Restauración y Enhancement de Imágenes</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .presentation { background: rgba(255, 255, 255, 0.95); color: #333; padding: 25px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
        .main-content { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 20px; }
        .upload-section { text-align: center; margin-bottom: 30px; border: 2px dashed #ddd; border-radius: 10px; padding: 40px; transition: all 0.3s ease; }
        .upload-section:hover { border-color: #667eea; background: #f8f9ff; }
        .file-input { display: none; }
        .upload-button { background: #667eea; color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; transition: all 0.3s ease; margin: 10px; }
        .upload-button:hover { background: #5a6fd8; transform: translateY(-2px); }
        .settings { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .setting-group { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }
        .setting-group h3 { margin-bottom: 15px; color: #333; font-size: 1.1em; }
        select, input[type="number"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; margin-bottom: 10px; }
        .process-button { background: #28a745; color: white; padding: 15px 40px; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; transition: all 0.3s ease; width: 100%; margin-top: 20px; }
        .process-button:hover { background: #218838; transform: translateY(-2px); }
        .process-button:disabled { background: #6c757d; cursor: not-allowed; transform: none; }
        .results { display: none; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; }
        .image-container { text-align: center; }
        .image-container h3 { margin-bottom: 15px; color: #333; }
        .image-preview { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 15px; }
        .report { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; white-space: pre-line; font-family: 'Courier New', monospace; font-size: 14px; max-height: 400px; overflow-y: auto; margin-top: 20px; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; margin: 20px 0; display: none; }
        .footer { text-align: center; color: white; margin-top: 30px; opacity: 0.8; }

        /* Estilos específicos para la sección de presentación */
        .presentation h2 {
            color: #667eea;
            margin-bottom: 35px;
            font-size: 1.8em;
            text-align: center;
        }

        .presentation h3 {
            color: #5a6fd8;
            margin: 40px 0 20px 0;
            font-size: 1.3em;
        }

        .presentation p {
            line-height: 1.8;
            margin-bottom: 25px;
        }

        .presentation ul {
            margin: 20px 0 30px 0;
            padding-left: 30px;
        }

        .presentation li {
            margin-bottom: 15px;
            line-height: 1.6;
        }

        .presentation .academic-info {
            margin-bottom: 30px !important;
        }

        @media (max-width: 768px) { .results { grid-template-columns: 1fr; } .settings { grid-template-columns: 1fr; } .header h1 { font-size: 2em; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Sistema de Restauración y Enhancement de Imágenes</h1>
            <p>Procesamiento avanzado con técnicas de deep learning • IFTS °24 Año 2025</p>
        </div>

        <div class="presentation">
            <h2>📚 Información del Proyecto</h2>
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #667eea;">
                <strong>Profesor:</strong> Matías Barreto<br>
                <strong>Alumno:</strong> Ana Lujan<br>
                <strong>Materia:</strong> Procesamiento de Imagen<br>
                <strong>Institución:</strong> IFTS 24 - Ciencia de Datos e Inteligencia Artificial
            </div>

            <h3>🎯 Trabajo sobre: Restauración y Enhancement</h3>
            <p><em>Ideal si te interesa: Mejorar la calidad visual y ajustar las características de las imágenes.</em></p>

            <h3>💡 Casos de Uso</h3>
            <ul>
                <li><strong>Ajuste inteligente de iluminación y contraste.</strong></li>
                <li><strong>Corrección de color automática.</strong></li>
                <li><strong>Mejora de nitidez adaptativa.</strong></li>
                <li><strong>Reducción de artefactos de compresión.</strong></li>
                <li><strong>HDR: Combinación de múltiples exposiciones.</strong></li>
            </ul>

            <h3>🤖 Modelos Sugeridos</h3>
            <ul>
                <li><strong>Modelos de difusión para image-to-image con prompts descriptivos.</strong></li>
                <li><strong>InstantID o similares para preservar la identidad mientras se mejora la calidad.</strong></li>
                <li><strong>ControlNet con edge detection (detección de bordes) para una mejora guiada.</strong></li>
            </ul>

            <h3>🔬 Conceptos de Procesamiento Digital Aplicados</h3>
            <ul>
                <li><strong>Histogramas y ecualización.</strong></li>
                <li><strong>Transformaciones de intensidad.</strong></li>
                <li><strong>Filtros de realce.</strong></li>
                <li><strong>Operaciones morfológicas.</strong></li>
            </ul>

            <p>Este proyecto demuestra cómo la integración de técnicas clásicas de procesamiento de imágenes con modelos modernos de aprendizaje profundo puede crear soluciones poderosas y accesibles para mejorar la calidad visual de las imágenes.</p>
        </div>

        <div class="main-content">
            <div class="upload-section" id="uploadSection">
                <h2>📤 Subir Imagen</h2>
                <p>Arrastra y suelta una imagen aquí, o haz clic para seleccionar</p>
                <input type="file" id="imageInput" class="file-input" accept="image/*">
                <br>
                <button class="upload-button" onclick="document.getElementById('imageInput').click()">Seleccionar Imagen</button>
                <div id="fileInfo"></div>
            </div>

            <div class="settings">
                <div class="setting-group">
                    <h3>🎯 Tipo de Procesamiento</h3>
                    <select id="enhancementType">
                        <option value="restauracion">Restauración</option>
                        <option value="enhancement">Super-Resolución</option>
                    </select>
                </div>
                <div class="setting-group">
                    <h3>🔧 Método</h3>
                    <select id="enhancementMethod">
                        <option value="opencv">OpenCV (Procesamiento clásico)</option>
                        <option value="beauty_face">Beauty Face Pro (Belleza Facial Profesional)</option>
                        <option value="perfect_enhancement">Perfect Enhancement (Mejora Perfecta Total)</option>
                        <option value="black_white">Black & White (Blanco y Negro Profesional)</option>
                        <option value="vintage_filters">Vintage Filters (Filtros Vintage)</option>
                        <option value="srcnn">SRCNN (Red Neuronal Convolucional)</option>
                        <option value="real-esrgan">Real-ESRGAN x4 (Super-Resolución SOTA)</option>
                        <option value="gfpgan">GFPGAN (Restauración Facial)</option>
                    </select>
                </div>
                <div class="setting-group">
                    <h3>📏 Factor de Escala</h3>
                    <select id="scaleFactor">
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
                <div class="setting-group">
                    <h3>📊 Modo</h3>
                    <select id="processingMode">
                        <option value="single">Procesamiento Individual</option>
                        <option value="batch">Procesamiento por Lotes</option>
                    </select>
                </div>
            </div>

            <button class="process-button" id="processButton" onclick="processImage()" disabled>🚀 Procesar Imagen</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Procesando imagen... Esto puede tomar unos segundos.</p>
            </div>

            <div class="error" id="error"></div>

            <div class="results" id="results">
                <div class="image-container">
                    <h3>📷 Imagen Original</h3>
                    <img id="originalImage" class="image-preview" alt="Imagen original">
                </div>
                <div class="image-container">
                    <h3>✨ Imagen Procesada</h3>
                    <img id="processedImage" class="image-preview" alt="Imagen procesada">
                    <br>
                    <button id="downloadBtn" onclick="downloadImage()" style="display: none; margin-top: 10px;">⬇️ Descargar Imagen</button>
                </div>
            </div>

            <div class="report" id="report" style="display: none;"></div>

            <div class="analytics" id="analytics" style="display: none; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3>📊 Dashboard de Analytics</h3>
                <div id="analyticsContent"></div>
            </div>
        </div>

        <div class="footer">
            <p>Desarrollado con técnicas avanzadas de procesamiento de imágenes • IFT 2025</p>
        </div>
    </div>

    <script>
        let selectedFile = null;

        const uploadSection = document.getElementById('uploadSection');
        const imageInput = document.getElementById('imageInput');
        const fileInfo = document.getElementById('fileInfo');
        const processButton = document.getElementById('processButton');

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            if (!file.type.startsWith('image/')) {
                alert('Por favor selecciona un archivo de imagen válido.');
                return;
            }

            selectedFile = file;
            fileInfo.innerHTML = `<p><strong>Archivo seleccionado:</strong> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</p>`;

            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('originalImage').src = e.target.result;
            };
            reader.readAsDataURL(file);

            processButton.disabled = false;
        }

        function downloadImage() {
            if (currentDownloadUrl) {
                window.open(currentDownloadUrl, '_blank');
            }
        }

        async function loadAnalytics() {
            try {
                const response = await fetch('/api/analytics');
                const data = await response.json();
                const content = document.getElementById('analyticsContent');
                content.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 5px;">
                            <strong>Procesadas</strong><br>${data.total_processed}
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 5px;">
                            <strong>PSNR Avg</strong><br>${data.average_psnr} dB
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 5px;">
                            <strong>SSIM Avg</strong><br>${data.average_ssim}
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 5px;">
                            <strong>Uptime</strong><br>${data.uptime}
                        </div>
                    </div>
                `;
                document.getElementById('analytics').style.display = 'block';
            } catch (err) {
                console.error('Error cargando analytics:', err);
            }
        }

        let currentDownloadUrl = null;

        function formatStructuredReport(report) {
            let html = `<div style="font-family: Arial, sans-serif; line-height: 1.6;">`;

            // Estado y método
            html += `<h4 style="color: ${report.status.includes('✅') ? '#28a745' : report.status.includes('⚠️') ? '#ffc107' : '#dc3545'}; margin-bottom: 15px;">${report.status || 'Estado desconocido'}</h4>`;
            html += `<p><strong>🎯 Método aplicado:</strong> ${report.method || 'Método no especificado'}</p>`;
            html += `<p><strong>🛠️ Tecnología:</strong> ${report.technology || 'Tecnología no especificada'}</p>`;

            // Métricas
            html += `<div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 15px 0;">`;
            html += `<h5 style="margin: 0 0 10px 0; color: #495057;">📊 Métricas de Calidad</h5>`;
            if (report.metrics && report.metrics.psnr !== undefined && report.metrics.ssim !== undefined) {
                html += `<p><strong>PSNR:</strong> ${report.metrics.psnr}</p>`;
                html += `<p><strong>SSIM:</strong> ${report.metrics.ssim}</p>`;
            } else {
                html += `<p><em>Métricas no disponibles</em></p>`;
            }
            html += `</div>`;

            // Explicabilidad
            if (report.explainability) {
                html += `<div style="background: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0;">`;
                html += `<h5 style="margin: 0 0 10px 0; color: #495057;">🔍 Explicación Técnica (XAI)</h5>`;

                // Diagnóstico
                if (report.explainability.diagnosis && Array.isArray(report.explainability.diagnosis) && report.explainability.diagnosis.length > 0) {
                    html += `<p><strong>🔬 Diagnóstico de la imagen:</strong></p>`;
                    html += `<ul>`;
                    report.explainability.diagnosis.forEach(item => {
                        html += `<li>${item}</li>`;
                    });
                    html += `</ul>`;
                }

                // Técnica aplicada
                if (report.explainability.technique_applied) {
                    html += `<p><strong>⚙️ Técnica aplicada:</strong> ${report.explainability.technique_applied}</p>`;
                }

                // Justificación de parámetros
                if (report.explainability.parameter_justification) {
                    html += `<p><strong>📋 Justificación de parámetros:</strong> ${report.explainability.parameter_justification}</p>`;
                }

                // Interpretación de métricas
                if (report.explainability.metrics_interpretation && Array.isArray(report.explainability.metrics_interpretation) && report.explainability.metrics_interpretation.length > 0) {
                    html += `<p><strong>📈 Interpretación de métricas:</strong></p>`;
                    html += `<ul>`;
                    report.explainability.metrics_interpretation.forEach(item => {
                        html += `<li>${item}</li>`;
                    });
                    html += `</ul>`;
                }

                // Detalles técnicos
                if (report.explainability.technical_details && typeof report.explainability.technical_details === 'object') {
                    html += `<p><strong>🔧 Detalles técnicos:</strong></p>`;
                    html += `<ul>`;
                    Object.entries(report.explainability.technical_details).forEach(([key, value]) => {
                        if (key === 'error_explicacion') {
                            html += `<li><strong>Explicación del Error:</strong> ${value}</li>`;
                        } else if (key !== 'error') {
                            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            html += `<li><strong>${label}:</strong> ${value}</li>`;
                        }
                    });
                    html += `</ul>`;
                }

                html += `</div>`;
            }

            // Descarga
            if (report.download_url) {
                html += `<p><strong>⬇️ Descarga:</strong> <a href="${report.download_url}" target="_blank">${report.download_url}</a></p>`;
            }

            html += `</div>`;
            return html;
        }

        async function processImage() {
            if (!selectedFile) {
                alert('Por favor selecciona una imagen primero.');
                return;
            }

            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            const report = document.getElementById('report');

            loading.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
            report.style.display = 'none';
            processButton.disabled = true;

            try {
                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('enhancement_type', document.getElementById('enhancementType').value);
                formData.append('enhancement_method', document.getElementById('enhancementMethod').value);
                formData.append('scale_factor', document.getElementById('scaleFactor').value);

                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    document.getElementById('processedImage').src = data.image;

                    // Mostrar reporte estructurado con explicabilidad
                    const reportDiv = document.getElementById('report');
                    reportDiv.innerHTML = formatStructuredReport(data.report);
                    results.style.display = 'grid';
                    report.style.display = 'block';

                    // Mostrar botón de descarga
                    if (data.download_url) {
                        currentDownloadUrl = data.download_url;
                        document.getElementById('downloadBtn').style.display = 'block';
                    }

                    // Cargar analytics
                    loadAnalytics();
                } else {
                    // Error manejado con estructura XAI
                    const reportDiv = document.getElementById('report');
                    reportDiv.innerHTML = formatStructuredReport(data.report);
                    report.style.display = 'block';
                    // No mostrar imagen procesada, mantener imagen original si existe
                }

            } catch (err) {
                error.textContent = `Error: ${err.message}`;
                error.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                processButton.disabled = false;
            }
        }

        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/health');
                if (!response.ok) {
                    document.getElementById('error').textContent = 'Error: No se puede conectar con el servidor';
                    document.getElementById('error').style.display = 'block';
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Error: Servidor no disponible';
                document.getElementById('error').style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/process', methods=['POST'])
def process():
    """Procesa la imagen subida con máxima robustez y métricas."""
    logger.info("Procesamiento de imagen iniciado")
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Archivo no encontrado'})

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío'})

        result = process_single_image(file, request.form)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error general: {e}")
        return jsonify({'success': False, 'error': 'Error interno del servidor'})


def process_single_image(file, form_data):
    """Procesa una sola imagen con métricas completas."""
    try:
        # Parámetros
        enhancement_type = form_data.get('enhancement_type', 'restauracion')
        enhancement_method = form_data.get('enhancement_method', 'opencv')
        scale_factor = int(form_data.get('scale_factor', 2))

        logger.info(f"Procesando: tipo={enhancement_type}, método={enhancement_method}, escala={scale_factor}")

        # Cargar imagen original con validación robusta
        try:
            image = Image.open(file)
            logger.info(f"Imagen cargada: formato={image.format}, modo={image.mode}, tamaño={image.size}")

            # Convertir a RGB si es necesario
            if image.mode not in ['RGB', 'L', 'P']:
                logger.warning(f"Modo de imagen no estándar: {image.mode}, convirtiendo a RGB")
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Verificar tamaño mínimo
            if image.size[0] < 1 or image.size[1] < 1:
                raise ValueError("Imagen con dimensiones inválidas")

            original_array = np.array(image)
            logger.info(f"Array numpy creado: shape={original_array.shape}, dtype={original_array.dtype}")

        except Exception as load_err:
            logger.error(f"Error cargando imagen: {load_err}")
            raise ValueError(f"No se pudo cargar la imagen: {str(load_err)}")

        # PROCESAMIENTO PRINCIPAL CON EFECTOS VISUALES DRAMÁTICOS Y DIFERENCIADOS
        logger.info(f"=== INICIANDO PROCESAMIENTO: método={enhancement_method}, tipo={enhancement_type} ===")

        # Asegurar que la imagen esté en RGB antes de cualquier procesamiento
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Imagen convertida a RGB para procesamiento")

        # Algoritmos con efectos visuales DRAMÁTICOS y diferenciados
        if enhancement_method == "black_white":
            # Blanco y negro PROFESIONAL con alto contraste
            img_array = np.array(image)
            # Convertir a escala de grises con método profesional
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Aplicar CLAHE para alto contraste
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray.astype(np.uint8))
            # Ecualización adicional para máximo contraste
            enhanced_gray = cv2.equalizeHist(enhanced_gray)
            # Filtro de nitidez extrema
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_gray = cv2.filter2D(enhanced_gray, -1, kernel)
            # Convertir de vuelta a RGB
            processed = Image.fromarray(enhanced_gray).convert('RGB')
            method = "Blanco y Negro Profesional (Alto Contraste + CLAHE)"

        elif enhancement_method == "perfect_enhancement":
            # Mejora PERFECTA con transformación completa
            img_array = np.array(image)
            # CLAHE para contraste dramático
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            # Ajustes de brillo y contraste extremos
            img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=20)
            # Nitidez extrema con Unsharp Mask agresivo
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 3.0)
            img_array = cv2.addWeighted(img_array, 2.5, gaussian, -1.5, 0)
            processed = Image.fromarray(img_array)
            method = "Perfect Enhancement (CLAHE + Alto Contraste + Nitidez Extrema)"

        elif enhancement_method == "beauty_face":
            # Belleza facial con efectos dramáticos
            img_array = np.array(image)
            # CLAHE agresivo
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            # Suavizado bilateral para piel perfecta
            img_array = cv2.bilateralFilter(img_array, 11, 80, 80)
            # Ajuste de saturación para pieles vibrantes
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 1.4
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            processed = Image.fromarray(img_array)
            method = "Beauty Face Pro (CLAHE + Bilateral + Saturación)"

        elif enhancement_method == "vintage_filters":
            # Filtros vintage con efectos retro dramáticos
            img_array = np.array(image)
            # Sepia intenso
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                    [0.349, 0.686, 0.168],
                                    [0.272, 0.534, 0.131]])
            sepia = cv2.transform(img_array, sepia_filter)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            # Contraste vintage extremo
            sepia = cv2.convertScaleAbs(sepia, alpha=1.3, beta=-30)
            # Granulado de película
            noise = np.random.normal(0, 15, sepia.shape).astype(np.uint8)
            sepia = cv2.add(sepia, noise)
            processed = Image.fromarray(sepia)
            method = "Vintage Filters (Sepia + Alto Contraste + Granulado)"

        else:  # opencv (default) - Restauración DRAMÁTICA
            if enhancement_type == "enhancement" and scale_factor > 1:
                # Super-resolución con efectos visuales extremos
                w, h = image.size
                new_w, new_h = w * scale_factor, h * scale_factor
                processed = image.resize((new_w, new_h), Image.LANCZOS)
                # Aplicar mejoras DRAMÁTICAS
                processed = processed.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=10))
                # Ajustes extremos de contraste y brillo
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(2.0)  # Contraste extremo
                enhancer = ImageEnhance.Brightness(processed)
                processed = enhancer.enhance(1.3)  # Brillo alto
                enhancer = ImageEnhance.Sharpness(processed)
                processed = enhancer.enhance(2.5)  # Nitidez máxima
                method = f"Super-Resolución DRAMÁTICA {scale_factor}x (LANCZOS + Efectos Extremos)"
            else:
                # Restauración con efectos VISUALES EXTREMOS
                processed = image.filter(ImageFilter.SHARPEN)
                processed = processed.filter(ImageFilter.UnsharpMask(radius=3, percent=400, threshold=10))
                # Ajustes de contraste y brillo DRAMÁTICOS
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(1.8)  # Contraste muy alto
                enhancer = ImageEnhance.Brightness(processed)
                processed = enhancer.enhance(1.2)  # Brillo aumentado
                enhancer = ImageEnhance.Sharpness(processed)
                processed = enhancer.enhance(2.0)  # Nitidez máxima
                # Filtro adicional para definición extrema
                processed = processed.filter(ImageFilter.EDGE_ENHANCE_MORE)
                method = "Restauración DRAMÁTICA (SHARPEN Extremo + Alto Contraste)"

        logger.info("=== PROCESAMIENTO PRINCIPAL COMPLETADO EXITOSAMENTE ===")

        # Validar imagen procesada
        if processed is None:
            raise ValueError("La imagen procesada es None")

        # Asegurar que la imagen procesada esté en RGB
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
            logger.info("Imagen procesada convertida a RGB")

        # Calcular métricas con manejo robusto de errores
        try:
            processed_array = np.array(processed)
            logger.info(f"Array procesado creado: shape={processed_array.shape}")
        except Exception as arr_err:
            logger.error(f"Error creando array numpy de imagen procesada: {arr_err}")
            raise ValueError(f"No se pudo procesar la imagen para métricas: {str(arr_err)}")

        # Calcular métricas PSNR y SSIM con manejo especial para super-resolución
        psnr = None
        ssim = None

        # Para super-resolución, las imágenes tienen diferentes tamaños
        # Calculamos métricas solo si tienen el mismo tamaño
        if original_array.shape == processed_array.shape:
            try:
                # PSNR usando OpenCV (máxima precisión)
                psnr = cv2.PSNR(original_array, processed_array)
                logger.info(f"PSNR calculado con OpenCV: {psnr}")
            except Exception as e:
                logger.warning(f"Error calculando PSNR con OpenCV: {e}")
                try:
                    # Fallback: scikit-image
                    psnr = peak_signal_noise_ratio(original_array, processed_array, data_range=255)
                    logger.info(f"PSNR calculado con scikit-image: {psnr}")
                except Exception as e2:
                    logger.warning(f"Error calculando PSNR con scikit-image: {e2}")
                    psnr = None

            try:
                # SSIM usando scikit-image
                min_side = min(original_array.shape[:2])
                if min_side >= 7:
                    ssim = structural_similarity(original_array, processed_array, multichannel=True, data_range=255, channel_axis=2)
                    logger.info(f"SSIM calculado: {ssim}")
                else:
                    logger.warning(f"Imagen demasiado pequeña para SSIM: {min_side}x{min_side} < 7x7")
                    ssim = None
            except Exception as e:
                logger.warning(f"Error calculando SSIM: {e}")
                ssim = None
        else:
            logger.info(f"Super-resolución aplicada - métricas no calculables (tamaños diferentes: {original_array.shape} vs {processed_array.shape})")
            psnr = "N/A (Super-resolución)"
            ssim = "N/A (Super-resolución)"

        # Crear directorio temp_uploads si no existe y guardar imagen procesada
        os.makedirs('temp_uploads', exist_ok=True)
        filename = f"processed_{file.filename}"
        processed.save(os.path.join('temp_uploads', filename))

        # Convertir a base64
        buffer = BytesIO()
        processed.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Preparar métricas con valores seguros
        metrics = {}
        if isinstance(psnr, (int, float)):
            metrics['psnr'] = f'{psnr:.2f} dB'
        else:
            metrics['psnr'] = str(psnr)

        if isinstance(ssim, (int, float)):
            metrics['ssim'] = f'{ssim:.4f}'
        else:
            metrics['ssim'] = str(ssim)

        # Reporte simple sin explicabilidad compleja
        explainability = {
            'diagnosis': ['Procesamiento completado exitosamente'],
            'technique_applied': method,
            'parameter_justification': f'Aplicado {method} según configuración seleccionada',
            'metrics_interpretation': ['Métricas calculadas automáticamente'],
            'technical_details': {'method': method, 'scale': scale_factor}
        }

        report = {
            'status': '✅ Procesamiento Exitoso',
            'method': method,
            'metrics': metrics,
            'technology': 'Pillow + NumPy',
            'explainability': explainability
        }

        return {
            'success': True,
            'image': f'data:image/png;base64,{img_b64}',
            'report': report,
            'metrics': {'psnr': psnr, 'ssim': ssim},
            'download_url': f'/download/{filename}'
        }

    except Exception as proc_err:
        # DEBUGGING CRÍTICO: Mostrar stack trace completo
        import traceback
        print("\n--- ERROR CRÍTICO DE PROCESAMIENTO FORZADO ---")
        print(f"Error específico: {proc_err}")
        print("Stack trace completo:")
        traceback.print_exc()
        print("--- FIN DEL STACK TRACE ---\n")

        logger.error(f"Error procesamiento: {proc_err}")
        # Fallback con estructura consistente
        try:
            image.seek(0)
            orig_image = Image.open(file)
            if orig_image.mode != 'RGB':
                orig_image = orig_image.convert('RGB')

            buffer = BytesIO()
            orig_image.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Estructura consistente para frontend
            return {
                'success': True,
                'image': f'data:image/png;base64,{img_b64}',
                'report': {
                    'status': '⚠️ Procesamiento básico (imagen original)',
                    'method': 'Fallback - Imagen original',
                    'metrics': {'psnr': 'N/A', 'ssim': 'N/A'},
                    'technology': 'Pillow',
                    'explainability': {
                        'diagnosis': ['Error en procesamiento avanzado'],
                        'technique_applied': 'Se devolvió la imagen original sin modificaciones',
                        'parameter_justification': 'Fallback automático por error en algoritmo principal',
                        'metrics_interpretation': ['Métricas no disponibles en modo fallback'],
                        'technical_details': {
                            'error': str(proc_err),
                            'error_explicacion': 'Ocurrió un problema técnico durante el procesamiento. Se muestra la imagen original como resultado seguro.'
                        }
                    }
                },
                'metrics': {'psnr': 0, 'ssim': 0}
            }
        except Exception as fallback_err:
            logger.error(f"Error fallback: {fallback_err}")
            # Estructura de error consistente
            return {
                'success': False,
                'error': 'Error procesando imagen',
                'report': {
                    'status': '❌ Error crítico',
                    'method': 'Error',
                    'metrics': {'psnr': 'N/A', 'ssim': 'N/A'},
                    'technology': 'Error',
                    'explainability': {
                        'diagnosis': ['Error irrecuperable en procesamiento'],
                        'technique_applied': 'No se pudo procesar la imagen',
                        'parameter_justification': 'Error en carga o procesamiento de imagen',
                        'metrics_interpretation': ['Sin métricas disponibles'],
                        'technical_details': {
                            'error': str(fallback_err),
                            'error_explicacion': 'Error crítico al procesar la imagen. No se pudo ni siquiera devolver la imagen original.'
                        }
                    }
                }
            }

@app.route('/health')
def health():
    """Endpoint de salud para verificar que la app funciona."""
    logger.info("Health check solicitado")
    return jsonify({'status': 'healthy', 'message': 'Sistema de restauración y enhancement operativo'})


# Para compatibilidad con gunicorn en HF Spaces
application = app

if __name__ == '__main__':
    try:
        # Detección inteligente de entorno: HF Spaces vs desarrollo local
        is_hf_spaces = 'HF_SPACE_ID' in os.environ or 'SPACE_ID' in os.environ

        if is_hf_spaces:
            port = int(os.environ.get('PORT', 7860))
            host = '0.0.0.0'
            debug_mode = False
            logger.info("Ejecutándose en HF Spaces")
        else:
            port = 5000  # Puerto fijo para desarrollo local
            host = '127.0.0.1'
            debug_mode = True
            logger.info("Ejecutándose en modo desarrollo local")
            print(f"🌐 Accede en: http://{host}:{port}")

        logger.info(f"Iniciando aplicación web en {host}:{port}")

        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}")
        raise