"""
Aplicación web Gradio para restauración y enhancement de imágenes.
Interfaz completa con side-by-side, sliders y caching.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Imports del proyecto
from src.pipeline import process_image_for_gradio
from src.utils.imagen import verify_image, get_image_info


def process_image(
    image,
    enhancement_type,
    enhancement_method,
    scale_factor,
    sharpen_amount,
    denoise_level,
    progress=gr.Progress()
):
    """
    Función principal para procesar imágenes con Gradio.

    Args:
        image: Imagen de entrada (numpy array)
        enhancement_type: "restauracion" o "enhancement"
        enhancement_method: Método para enhancement
        scale_factor: Factor de escala
        sharpen_amount: Nivel de nitidez
        denoise_level: Nivel de denoising
        progress: Barra de progreso de Gradio

    Returns:
        Tuple: (imagen_original, imagen_procesada, reporte)
    """
    try:
        if image is None:
            return None, None, "❌ Error: No se cargó ninguna imagen. Por favor, sube una imagen válida."

        progress(0.1, desc="Validando imagen...")

        # Validar imagen
        verify_image(image)

        # Información de la imagen
        info = get_image_info(image)
        progress(0.3, desc="Procesando imagen...")

        # Procesar imagen
        processed_rgb, report = process_image_for_gradio(
            image,
            enhancement_type=enhancement_type,
            enhancement_method=enhancement_method,
            scale_factor=scale_factor,
            denoise=denoise_level,
            sharpness=sharpen_amount,
            compression_reduction=0.5,
            edge_enhancement=0.2,
            hdr_intensity=0.5
        )

        progress(0.9, desc="Finalizando...")

        # Agregar información de la imagen al reporte
        full_report = f"""📊 Información de la imagen:
• Dimensiones: {info['shape'][0]}×{info['shape'][1]} píxeles
• Canales: {info['shape'][2]}
• Tamaño: {info['size_mb']:.1f} MB
• Rango: [{info['min_value']}, {info['max_value']}]

{report}"""

        progress(1.0, desc="¡Completado!")

        return image, processed_rgb, full_report

    except Exception as e:
        error_msg = f"❌ Error durante el procesamiento: {str(e)}"
        print(f"Error detallado: {e}")
        return None, None, error_msg


def create_demo():
    """
    Crea la aplicación Gradio completa.

    Returns:
        gr.Blocks: Aplicación Gradio
    """
    # Tema moderno
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    )

    with gr.Blocks(
        title="🎨 Sistema de Restauración y Enhancement de Imágenes",
        theme=theme,
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .title {
            text-align: center;
            color: #2563eb;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1em;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1.2em;
            margin-bottom: 2em;
        }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div class="title">🎨 Sistema de Restauración y Enhancement de Imágenes</div>
        <div class="subtitle">
            Desarrollado con IA • Optimizado para CPU • Compatible con Hugging Face Spaces<br>
            <strong>IFT 2025</strong> | Docente: Matías Barreto | Alumna: Ana Lujan
        </div>
        """)

        with gr.Row():
            # Panel izquierdo - Controles
            with gr.Column(scale=1):

                # Input de imagen
                input_image = gr.Image(
                    label="📤 Subir Imagen",
                    type="numpy",
                    height=300,
                    elem_classes="input-image"
                )

                gr.Markdown("*Formatos soportados: JPG, PNG, BMP*")

                # Tipo de procesamiento
                enhancement_type = gr.Dropdown(
                    choices=["restauracion", "enhancement"],
                    value="restauracion",
                    label="🎯 Tipo de Procesamiento",
                    info="Restauración: remueve ruido, mejora nitidez. Enhancement: super-resolución."
                )

                # Método de enhancement (solo visible cuando es enhancement)
                enhancement_method = gr.Dropdown(
                    choices=["opencv", "srcnn", "realesrgan"],
                    value="opencv",
                    label="🔧 Método de Enhancement",
                    info="OpenCV: rápido, SRCNN: modelo entrenado, Real-ESRGAN: alta calidad",
                    visible=False
                )

                # Factor de escala (solo visible para enhancement)
                scale_factor = gr.Dropdown(
                    choices=[2, 4],
                    value=2,
                    label="📏 Factor de Escala",
                    info="2x: duplicar resolución, 4x: cuadruplicar resolución",
                    visible=False
                )

                # Parámetros avanzados (acordeón)
                with gr.Accordion("⚙️ Parámetros Avanzados", open=False):
                    sharpen_amount = gr.Slider(
                        0, 2, value=0.5, step=0.1,
                        label="✨ Nitidez",
                        info="Mayor valor = más nitidez (puede crear halos)"
                    )

                    denoise_level = gr.Slider(
                        0, 1, value=0.3, step=0.1,
                        label="🧹 Reducción de Ruido",
                        info="Mayor valor = menos ruido (puede suavizar detalles)"
                    )

                # Botón de procesamiento
                process_btn = gr.Button(
                    "🚀 Procesar Imagen",
                    variant="primary",
                    size="lg"
                )

                # Información del sistema
                gr.Markdown("### 💡 Información del Sistema")
                system_info = gr.Textbox(
                    label="Estado",
                    value="✅ Sistema listo para procesar imágenes",
                    interactive=False,
                    lines=2
                )

            # Panel derecho - Resultados
            with gr.Column(scale=1):

                # Comparación side-by-side
                gr.Markdown("### 🔍 Comparación Antes/Después")

                with gr.Row():
                    original_display = gr.Image(
                        label="📷 Original",
                        height=250,
                        interactive=False
                    )

                    processed_display = gr.Image(
                        label="✨ Procesada",
                        height=250,
                        interactive=False
                    )

                # Reporte detallado
                with gr.Accordion("📊 Reporte de Procesamiento", open=True):
                    report_text = gr.Textbox(
                        label="Detalles Técnicos",
                        lines=8,
                        interactive=False,
                        show_copy_button=True
                    )

                # Descarga
                download_btn = gr.DownloadButton(
                    label="📥 Descargar Imagen Procesada",
                    variant="secondary",
                    size="sm"
                )

        # Ejemplos
        gr.Examples(
            examples=[
                ["manzana.jpg"],
            ],
            inputs=input_image,
            label="📖 Ejemplo de Uso",
            examples_per_page=1
        )

        # Footer
        gr.Markdown("""
        ---
        ### 🧠 Sobre el Sistema
        - **Modelo**: SRCNN personalizado entrenado en dataset sintético
        - **Arquitectura**: 3 capas convolucionales optimizadas para CPU
        - **Métricas**: PSNR y SSIM en tiempo real
        - **Compatibilidad**: Funciona en CPU, no requiere GPU

        ### 📚 Enlaces
        - [Código Fuente](https://github.com/)
        - [Dataset](https://huggingface.co/datasets/AnaLujan/restauracion-superres)
        - [Modelo](https://huggingface.co/models/)
        """)

        # === EVENTOS ===

        # Mostrar/ocultar controles según tipo de procesamiento
        def update_controls(enh_type):
            if enh_type == "enhancement":
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)

        enhancement_type.change(
            update_controls,
            inputs=[enhancement_type],
            outputs=[enhancement_method, scale_factor]
        )

        # Procesamiento principal
        process_btn.click(
            process_image,
            inputs=[
                input_image,
                enhancement_type,
                enhancement_method,
                scale_factor,
                sharpen_amount,
                denoise_level
            ],
            outputs=[
                original_display,
                processed_display,
                report_text
            ]
        )

        # Conectar descarga
        def get_download_file(processed_img):
            if processed_img is not None:
                # Crear archivo temporal para descarga
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_path = temp_file.name
                temp_file.close()

                # Guardar imagen procesada
                cv2.imwrite(temp_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

                return temp_path
            return None

        download_btn.click(
            get_download_file,
            inputs=[processed_display],
            outputs=[download_btn]
        )

    return demo


if __name__ == "__main__":
    # Crear y lanzar la aplicación
    demo = create_demo()

    print("🚀 Iniciando aplicación Gradio...")
    print("📱 Accede en: http://127.0.0.1:7860")

    # Configuración para HF Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False  # Cambiar a True para compartir públicamente
    )