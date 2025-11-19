import cv2
import numpy as np
import torch
from pathlib import Path
try:
    # When imported as module
    from .utils.imagen import normalize_image, convert_to_bgr, verify_image
    from .utils.preprocessing import (apply_white_balance, apply_clahe, reduce_jpeg_artifacts,
                                    apply_color_correction, apply_hdr_tone_mapping,
                                    enhance_contrast_adaptive)
    from .utils.postprocessing import (apply_sharpening, apply_bilateral_denoise, final_contrast_adjustment,
                                     apply_adaptive_sharpening, apply_morphological_operations,
                                     apply_edge_enhancement, apply_compression_artifact_reduction,
                                     apply_intensity_transformation)
    from .utils.metrics import calculate_psnr, calculate_ssim, get_comprehensive_metrics
    from .models import load_model_checkpoint, SRCNN
except ImportError:
    # When run as standalone script
    from utils.imagen import normalize_image, convert_to_bgr, verify_image
    from utils.preprocessing import (apply_white_balance, apply_clahe, reduce_jpeg_artifacts,
                                    apply_color_correction, apply_hdr_tone_mapping,
                                    enhance_contrast_adaptive)
    from utils.postprocessing import (apply_sharpening, apply_bilateral_denoise, final_contrast_adjustment,
                                     apply_adaptive_sharpening, apply_morphological_operations,
                                     apply_edge_enhancement, apply_compression_artifact_reduction,
                                     apply_intensity_transformation)
    from utils.metrics import calculate_psnr, calculate_ssim, get_comprehensive_metrics
    from models import load_model_checkpoint, SRCNN

# Modelos cargados de forma lazy
_srcnn_model = None
_realesrgan_model = None

def load_srcnn_model(model_path: str = "model/srcnn_best.pth", device: str = 'cpu'):
    """
    Carga el modelo SRCNN entrenado de forma lazy.

    Args:
        model_path: Path al modelo guardado
        device: Dispositivo

    Returns:
        Modelo cargado
    """
    global _srcnn_model
    if _srcnn_model is not None:
        return _srcnn_model

    try:
        checkpoint = load_model_checkpoint(model_path, device)
        _srcnn_model = checkpoint['model']
        print(f"SRCNN cargado desde {model_path}")
        return _srcnn_model
    except Exception as e:
        print(f"Advertencia: No se pudo cargar SRCNN ({str(e)}). Usando upscaling básico.")
        return None

def load_realesrgan_model():
    """
    Carga el modelo Real-ESRGAN de forma lazy para super resolución.
    """
    global _realesrgan_model
    if _realesrgan_model is not None:
        return _realesrgan_model

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        upsampler = RealESRGANer(
            scale=netscale,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False  # CPU
        )
        _realesrgan_model = upsampler
        return upsampler
    except Exception as e:
        print(f"Advertencia: No se pudo cargar Real-ESRGAN ({str(e)}). Super resolución no disponible.")
        return None

def apply_restoration(image, params=None):
    """
    Aplica restauración avanzada: remueve ruido, mejora nitidez, corrige iluminación.
    Incluye operaciones morfológicas, corrección de color avanzada, y reducción de artefactos.

    Args:
        image: Imagen BGR
        params: Parámetros de control (opcional)

    Returns:
        Imagen restaurada
    """
    if params is None:
        params = {}

    # Corrección de color avanzada
    image = apply_color_correction(image)

    # Balance de blancos
    image = apply_white_balance(image)

    # Reducción de ruido con denoising bilateral
    denoise_strength = params.get('denoise', 0.3)
    image = apply_bilateral_denoise(image, denoise_strength)

    # Operaciones morfológicas para limpieza
    image = apply_morphological_operations(image, 'opening', kernel_size=3)

    # Mejora de contraste adaptativa
    contrast_method = params.get('contrast_method', 'clahe')
    image = enhance_contrast_adaptive(image, contrast_method)

    # Reducción avanzada de artefactos de compresión
    compression_strength = params.get('compression_reduction', 0.5)
    image = apply_compression_artifact_reduction(image, compression_strength)

    # Nitidez adaptativa
    sharpness_strength = params.get('sharpness', 0.5)
    image = apply_adaptive_sharpening(image, sharpness_strength)

    # Realce de bordes
    edge_strength = params.get('edge_enhancement', 0.2)
    image = apply_edge_enhancement(image, edge_strength)

    # Ajuste final de contraste
    image = final_contrast_adjustment(image)

    return image

def apply_enhancement(image, scale_factor=2, method='srcnn', params=None):
    """
    Aplica enhancement avanzado: mejora de colores, resolución y calidad.
    Soporta múltiples métodos incluyendo modelos de deep learning.

    Args:
        image: Imagen BGR
        scale_factor: Factor de escala (2 o 4)
        method: Método ('opencv', 'srcnn', 'realesrgan')
        params: Parámetros adicionales

    Returns:
        Imagen mejorada
    """
    if params is None:
        params = {}

    # Pre-procesamiento
    image = apply_color_correction(image)
    image = apply_white_balance(image)
    image = enhance_contrast_adaptive(image, 'clahe')

    # Super-resolución según método
    if method == 'srcnn':
        image = apply_srcnn_enhancement(image, scale_factor)
    elif method == 'realesrgan':
        image = apply_realesrgan_enhancement(image, scale_factor)
    else:  # opencv
        h, w = image.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Post-procesamiento
    image = apply_adaptive_sharpening(image, params.get('sharpness', 0.3))
    image = apply_hdr_tone_mapping(image, params.get('hdr_intensity', 0.5))

    return image

def apply_srcnn_enhancement(image, scale_factor=2):
    """
    Aplica enhancement usando modelo SRCNN entrenado.

    Args:
        image: Imagen BGR
        scale_factor: Factor de escala

    Returns:
        Imagen mejorada con SRCNN
    """
    model = load_srcnn_model()
    if model is None:
        # Fallback a upscaling básico
        h, w = image.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Preprocesamiento para el modelo
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    device = next(model.parameters()).device
    tensor_image = tensor_image.to(device)

    # Inferencia
    with torch.no_grad():
        output = model(tensor_image)

    # Post-procesamiento
    output = torch.clamp(output, 0, 1)
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

def apply_realesrgan_enhancement(image, scale_factor=4):
    """
    Aplica enhancement usando Real-ESRGAN.

    Args:
        image: Imagen BGR
        scale_factor: Factor de escala (usualmente 4)

    Returns:
        Imagen mejorada con Real-ESRGAN
    """
    upsampler = load_realesrgan_model()
    if upsampler is None:
        # Fallback
        h, w = image.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Real-ESRGAN espera RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar
    output, _ = upsampler.enhance(rgb_image, outscale=scale_factor)

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

def image_enhancement_pipeline(image_path, enhancement_type="restauracion", enhancement_method="opencv",
                              scale_factor=2, **params):
    """
    Pipeline completo de procesamiento de imágenes con parámetros avanzados.

    Args:
        image_path: Path a la imagen
        enhancement_type: "restauracion" o "enhancement"
        enhancement_method: Método para enhancement
        scale_factor: Factor de escala
        **params: Parámetros adicionales

    Returns:
        Tuple: (imagen_procesada, reporte)
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen")

    original = image.copy()
    verify_image(image)

    # Aplicar procesamiento según tipo
    if enhancement_type == "restauracion":
        processed = apply_restoration(image, params)
        method = f"Restauración Avanzada (color correction, denoising, morphological ops, adaptive sharpness)"
    elif enhancement_type == "enhancement":
        processed = apply_enhancement(image, scale_factor, enhancement_method, params)
        method = f"Enhancement Avanzado ({enhancement_method.upper()} x{scale_factor}, HDR, color enhancement)"
    else:
        raise ValueError("Tipo de enhancement no válido")

    # Normalizar resultado
    processed = normalize_image(processed)

    # Calcular métricas comprehensivas
    metrics = get_comprehensive_metrics(original, processed)

    # Reporte detallado
    report = f"""🎨 Procesamiento Completado

📋 Método: {method}
📊 Métricas de Calidad:
• PSNR: {metrics['psnr']:.2f} dB
• SSIM: {metrics['ssim']:.4f}
• MSE: {metrics['mse']:.4f}
• RMSE: {metrics['rmse']:.4f}
• Similitud de Histograma: {metrics['histogram_similarity']:.4f}
• Preservación de Bordes: {metrics['edge_preservation']:.4f}

🔧 Técnicas Aplicadas:
• Corrección de Color Automática
• Balance de Blancos
• Ecualización CLAHE
• Operaciones Morfológicas
• Nitidez Adaptativa
• Reducción de Artefactos
• Tone Mapping HDR
• Mejora de Bordes"""

    return processed, report

def process_image_for_gradio(image, enhancement_type="restauracion", enhancement_method="opencv",
                           scale_factor=2, **params):
    """
    Función especializada para Gradio que procesa imágenes numpy.

    Args:
        image: Imagen numpy array (RGB)
        enhancement_type: Tipo de procesamiento
        enhancement_method: Método
        scale_factor: Factor de escala
        **params: Parámetros adicionales

    Returns:
        Tuple: (imagen_rgb_procesada, reporte)
    """
    # Convertir de RGB (Gradio) a BGR (OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = image

    verify_image(bgr_image)

    # Aplicar procesamiento
    if enhancement_type == "restauracion":
        processed_bgr = apply_restoration(bgr_image, params)
        method_desc = "Restauración Avanzada"
    elif enhancement_type == "enhancement":
        processed_bgr = apply_enhancement(bgr_image, scale_factor, enhancement_method, params)
        method_desc = f"Enhancement {enhancement_method.upper()} x{scale_factor}"
    else:
        raise ValueError("Tipo de enhancement no válido")

    # Convertir de vuelta a RGB para Gradio
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # Calcular métricas
    original_bgr = bgr_image
    metrics = get_comprehensive_metrics(original_bgr, processed_bgr)

    # Reporte para Gradio
    report = f"""✅ Procesamiento Completado

🎯 Método: {method_desc}
📊 Calidad:
• PSNR: {metrics['psnr']:.2f} dB
• SSIM: {metrics['ssim']:.4f}
• Similitud: {metrics['histogram_similarity']:.2f}

🛠️ Características Aplicadas:
• Corrección de Color Inteligente
• Mejora de Iluminación Adaptativa
• Nitidez Adaptativa
• Reducción de Artefactos
• Operaciones Morfológicas
• Tone Mapping HDR"""

    return processed_rgb, report

def enhance_image(image_path, enhancement_type="restauracion", enhancement_method="opencv",
                 scale_factor=2, **params):
    """
    Función de alto nivel para enhancement de imágenes (compatibilidad).

    Args:
        image_path: Path a la imagen
        enhancement_type: Tipo de procesamiento
        enhancement_method: Método
        scale_factor: Factor de escala
        **params: Parámetros adicionales

    Returns:
        Tuple: (imagen_procesada, reporte)
    """
    return image_enhancement_pipeline(image_path, enhancement_type, enhancement_method, scale_factor, **params)