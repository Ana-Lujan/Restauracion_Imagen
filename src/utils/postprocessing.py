"""
Funciones de postprocesamiento de imágenes.
Incluye nitidez, denoising, ajustes finales de contraste.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Aplica nitidez adaptativa usando unsharp masking.

    Args:
        image: Imagen BGR
        strength: Fuerza de la nitidez (0-1)

    Returns:
        Imagen con nitidez aplicada
    """
    if strength <= 0:
        return image

    # Kernel unsharp mask
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)

    # Aplicar filtro
    sharpened = cv2.filter2D(image, -1, kernel * strength)

    # Mezclar con original para controlar la intensidad
    return cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)


def apply_adaptive_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Aplica nitidez adaptativa basada en el contenido de la imagen.

    Args:
        image: Imagen BGR
        strength: Fuerza base de la nitidez

    Returns:
        Imagen con nitidez adaptativa
    """
    # Convertir a gris para análisis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcular varianza local como medida de detalle
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variance = cv2.Laplacian(blur, cv2.CV_64F).var()

    # Ajustar strength basado en la varianza
    # Imágenes con bajo detalle necesitan más nitidez
    adaptive_strength = strength * (1.0 / (variance / 100.0 + 0.1))
    adaptive_strength = np.clip(adaptive_strength, 0.1, 2.0)

    return apply_sharpening(image, adaptive_strength)


def apply_bilateral_denoise(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Aplica denoising bilateral.

    Args:
        image: Imagen BGR
        strength: Fuerza del denoising (0-1)

    Returns:
        Imagen con ruido reducido
    """
    if strength <= 0:
        return image

    # Parámetros del filtro bilateral
    d = max(5, int(15 * strength))
    sigma_color = max(10, int(75 * strength))
    sigma_space = max(10, int(75 * strength))

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_morphological_operations(image: np.ndarray, operation: str = 'opening', kernel_size: int = 3) -> np.ndarray:
    """
    Aplica operaciones morfológicas.

    Args:
        image: Imagen BGR (se aplica a cada canal)
        operation: Tipo de operación ('opening', 'closing', 'erosion', 'dilation')
        kernel_size: Tamaño del kernel

    Returns:
        Imagen procesada
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'opening':
        morph_func = cv2.morphologyEx
        operation_type = cv2.MORPH_OPEN
    elif operation == 'closing':
        morph_func = cv2.morphologyEx
        operation_type = cv2.MORPH_CLOSE
    elif operation == 'erosion':
        morph_func = cv2.erode
        return morph_func(image, kernel, iterations=1)
    elif operation == 'dilation':
        morph_func = cv2.dilate
        return morph_func(image, kernel, iterations=1)
    else:
        return image

    # Para opening/closing, aplicar a cada canal por separado
    result = image.copy()
    for i in range(3):
        result[:, :, i] = morph_func(image[:, :, i], operation_type, kernel)

    return result


def final_contrast_adjustment(image: np.ndarray, method: str = 'auto') -> np.ndarray:
    """
    Ajuste final de contraste.

    Args:
        image: Imagen BGR
        method: Método ('auto', 'stretch', 'clahe')

    Returns:
        Imagen con contraste ajustado
    """
    if method == 'auto':
        # Auto contraste basado en percentiles
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Calcular percentiles para stretching
        p1, p99 = np.percentile(img_yuv[:, :, 0], (1, 99))
        img_yuv[:, :, 0] = np.clip((img_yuv[:, :, 0] - p1) / (p99 - p1) * 255, 0, 255)

        return cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)

    elif method == 'stretch':
        # Contrast stretching simple
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.normalize(img_yuv[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    else:
        return image


def apply_edge_enhancement(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Mejora bordes usando filtro de realce.

    Args:
        image: Imagen BGR
        strength: Fuerza del realce

    Returns:
        Imagen con bordes realzados
    """
    # Convertir a gris para detectar bordes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar Laplacian para detectar bordes
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Normalizar y aplicar
    laplacian = cv2.convertScaleAbs(laplacian)

    # Mezclar con imagen original
    enhanced = cv2.addWeighted(image, 1.0, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), strength, 0)

    return enhanced


def apply_intensity_transformation(image: np.ndarray, gamma: float = 1.0, contrast: float = 1.0, brightness: int = 0) -> np.ndarray:
    """
    Aplica transformación de intensidad: gamma, contraste, brillo.

    Args:
        image: Imagen BGR
        gamma: Corrección gamma
        contrast: Factor de contraste
        brightness: Ajuste de brillo

    Returns:
        Imagen transformada
    """
    # Convertir a float
    img_float = image.astype(np.float32)

    # Aplicar brillo
    img_float += brightness

    # Aplicar contraste
    img_float = img_float * contrast

    # Aplicar gamma
    if gamma != 1.0:
        img_float = np.power(img_float / 255.0, gamma) * 255.0

    # Clip y convertir
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_compression_artifact_reduction(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Reduce artefactos de compresión avanzada.

    Args:
        image: Imagen BGR
        strength: Fuerza de la reducción

    Returns:
        Imagen con artefactos reducidos
    """
    if strength <= 0:
        return image

    # Combinación de filtros para reducir artefactos
    # 1. Filtro mediano para ruido de sal y pimienta
    median = cv2.medianBlur(image, 3)

    # 2. Bilateral para suavizar bloques
    bilateral = cv2.bilateralFilter(median, 9, 75, 75)

    # 3. Mezclar con original
    return cv2.addWeighted(image, 1 - strength, bilateral, strength, 0)