"""
Funciones de preprocesamiento de imágenes.
Incluye balance de blancos, CLAHE, reducción de artefactos JPEG.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Aplica balance de blancos automático usando el algoritmo de Gray World.

    Args:
        image: Imagen RGB/BGR

    Returns:
        Imagen con balance de blancos corregido
    """
    # Convertir a float32 para cálculos
    img_float = image.astype(np.float32)

    # Calcular promedio de cada canal
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])

    # Calcular promedio global
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    # Corregir cada canal
    img_float[:, :, 0] = img_float[:, :, 0] * (avg_gray / avg_b) if avg_b > 0 else img_float[:, :, 0]
    img_float[:, :, 1] = img_float[:, :, 1] * (avg_gray / avg_g) if avg_g > 0 else img_float[:, :, 1]
    img_float[:, :, 2] = img_float[:, :, 2] * (avg_gray / avg_r) if avg_r > 0 else img_float[:, :, 2]

    # Clip y convertir de vuelta
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Imagen BGR
        clip_limit: Límite de recorte para CLAHE
        tile_grid_size: Tamaño de la cuadrícula de tiles

    Returns:
        Imagen con CLAHE aplicado
    """
    # Convertir a LAB para mejor equalización de color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Aplicar CLAHE al canal L (luminancia)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convertir de vuelta a BGR
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def reduce_jpeg_artifacts(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Reduce artefactos de compresión JPEG usando filtro bilateral.

    Args:
        image: Imagen BGR
        strength: Fuerza de la reducción (0-1)

    Returns:
        Imagen con artefactos reducidos
    """
    if strength <= 0:
        return image

    # Aplicar filtro bilateral para suavizar artefactos
    # Los parámetros se ajustan según la fuerza
    d = max(5, int(15 * strength))
    sigma_color = max(10, int(100 * strength))
    sigma_space = max(10, int(100 * strength))

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast_adaptive(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Mejora el contraste de forma adaptativa.

    Args:
        image: Imagen BGR
        method: Método ('clahe', 'gamma', 'histogram')

    Returns:
        Imagen con contraste mejorado
    """
    if method == 'clahe':
        return apply_clahe(image)
    elif method == 'gamma':
        # Corrección gamma adaptativa
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        gamma = 1.0 / (mean_brightness + 0.1)  # Evitar división por cero
        gamma = np.clip(gamma, 0.5, 2.0)
        return apply_gamma_correction(image, gamma)
    elif method == 'histogram':
        # Equalización de histograma global
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        return image


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Aplica corrección gamma.

    Args:
        image: Imagen BGR
        gamma: Valor gamma

    Returns:
        Imagen corregida
    """
    # Crear tabla de lookup
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    return cv2.LUT(image, table)


def apply_color_correction(image: np.ndarray) -> np.ndarray:
    """
    Corrección automática de color usando balance de blancos avanzado.

    Args:
        image: Imagen BGR

    Returns:
        Imagen con color corregido
    """
    # Método de balance de blancos perfecto
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Calcular promedio de cada canal en LAB
    avg_l = np.mean(result[:, :, 0])
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])

    # Ajustar canales a y b para neutralizar
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.5)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.5)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def apply_hdr_tone_mapping(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Aplica tone mapping simple para simular HDR.

    Args:
        image: Imagen BGR
        intensity: Intensidad del efecto

    Returns:
        Imagen con tone mapping
    """
    # Convertir a float
    img_float = image.astype(np.float32) / 255.0

    # Aplicar Reinhard tone mapping simple
    img_float = img_float / (1.0 + img_float)
    img_float = np.power(img_float, 1.0 / (intensity + 0.1))

    return (img_float * 255).astype(np.uint8)