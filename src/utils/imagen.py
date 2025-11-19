"""
Utilidades básicas para manipulación de imágenes.
Incluye normalización, conversión de formatos y validación.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza imagen al rango [0, 255] uint8.

    Args:
        image: Imagen numpy array

    Returns:
        Imagen normalizada uint8
    """
    if image.dtype == np.uint8:
        return image

    # Convertir a float si es necesario
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    # Normalizar al rango [0, 1] si está en otro rango
    if image.max() > 1.0:
        image = image / 255.0

    # Asegurar rango [0, 1]
    image = np.clip(image, 0.0, 1.0)

    # Convertir a uint8
    return (image * 255).astype(np.uint8)


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convierte imagen RGB a BGR (formato OpenCV).

    Args:
        image: Imagen RGB

    Returns:
        Imagen BGR
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def verify_image(image: np.ndarray) -> None:
    """
    Valida que la imagen sea un array numpy válido.

    Args:
        image: Imagen a validar

    Raises:
        ValueError: Si la imagen no es válida
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("La imagen debe ser un array numpy")

    if len(image.shape) not in [2, 3]:
        raise ValueError("La imagen debe tener 2 o 3 dimensiones")

    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        raise ValueError("La imagen debe tener 1 o 3 canales")

    if image.size == 0:
        raise ValueError("La imagen no puede estar vacía")


def get_image_info(image: np.ndarray) -> dict:
    """
    Obtiene información básica de la imagen.

    Args:
        image: Imagen numpy

    Returns:
        Dict con información
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size_mb = image.nbytes / (1024 * 1024)

    return {
        'shape': (height, width, channels),
        'size_mb': size_mb,
        'min_value': float(image.min()),
        'max_value': float(image.max()),
        'dtype': str(image.dtype)
    }


def resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Redimensiona imagen manteniendo aspect ratio.

    Args:
        image: Imagen de entrada
        target_size: (width, height) objetivo
        interpolation: Método de interpolación

    Returns:
        Imagen redimensionada
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def pad_image_to_size(image: np.ndarray, target_size: Tuple[int, int], pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Rellena imagen para alcanzar tamaño objetivo.

    Args:
        image: Imagen de entrada
        target_size: (width, height) objetivo
        pad_color: Color de relleno BGR

    Returns:
        Imagen rellenada
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top

    padded = cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded