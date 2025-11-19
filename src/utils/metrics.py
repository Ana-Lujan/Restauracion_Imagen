"""
Métricas de evaluación para super-resolución.
PSNR, SSIM y otras métricas de calidad de imagen.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcula PSNR (Peak Signal-to-Noise Ratio).

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        PSNR en dB
    """
    # Asegurar mismo tamaño
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    # Convertir a float
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    # Calcular MSE
    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        return float('inf')

    # Calcular PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_ssim(original: np.ndarray, processed: np.ndarray, win_size: int = 11) -> float:
    """
    Calcula SSIM (Structural Similarity Index).

    Args:
        original: Imagen original
        processed: Imagen procesada
        win_size: Tamaño de la ventana

    Returns:
        SSIM (0-1)
    """
    # Asegurar mismo tamaño
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    # Convertir a float
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    # Constantes
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Calcular medias locales
    mu1 = cv2.GaussianBlur(original, (win_size, win_size), 0)
    mu2 = cv2.GaussianBlur(processed, (win_size, win_size), 0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calcular varianzas y covarianza
    sigma1_sq = cv2.GaussianBlur(original ** 2, (win_size, win_size), 0) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(processed ** 2, (win_size, win_size), 0) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * processed, (win_size, win_size), 0) - mu1_mu2

    # Calcular SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcula MSE (Mean Squared Error).

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        MSE
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


def calculate_rmse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcula RMSE (Root Mean Squared Error).

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        RMSE
    """
    return np.sqrt(calculate_mse(original, processed))


def calculate_image_quality_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Calcula múltiples métricas de calidad de imagen.

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        Dict con métricas
    """
    return {
        'psnr': calculate_psnr(original, processed),
        'ssim': calculate_ssim(original, processed),
        'mse': calculate_mse(original, processed),
        'rmse': calculate_rmse(original, processed)
    }


def calculate_histogram_similarity(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcula similitud de histogramas usando correlación.

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        Correlación de histogramas (0-1)
    """
    # Convertir a HSV para mejor comparación de color
    if len(original.shape) == 3:
        original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        processed_hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # Calcular histogramas para canal V (brillo)
        hist_orig = cv2.calcHist([original_hsv], [2], None, [256], [0, 256])
        hist_proc = cv2.calcHist([processed_hsv], [2], None, [256], [0, 256])
    else:
        hist_orig = cv2.calcHist([original], [0], None, [256], [0, 256])
        hist_proc = cv2.calcHist([processed], [0], None, [256], [0, 256])

    # Normalizar histogramas
    hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
    hist_proc = cv2.normalize(hist_proc, hist_proc).flatten()

    # Calcular correlación
    correlation = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)

    # Convertir a escala 0-1 (correlación va de -1 a 1)
    return (correlation + 1) / 2


def calculate_edge_preservation(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calcula preservación de bordes usando gradientes.

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        Métrica de preservación de bordes (0-1)
    """
    # Convertir a gris
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        proc_gray = processed

    # Calcular gradientes
    grad_orig_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_orig_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(grad_orig_x**2 + grad_orig_y**2)

    grad_proc_x = cv2.Sobel(proc_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_proc_y = cv2.Sobel(proc_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_proc = np.sqrt(grad_proc_x**2 + grad_proc_y**2)

    # Calcular similitud de gradientes
    return calculate_ssim(grad_orig.astype(np.uint8), grad_proc.astype(np.uint8))


def get_comprehensive_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas comprehensivas de calidad.

    Args:
        original: Imagen original
        processed: Imagen procesada

    Returns:
        Dict con todas las métricas
    """
    basic_metrics = calculate_image_quality_metrics(original, processed)

    additional_metrics = {
        'histogram_similarity': calculate_histogram_similarity(original, processed),
        'edge_preservation': calculate_edge_preservation(original, processed)
    }

    return {**basic_metrics, **additional_metrics}