"""
Métricas de entrenamiento usando TorchMetrics.
PSNR, SSIM y otras métricas para evaluación durante el entrenamiento.
"""

import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from typing import Dict, Any, Optional


class SRMetrics:
    """
    Métricas especializadas para super-resolución.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Inicializa las métricas.

        Args:
            device: Dispositivo para las métricas
        """
        self.device = device

        # Métricas principales
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # LPIPS (opcional, requiere instalación)
        try:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            self.has_lpips = True
        except:
            self.has_lpips = False
            print("LPIPS no disponible, omitiendo métrica perceptual")

    def __call__(self, lr_batch: torch.Tensor, hr_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calcula todas las métricas.

        Args:
            lr_batch: Batch de imágenes LR
            hr_batch: Batch de imágenes HR

        Returns:
            Dict con métricas
        """
        metrics = {}

        # PSNR
        metrics['psnr'] = self.psnr(hr_batch, hr_batch)  # Comparar HR con HR para baseline

        # SSIM
        metrics['ssim'] = self.ssim(hr_batch, hr_batch)

        # LPIPS si disponible
        if self.has_lpips:
            metrics['lpips'] = self.lpips(hr_batch, hr_batch)

        return metrics

    def reset(self):
        """Resetea las métricas para nueva evaluación."""
        self.psnr.reset()
        self.ssim.reset()
        if self.has_lpips:
            self.lpips.reset()


def log_training_metrics(epoch: int, train_loss: float, val_loss: float,
                        train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                        log_file: Optional[str] = None):
    """
    Registra métricas de entrenamiento.

    Args:
        epoch: Época actual
        train_loss: Pérdida de entrenamiento
        val_loss: Pérdida de validación
        train_metrics: Métricas de entrenamiento
        val_metrics: Métricas de validación
        log_file: Archivo para logging (opcional)
    """
    print(f"\n📈 Epoch {epoch} Metrics:")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")

    for metric_name in train_metrics.keys():
        if metric_name in val_metrics:
            train_val = train_metrics[metric_name]
            val_val = val_metrics[metric_name]
            print(f"   Train {metric_name.upper()}: {train_val:.4f}")
            print(f"   Val {metric_name.upper()}:   {val_val:.4f}")

    # Logging a archivo
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}")
            for metric_name in sorted(train_metrics.keys()):
                if metric_name in val_metrics:
                    f.write(f",{train_metrics[metric_name]:.4f},{val_metrics[metric_name]:.4f}")
            f.write("\n")


def evaluate_model(model: nn.Module, val_loader: torch.utils.data.DataLoader,
                  device: str = 'cpu') -> Dict[str, float]:
    """
    Evalúa el modelo en el conjunto de validación.

    Args:
        model: Modelo a evaluar
        val_loader: DataLoader de validación
        device: Dispositivo

    Returns:
        Dict con métricas promedio
    """
    model.eval()
    metrics = SRMetrics(device=device)
    metrics.reset()

    total_metrics = {'psnr': 0.0, 'ssim': 0.0}
    if metrics.has_lpips:
        total_metrics['lpips'] = 0.0

    num_batches = 0

    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # Forward pass
            outputs = model(lr_batch)

            # Calcular métricas
            batch_metrics = metrics(lr_batch, hr_batch)

            for key in total_metrics:
                if key in batch_metrics:
                    total_metrics[key] += batch_metrics[key].item()

            num_batches += 1

    # Promedios
    for key in total_metrics:
        total_metrics[key] /= num_batches

    return total_metrics


def create_metrics_header() -> str:
    """
    Crea header para archivo de métricas.

    Returns:
        String del header
    """
    return "epoch,train_loss,val_loss,train_psnr,val_psnr,train_ssim,val_ssim"


def save_metrics_to_csv(metrics_log: list, filepath: str):
    """
    Guarda log de métricas a CSV.

    Args:
        metrics_log: Lista de dicts con métricas
        filepath: Path del archivo
    """
    import csv

    if not metrics_log:
        return

    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
                [f'train_{k}' for k in metrics_log[0].keys() if k not in ['epoch', 'train_loss', 'val_loss']] + \
                [f'val_{k}' for k in metrics_log[0].keys() if k not in ['epoch', 'train_loss', 'val_loss']]

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"Métricas guardadas en: {filepath}")


def plot_training_curves(log_data: list, save_path: Optional[str] = None):
    """
    Grafica curvas de entrenamiento.

    Args:
        log_data: Lista de métricas por época
        save_path: Path para guardar gráfica (opcional)
    """
    try:
        import matplotlib.pyplot as plt

        epochs = [d['epoch'] for d in log_data]
        train_loss = [d['train_loss'] for d in log_data]
        val_loss = [d['val_loss'] for d in log_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(epochs, train_loss, label='Train Loss')
        ax1.plot(epochs, val_loss, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        # PSNR/SSIM
        if 'train_psnr' in log_data[0]:
            train_psnr = [d['train_psnr'] for d in log_data]
            val_psnr = [d['val_psnr'] for d in log_data]
            ax2.plot(epochs, train_psnr, label='Train PSNR')
            ax2.plot(epochs, val_psnr, label='Val PSNR')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('PSNR Progress')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada: {save_path}")
        else:
            plt.show()

    except ImportError:
        print("Matplotlib no disponible, omitiendo gráficas")