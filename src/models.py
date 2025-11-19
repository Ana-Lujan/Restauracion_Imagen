"""
Modelos de deep learning para super-resolución.
Implementación de SRCNN y arquitecturas relacionadas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
from typing import Dict, Any, Optional


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network.
    Arquitectura clásica de 3 capas convolucionales.
    """

    def __init__(self, scale_factor: int = 2):
        """
        Inicializa SRCNN.

        Args:
            scale_factor: Factor de escala (2 o 4)
        """
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor

        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        # Inicialización de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialización de pesos usando Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input LR tensor (B, 3, H, W)

        Returns:
            Output HR tensor (B, 3, H*scale, W*scale)
        """
        # Upscaling inicial con bilinear
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Feature extraction
        x = F.relu(self.conv1(x))

        # Non-linear mapping
        x = F.relu(self.conv2(x))

        # Reconstruction
        x = self.conv3(x)

        return x

    def get_num_params(self) -> int:
        """Retorna número de parámetros del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedSRCNN(nn.Module):
    """
    SRCNN mejorado con skip connections y más capas.
    """

    def __init__(self, scale_factor: int = 2, num_features: int = 64):
        super(EnhancedSRCNN, self).__init__()
        self.scale_factor = scale_factor

        # Encoder
        self.conv1 = nn.Conv2d(3, num_features, 9, padding=4)
        self.conv2 = nn.Conv2d(num_features, num_features//2, 1, padding=0)
        self.conv3 = nn.Conv2d(num_features//2, num_features//2, 3, padding=1)

        # Decoder
        self.conv4 = nn.Conv2d(num_features//2, num_features, 3, padding=1)
        self.conv5 = nn.Conv2d(num_features, 3, 5, padding=2)

        # Skip connection
        self.skip_conv = nn.Conv2d(3, 3, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upscaling inicial
        upscaled = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Encoder
        x1 = F.relu(self.conv1(upscaled))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # Decoder
        x4 = F.relu(self.conv4(x3))
        out = self.conv5(x4)

        # Skip connection
        skip = self.skip_conv(upscaled)

        return out + skip

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_name: str, scale_factor: int = 2) -> nn.Module:
    """
    Factory function para crear modelos.

    Args:
        model_name: Nombre del modelo ('srcnn', 'enhanced_srcnn')
        scale_factor: Factor de escala

    Returns:
        Modelo instanciado
    """
    if model_name.lower() == 'srcnn':
        return SRCNN(scale_factor=scale_factor)
    elif model_name.lower() == 'enhanced_srcnn':
        return EnhancedSRCNN(scale_factor=scale_factor)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str):
    """
    Guarda checkpoint del modelo.

    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        loss: Pérdida actual
        filepath: Path donde guardar
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_name': model.__class__.__name__,
        'scale_factor': getattr(model, 'scale_factor', 2)
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint guardado: {filepath}")


def load_model_checkpoint(filepath: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Carga checkpoint del modelo.

    Args:
        filepath: Path del checkpoint
        device: Dispositivo

    Returns:
        Dict con modelo y metadata
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Crear modelo
    model_name = checkpoint.get('model_name', 'SRCNN')
    scale_factor = checkpoint.get('scale_factor', 2)

    if model_name == 'SRCNN':
        model = SRCNN(scale_factor=scale_factor)
    elif model_name == 'EnhancedSRCNN':
        model = EnhancedSRCNN(scale_factor=scale_factor)
    else:
        raise ValueError(f"Modelo desconocido en checkpoint: {model_name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return {
        'model': model,
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'optimizer_state_dict': checkpoint['optimizer_state_dict']
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Obtiene información del modelo.

    Args:
        model: Modelo

    Returns:
        Dict con información
    """
    return {
        'name': model.__class__.__name__,
        'parameters': model.get_num_params(),
        'scale_factor': getattr(model, 'scale_factor', 2),
        'device': next(model.parameters()).device
    }


def export_to_onnx(model: nn.Module, filepath: str, input_size: tuple = (1, 3, 256, 256)):
    """
    Exporta modelo a ONNX para inferencia optimizada.

    Args:
        model: Modelo PyTorch
        filepath: Path de salida
        input_size: Tamaño de input (B, C, H, W)
    """
    model.eval()
    dummy_input = torch.randn(*input_size)

    torch.onnx.export(
        model, dummy_input, filepath,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Modelo exportado a ONNX: {filepath}")