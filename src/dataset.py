"""
Dataset personalizado para super-resolución.
Manejo de pares HR/LR con data augmentation.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List


class SuperResolutionDataset(Dataset):
    """
    Dataset personalizado para super-resolución con pares HR/LR.
    """

    def __init__(self, hr_dir: str, lr_dir: str, scale_factor: int = 2,
                 transform: Optional[transforms.Compose] = None):
        """
        Inicializa el dataset.

        Args:
            hr_dir: Directorio con imágenes HR
            lr_dir: Directorio con imágenes LR
            scale_factor: Factor de escala
            transform: Transformaciones a aplicar
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale_factor = scale_factor

        # Obtener lista de archivos
        self.hr_files = sorted(list(self.hr_dir.glob("*.png"))) + sorted(list(self.hr_dir.glob("*.jpg")))
        self.lr_files = sorted(list(self.lr_dir.glob("*.png"))) + sorted(list(self.lr_dir.glob("*.jpg")))

        # Verificar que coincidan
        assert len(self.hr_files) == len(self.lr_files), "Número diferente de archivos HR/LR"

        # Transformaciones por defecto
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cargar imágenes
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        # Aplicar transformaciones
        if self.transform:
            hr_tensor = self.transform(hr_image)
            lr_tensor = self.transform(lr_image)
        else:
            hr_tensor = transforms.ToTensor()(hr_image)
            lr_tensor = transforms.ToTensor()(lr_image)

        return lr_tensor, hr_tensor


class HFDatasetAdapter(Dataset):
    """
    Adaptador para datasets de Hugging Face.
    """

    def __init__(self, hf_dataset, scale_factor: int = 2,
                 transform: Optional[transforms.Compose] = None):
        """
        Inicializa el adaptador.

        Args:
            hf_dataset: Dataset de HF
            scale_factor: Factor de escala
            transform: Transformaciones
        """
        self.dataset = hf_dataset
        self.scale_factor = scale_factor

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        # Asumir que la imagen es PIL
        image = sample['image']
        label = sample['label']

        # Convertir a tensor
        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        # Si es HR (label 0), crear LR downsampling
        if label == 0:  # HR
            hr_tensor = tensor
            # Downscale para crear LR
            lr_tensor = transforms.functional.resize(
                hr_tensor,
                (hr_tensor.shape[1] // self.scale_factor, hr_tensor.shape[2] // self.scale_factor),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
            return lr_tensor, hr_tensor
        else:  # LR
            lr_tensor = tensor
            # Upscale para crear HR (simulado)
            hr_tensor = transforms.functional.resize(
                lr_tensor,
                (lr_tensor.shape[1] * self.scale_factor, lr_tensor.shape[2] * self.scale_factor),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
            return lr_tensor, hr_tensor


def create_data_transforms(augment: bool = False) -> transforms.Compose:
    """
    Crea transformaciones de datos con opcional data augmentation.

    Args:
        augment: Si aplicar augmentation

    Returns:
        Compose de transformaciones
    """
    transform_list = []

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transforms.Compose(transform_list)


def create_synthetic_dataset(num_samples: int = 100, output_dir: str = 'synthetic_dataset',
                           image_size: Tuple[int, int] = (512, 512), scale_factor: int = 2) -> str:
    """
    Crea un dataset sintético para pruebas.

    Args:
        num_samples: Número de muestras
        output_dir: Directorio de salida
        image_size: Tamaño de imágenes HR
        scale_factor: Factor de escala

    Returns:
        Path al dataset creado
    """
    from generate_dataset import generate_artificial_image, downscale_bicubic

    output_path = Path(output_dir)
    hr_dir = output_path / 'train' / 'HR'
    lr_dir = output_path / 'train' / 'LR'

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    patterns = ['random', 'gradient', 'checkerboard', 'solid']

    for i in range(num_samples):
        pattern = np.random.choice(patterns)
        hr_img = generate_artificial_image(size=image_size, pattern_type=pattern)
        lr_img = downscale_bicubic(hr_img, scale_factor=scale_factor)

        hr_path = hr_dir / "04d"
        lr_path = lr_dir / "04d"

        hr_img.save(hr_path)
        lr_img.save(lr_path)

    return str(output_path)