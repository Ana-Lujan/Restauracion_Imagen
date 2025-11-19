"""
Generación de dataset de super-resolución compatible con Hugging Face.
Crea imágenes HR/LR artificiales y las sube automáticamente.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from typing import Optional


def generate_artificial_image(size=(512, 512), pattern_type='random'):
    """
    Genera una imagen artificial de ejemplo para dataset de SR.

    Args:
        size: Tupla (width, height)
        pattern_type: Tipo de patrón ('random', 'gradient', 'checkerboard', 'solid')

    Returns:
        PIL Image
    """
    if pattern_type == 'random':
        # Imagen con colores aleatorios
        img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
    elif pattern_type == 'gradient':
        # Gradiente simple
        img = Image.new('RGB', size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        for y in range(size[1]):
            r = int(255 * (y / size[1]))
            g = int(255 * (1 - y / size[1]))
            b = 128
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    elif pattern_type == 'checkerboard':
        # Patrón de ajedrez
        img = Image.new('RGB', size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        square_size = 32
        for x in range(0, size[0], square_size):
            for y in range(0, size[1], square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    draw.rectangle([x, y, x+square_size, y+square_size], fill=(0, 0, 0))
    else:
        # Color sólido aleatorio
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', size, color=color)

    return img


def downscale_bicubic(img: Image.Image, scale_factor: int = 2) -> Image.Image:
    """
    Reduce la resolución usando interpolación bicubic.

    Args:
        img: Imagen PIL
        scale_factor: Factor de reducción

    Returns:
        Imagen PIL downscaled
    """
    w, h = img.size
    new_w, new_h = w // scale_factor, h // scale_factor
    return img.resize((new_w, new_h), Image.BICUBIC)


def create_dataset_structure(base_dir='dataset', num_images=50):
    """
    Crea la estructura del dataset con imágenes HR y LR.

    Args:
        base_dir: Directorio base
        num_images: Número de pares HR/LR a generar

    Returns:
        Path al directorio creado
    """
    hr_dir = Path(base_dir) / 'train' / 'HR'
    lr_dir = Path(base_dir) / 'train' / 'LR'

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    patterns = ['random', 'gradient', 'checkerboard', 'solid']

    print(f"🎨 Generando {num_images} pares HR/LR artificiales...")

    for i in range(num_images):
        # Elegir patrón aleatorio
        pattern = random.choice(patterns)

        # Generar imagen HR
        hr_img = generate_artificial_image(size=(512, 512), pattern_type=pattern)

        # Generar LR downscaled x2
        lr_img = downscale_bicubic(hr_img, scale_factor=2)

        # Nombres de archivo consistentes
        filename = "04d"

        # Guardar imágenes
        hr_path = hr_dir / filename
        lr_path = lr_dir / filename

        hr_img.save(hr_path)
        lr_img.save(lr_path)

        if (i + 1) % 10 == 0:
            print(f"✅ Generadas {i+1}/{num_images} imágenes")

    print("✅ Dataset generado localmente!")
    return base_dir


def create_dataset_info_json(base_dir='dataset', repo_name='AnaLujan/restauracion-superres'):
    """
    Crea el archivo dataset_info.json para compatibilidad con HF.

    Args:
        base_dir: Directorio base
        repo_name: Nombre del repo en HF
    """
    dataset_info = {
        "dataset_info": {
            "description": "Dataset de super-resolución con imágenes HR y LR para entrenamiento de modelos de restauración de imágenes.",
            "citation": "",
            "homepage": "",
            "license": "mit",
            "features": {
                "image": {
                    "decode": True,
                    "id": None,
                    "_type": "Image"
                },
                "label": {
                    "names": ["HR", "LR"],
                    "_type": "ClassLabel"
                }
            },
            "supervised_keys": {
                "input": "image",
                "output": "label"
            },
            "task_templates": [
                {
                    "task": "image-classification",
                    "image_column": "image",
                    "label_column": "label"
                }
            ],
            "builder_name": "imagefolder",
            "config_name": "default",
            "version": {
                "version_str": "1.0.0",
                "major": 1,
                "minor": 0,
                "patch": 0
            },
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": 0,
                    "num_examples": 100,
                    "dataset_name": repo_name
                }
            },
            "download_checksums": None,
            "download_size": 0,
            "post_processing_size": None,
            "dataset_size": 0,
            "size_in_bytes": 0
        }
    }

    import json
    info_path = Path(base_dir) / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print("📄 dataset_info.json creado!")


def create_readme(base_dir='dataset', repo_name='AnaLujan/restauracion-superres'):
    """
    Crea un README.md para el dataset.

    Args:
        base_dir: Directorio base
        repo_name: Nombre del repo
    """
    readme_content = f"""---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: class_label
    names:
      '0': HR
      '1': LR
  splits:
  - name: train
    num_examples: 100
  download_size: 0
  dataset_size: 0
---

# Dataset de Super-Resolución

Dataset artificial para entrenamiento de modelos de super-resolución.

## Estructura

- `train/HR/`: Imágenes de alta resolución (512×512)
- `train/LR/`: Imágenes de baja resolución (256×256, downscaled bicubic ×2)

## Uso

```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}", split="train")

# Acceder a imágenes
for sample in dataset:
    image = sample['image']  # PIL Image
    label = sample['label']  # 0 para HR, 1 para LR
```

## Generación

Imágenes generadas artificialmente con patrones aleatorios, gradientes y checkerboards.
"""

    readme_path = Path(base_dir) / 'README.md'
    with readme_path.open('w') as f:
        f.write(readme_content)

    print("📝 README.md creado!")


def upload_to_huggingface(
    base_dir='dataset',
    repo_name='AnaLujan/restauracion-superres',
    token: Optional[str] = None
) -> bool:
    """
    Sube el dataset a Hugging Face Hub.

    Args:
        base_dir: Directorio local
        repo_name: Nombre del repo en HF
        token: Token de HF (opcional, se pide si no se proporciona)

    Returns:
        True si exitoso
    """
    if token is None:
        print("❌ Error: Necesitas proporcionar un token de Hugging Face.")
        print("Obtén tu token en: https://huggingface.co/settings/tokens")
        return False

    api = HfApi(token=token)

    try:
        # Crear repo si no existe
        create_repo(repo_name, repo_type="dataset", exist_ok=True, token=token)
        print(f"📦 Repo {repo_name} creado/verificado!")

        # Subir archivos
        api.upload_folder(
            folder_path=base_dir,
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Subida inicial del dataset de super-resolución"
        )

        print("✅ Dataset subido exitosamente a Hugging Face Hub!")
        print(f"🔗 URL: https://huggingface.co/datasets/{repo_name}")

        return True

    except Exception as e:
        print(f"❌ Error al subir: {str(e)}")
        return False


def validate_dataset(base_dir='dataset') -> bool:
    """
    Valida que el dataset esté correctamente estructurado.

    Args:
        base_dir: Directorio a validar

    Returns:
        True si válido
    """
    base_path = Path(base_dir)
    hr_dir = base_path / 'train' / 'HR'
    lr_dir = base_path / 'train' / 'LR'

    if not hr_dir.exists():
        print(f"❌ Directorio HR no existe: {hr_dir}")
        return False

    if not lr_dir.exists():
        print(f"❌ Directorio LR no existe: {lr_dir}")
        return False

    hr_files = list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg"))
    lr_files = list(lr_dir.glob("*.png")) + list(lr_dir.glob("*.jpg"))

    if len(hr_files) == 0:
        print("❌ No hay imágenes HR")
        return False

    if len(lr_files) == 0:
        print("❌ No hay imágenes LR")
        return False

    if len(hr_files) != len(lr_files):
        print(f"❌ Número diferente de archivos: HR={len(hr_files)}, LR={len(lr_files)}")
        return False

    # Verificar pares
    hr_names = {f.stem for f in hr_files}
    lr_names = {f.stem for f in lr_files}

    if hr_names != lr_names:
        print("❌ Nombres de archivos no coinciden entre HR y LR")
        return False

    print(f"✅ Dataset válido: {len(hr_files)} pares HR/LR")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generar y subir dataset de super-resolución')
    parser.add_argument('--num_images', type=int, default=50,
                       help='Número de imágenes a generar (default: 50)')
    parser.add_argument('--repo_name', type=str, default='AnaLujan/restauracion-superres',
                       help='Nombre del repo en HF (default: AnaLujan/restauracion-superres)')
    parser.add_argument('--upload', action='store_true',
                       help='Subir a Hugging Face')
    parser.add_argument('--token', type=str, default=None,
                       help='Token de Hugging Face (requerido para upload)')

    args = parser.parse_args()

    # Generar dataset
    dataset_dir = create_dataset_structure(num_images=args.num_images)
    create_dataset_info_json(dataset_dir, args.repo_name)
    create_readme(dataset_dir, args.repo_name)

    # Validar
    if validate_dataset(dataset_dir):
        print(f"📁 Dataset creado en: {dataset_dir}")
        print("Estructura:")
        print("  dataset/")
        print("    train/")
        print("      HR/  (512x512 imágenes)")
        print("      LR/  (256x256 imágenes)")
        print("    dataset_info.json")
        print("    README.md")

        # Subir si se solicita
        if args.upload:
            if args.token:
                success = upload_to_huggingface(dataset_dir, args.repo_name, args.token)
                if success:
                    print("\n🎉 ¡Dataset listo para usar!")
                    print("from datasets import load_dataset")
                    print(f'dataset = load_dataset("{args.repo_name}", split="train")')
            else:
                print("❌ Para subir, proporciona --token TU_TOKEN_HF")
    else:
        print("❌ Dataset no válido, revisa la estructura")