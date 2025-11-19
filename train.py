"""
Entrenamiento de modelo SRCNN para super-resolución.
Script completo y funcional para CPU con métricas profesionales.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Imports del proyecto
from src.dataset import SuperResolutionDataset, HFDatasetAdapter
from src.models import create_model, save_model_checkpoint
from src.metrics import SRMetrics, log_training_metrics, evaluate_model
from src.utils import create_data_transforms


def train_model(
    epochs: int = 10,
    batch_size: int = 8,
    scale_factor: int = 2,
    learning_rate: float = 1e-3,
    dataset_path: str = None,
    hf_dataset: str = "MCG-NJU/vdsr-2k",
    model_name: str = "srcnn"
):
    """
    Función principal de entrenamiento del modelo de super-resolución.

    Args:
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        scale_factor: Factor de escala (2 o 4)
        learning_rate: Tasa de aprendizaje
        dataset_path: Ruta a dataset local (opcional)
        hf_dataset: Nombre del dataset en HF
        model_name: Nombre del modelo a usar
    """
    print(f"🚀 Iniciando entrenamiento {model_name.upper()}")
    print(f"   📊 Epochs: {epochs}, Batch size: {batch_size}, Scale: {scale_factor}x")
    print(f"   🎯 Learning rate: {learning_rate}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   💻 Device: {device}")

    # Preparar dataset
    if dataset_path and Path(dataset_path).exists():
        # Usar dataset local
        hr_dir = Path(dataset_path) / "train" / "HR"
        lr_dir = Path(dataset_path) / "train" / "LR"

        if hr_dir.exists() and lr_dir.exists():
            print(f"📁 Usando dataset local: {dataset_path}")
            train_dataset = SuperResolutionDataset(str(hr_dir), str(lr_dir), scale_factor)
            # Para validación, usar mismo dataset (en producción separar)
            val_dataset = train_dataset
        else:
            raise FileNotFoundError(f"Directorios HR/LR no encontrados en {dataset_path}")
    else:
        # Usar dataset de Hugging Face
        print(f"📥 Cargando dataset HF: {hf_dataset}")
        hf_data = load_dataset(hf_dataset, split='train')

        # Dividir en train/val
        train_size = int(0.8 * len(hf_data))
        train_split = hf_data.select(range(train_size))
        val_split = hf_data.select(range(train_size, len(hf_data)))

        # Crear datasets adaptados
        train_dataset = HFDatasetAdapter(train_split, scale_factor=scale_factor)
        val_dataset = HFDatasetAdapter(val_split, scale_factor=scale_factor)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU only
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"   📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # Crear modelo
    model = create_model(model_name, scale_factor=scale_factor)
    model.to(device)

    print(f"   🧠 Modelo: {model.__class__.__name__}")
    print(f"   🔢 Parámetros: {model.get_num_params():,}")

    # Loss y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Métricas
    metrics = SRMetrics(device=device)

    # Directorios
    model_dir = Path("model")
    samples_dir = Path("samples")
    model_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    # Variables de seguimiento
    best_psnr = 0.0
    training_log = []

    print("\n🏃 Iniciando entrenamiento...\n")

    # Loop de entrenamiento
    for epoch in range(epochs):
        # === ENTRENAMIENTO ===
        model.train()
        train_loss = 0.0
        train_metrics = {'psnr': 0.0, 'ssim': 0.0}
        num_train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [Train]")
        for lr_batch, hr_batch in train_pbar:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            outputs = model(lr_batch)
            loss = criterion(outputs, hr_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calcular métricas en batch
            batch_metrics = metrics(lr_batch, hr_batch)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key].item()
            num_train_batches += 1

            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{batch_metrics['psnr'].item():.2f}"
            })

        # Promedios de entrenamiento
        avg_train_loss = train_loss / len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= num_train_batches

        # === VALIDACIÓN ===
        model.eval()
        val_loss = 0.0
        val_metrics = {'psnr': 0.0, 'ssim': 0.0}
        num_val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [Val]  ")
            for lr_batch, hr_batch in val_pbar:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                outputs = model(lr_batch)
                loss = criterion(outputs, hr_batch)
                val_loss += loss.item()

                # Calcular métricas
                batch_metrics = metrics(lr_batch, hr_batch)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key].item()
                num_val_batches += 1

                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'psnr': f"{batch_metrics['psnr'].item():.2f}"
                })

        # Promedios de validación
        avg_val_loss = val_loss / len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_val_batches

        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_psnr': train_metrics['psnr'],
            'val_psnr': val_metrics['psnr'],
            'train_ssim': train_metrics['ssim'],
            'val_ssim': val_metrics['ssim']
        }
        training_log.append(log_entry)

        # Imprimir métricas
        print(f"\n📈 Epoch {epoch+1:2d} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
        print(f"   Train PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"   Val PSNR:   {val_metrics['psnr']:.2f} dB")
        print(f"   Train SSIM: {train_metrics['ssim']:.4f}")
        print(f"   Val SSIM:   {val_metrics['ssim']:.4f}")
        # Guardar mejor modelo
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            checkpoint_path = model_dir / f"{model_name}_best.pth"
            save_model_checkpoint(
                model, optimizer, epoch + 1,
                avg_val_loss, str(checkpoint_path)
            )
            print(f"💾 Mejor modelo guardado (PSNR: {best_psnr:.2f})")

        # Guardar samples cada 5 epochs
        if (epoch + 1) % 5 == 0:
            _save_training_samples(model, val_loader, epoch + 1, samples_dir, device)

    # Guardar modelo final
    final_checkpoint = model_dir / f"{model_name}_final.pth"
    save_model_checkpoint(model, optimizer, epochs, avg_val_loss, str(final_checkpoint))
    print(f"✅ Entrenamiento completado! Modelo final guardado en {final_checkpoint}")

    # Log final
    _save_training_log(training_log, model_dir / "training_log.txt")

    return model


def _save_training_samples(model, val_loader, epoch, save_dir, device, num_samples=2):
    """Guarda imágenes de ejemplo durante el entrenamiento."""
    model.eval()

    with torch.no_grad():
        lr_batch, hr_batch = next(iter(val_loader))
        lr_batch = lr_batch[:num_samples].to(device)
        hr_batch = hr_batch[:num_samples].to(device)

        sr_batch = model(lr_batch)

        # Convertir a imágenes
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()

        for i in range(num_samples):
            # LR
            lr_img = to_pil(lr_batch[i].cpu())
            lr_img.save(save_dir / f"epoch_{epoch}_sample_{i}_lr.png")
            # SR
            sr_img = to_pil(torch.clamp(sr_batch[i].cpu(), 0, 1))
            sr_img.save(save_dir / f"epoch_{epoch}_sample_{i}_sr.png")
            # HR
            hr_img = to_pil(hr_batch[i].cpu())
            hr_img.save(save_dir / f"epoch_{epoch}_sample_{i}_hr.png")
def _save_training_log(log_data, log_path):
    """Guarda el log de entrenamiento en formato legible."""
    with open(log_path, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Epoch':<8}")
        f.write("-" * 70 + "\n")

        for entry in log_data:
            f.write(f"{entry['epoch']:<8}")
            f.write(f"{entry['train_loss']:<12.4f}")
            f.write(f"{entry['val_loss']:<12.4f}")
            f.write(f"{entry['train_psnr']:<10.2f}")
            f.write(f"{entry['val_psnr']:<10.4f}")
            f.write(f"{entry['train_ssim']:<10.2f}")
            f.write(f"{entry['val_ssim']:<10.4f}")
            f.write("\n")

    print(f"📝 Log guardado en {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo de Super-Resolución')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Número de epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Tamaño del batch (default: 8)')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                       help='Factor de escala (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Ruta a dataset local (opcional)')
    parser.add_argument('--hf_dataset', type=str, default='MCG-NJU/vdsr-2k',
                       help='Dataset de HF a usar (default: MCG-NJU/vdsr-2k)')
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'enhanced_srcnn'],
                       help='Modelo a usar (default: srcnn)')

    args = parser.parse_args()

    # Ejecutar entrenamiento
    trained_model = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        scale_factor=args.scale,
        learning_rate=args.lr,
        dataset_path=args.dataset_path,
        hf_dataset=args.hf_dataset,
        model_name=args.model
    )

    print("🎉 ¡Entrenamiento completado exitosamente!")