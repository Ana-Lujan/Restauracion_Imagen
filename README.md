---
title: Sistema de Restauración y Mejora de Imágenes
emoji: 🎨
colorFrom: blue
colorTo: green
sdk: static
app_file: app.py
pinned: false
---

# Sistema de Restauración y Mejora de Imágenes

## 📋 Descripción

Este proyecto implementa un sistema avanzado de procesamiento de imágenes que mejora automáticamente la calidad visual de fotografías mediante técnicas de visión computacional. El sistema está diseñado para ejecutarse en la nube utilizando Hugging Face Spaces.

## 🎯 Funcionalidades

- **Restauración de Imágenes**: Corrección automática de ruido, mejora de nitidez y balance de color
- **Super-Resolución**: Aumento de resolución utilizando algoritmos de interpolación avanzada
- **Métricas de Calidad**: Evaluación cuantitativa con PSNR y SSIM
- **Interfaz Web**: Aplicación interactiva con drag & drop

## 🏗️ Arquitectura Técnica

### Tecnologías Utilizadas
- **Python 3.10**: Lenguaje de programación principal
- **Flask**: Framework web para la interfaz
- **OpenCV**: Biblioteca de visión computacional
- **NumPy**: Computación numérica
- **Pillow**: Procesamiento de imágenes
- **Docker**: Contenedorización para deployment

### Estructura del Sistema
```
├── app.py              # Aplicación Flask principal
├── requirements.txt    # Dependencias del proyecto
├── Dockerfile         # Configuración de contenedor
├── README.md          # Documentación
└── Procfile           # Configuración de servidor
```

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.10 o superior
- Docker (para deployment local)
- Git

### Instalación Local
```bash
# Clonar repositorio
git clone <repository-url>
cd restauracion-imagenes

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python app.py
```

### Acceso a la Aplicación
Una vez ejecutada, acceder en: `http://127.0.0.1:5000` (desarrollo local) o `http://localhost:7860` (HF Spaces)

## 📊 Algoritmos Implementados

### Restauración de Imágenes
- **Denoising Bilateral**: Reducción de ruido preservando bordes
- **Sharpening**: Mejora de nitidez con filtros de realce
- **Corrección de Color**: Ajuste automático de balance de blancos

### Super-Resolución
- **Interpolación Bicúbica**: Método clásico de aumento de resolución
- **Procesamiento Adaptativo**: Ajustes basados en contenido de imagen

### Métricas de Evaluación
- **PSNR (Peak Signal-to-Noise Ratio)**: Medida de calidad de reconstrucción
- **SSIM (Structural Similarity Index)**: Evaluación de similitud estructural

## 🔬 Metodología

### Enfoque de Desarrollo
1. **Análisis de Requisitos**: Identificación de problemas comunes en imágenes
2. **Diseño de Algoritmos**: Selección de técnicas apropiadas para cada tipo de mejora
3. **Implementación Modular**: Código organizado en funciones reutilizables
4. **Testing y Validación**: Verificación de resultados con métricas cuantitativas
5. **Optimización**: Ajustes para rendimiento en entornos cloud

### Evaluación de Resultados
- **Métricas Objetivas**: PSNR y SSIM para medición cuantitativa
- **Evaluación Subjetiva**: Análisis visual de mejoras percibidas
- **Comparación de Métodos**: Benchmarking contra técnicas estándar

## 📈 Resultados

### Rendimiento del Sistema
- **Tiempo de Procesamiento**: < 2 segundos por imagen
- **Compatibilidad**: Funciona en CPU estándar
- **Escalabilidad**: Procesamiento de imágenes de diversos tamaños

### Casos de Uso
1. **Mejora de Fotografías Antiguas**: Restauración de imágenes deterioradas
2. **Optimización Web**: Preparación de imágenes para internet
3. **Procesamiento Batch**: Mejora masiva de colecciones de imágenes

## 🤝 Contribución

### Información del Proyecto
- **Institución**: IFTS 24
- **Materia**: Procesamiento de Imagen
- **Profesor**: Matías Barreto
- **Estudiante**: Ana Lujan

### Desarrollo Colaborativo
Este proyecto fue desarrollado siguiendo metodologías de ingeniería de software, con énfasis en la reproducibilidad y documentación técnica.

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT para uso educativo y de investigación.

## 🙏 Agradecimientos

- **Profesor Matías Barreto**: Por la guía académica y metodológica
- **Hugging Face**: Por la plataforma de deployment
- **Comunidad Open Source**: Por las bibliotecas utilizadas

---

**Proyecto Final - Procesamiento de Imagen**
**IFTS 24 - Ciencia de Datos e Inteligencia Artificial**