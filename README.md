---
title: Sistema de Restauración y Mejora de Imágenes
emoji: 🎨
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🎨 Sistema Inteligente de Restauración y Mejora de Imágenes

¡Bienvenido! Este proyecto presenta una aplicación web innovadora que utiliza técnicas avanzadas de procesamiento digital de imágenes para mejorar automáticamente la calidad visual de tus fotografías. Diseñado tanto para usuarios casuales como para profesionales, nuestro sistema combina algoritmos de vanguardia con una interfaz intuitiva.

## 🌟 ¿Qué hace este sistema?

Imagina tener una herramienta que puede transformar tus fotos antiguas, borrosas o de baja calidad en imágenes nítidas y vibrantes con solo unos clics. Nuestro sistema aplica técnicas profesionales de restauración y mejora que normalmente requieren software especializado y conocimientos técnicos avanzados.

### ✨ Características principales

- **🖼️ Restauración Inteligente**: Corrige automáticamente imperfecciones, mejora la nitidez y optimiza los colores
- **🔍 Super-Resolución**: Aumenta la resolución de tus imágenes manteniendo la calidad
- **📊 Análisis de Calidad**: Mide objetivamente las mejoras con métricas técnicas profesionales
- **🌐 Interfaz Web Moderna**: Fácil de usar desde cualquier dispositivo con conexión a internet
- **⚡ Procesamiento Rápido**: Resultados en segundos, no en minutos

## 🎯 ¿Para quién es este proyecto?

- **📸 Fotógrafos aficionados** que quieren mejorar sus fotos sin software complejo
- **🏛️ Archiveros y museos** que necesitan restaurar colecciones de imágenes históricas
- **💼 Profesionales del marketing** que optimizan imágenes para redes sociales y web
- **🎓 Estudiantes y profesores** que aprenden sobre procesamiento digital de imágenes
- **👨‍💻 Desarrolladores** interesados en visión computacional y aplicaciones web

## 🚀 Cómo usar el sistema

### Opción 1: Versión en línea (Más fácil)

Visita nuestra aplicación desplegada en Hugging Face Spaces:
**https://huggingface.co/spaces/Ana-Lujan/Restauracion-enchancement**

1. Abre el enlace en tu navegador web
2. Arrastra y suelta tu imagen o haz clic para seleccionarla
3. Elige el tipo de mejora que deseas aplicar
4. ¡Listo! Tu imagen mejorada aparecerá automáticamente

### Opción 2: Ejecutar localmente (Para desarrolladores)

Si quieres ejecutar el sistema en tu propia computadora:

#### Requisitos previos
- Python 3.10 o superior instalado
- Conexión a internet para descargar las bibliotecas necesarias

#### Pasos de instalación
```bash
# 1. Clona este repositorio
git clone [URL-del-repositorio]
cd restauracion-enhancement

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Ejecuta la aplicación
python app.py
```

#### Acceso a la aplicación
Abre tu navegador web y ve a: **http://127.0.0.1:5000**

## 🛠️ Tecnologías utilizadas

Este proyecto combina varias tecnologías modernas para ofrecer una experiencia completa:

### Lenguajes y Frameworks
- **Python**: Lenguaje principal para el procesamiento de imágenes
- **Flask**: Framework web que crea la interfaz de usuario
- **HTML/CSS/JavaScript**: Tecnologías web para la interfaz interactiva

### Bibliotecas de Procesamiento de Imágenes
- **OpenCV**: Biblioteca profesional para visión computacional
- **Pillow (PIL)**: Procesamiento básico de imágenes
- **NumPy**: Computación numérica eficiente
- **Scikit-Image**: Algoritmos avanzados de procesamiento

### Infraestructura
- **Docker**: Contenedorización para despliegue consistente
- **Hugging Face Spaces**: Plataforma cloud para aplicaciones de IA
- **Gunicorn**: Servidor web optimizado para Python

## 📚 ¿Cómo funciona técnicamente?

### El proceso de mejora de imágenes

1. **📤 Carga de imagen**: Tu foto se sube de forma segura al servidor
2. **🔍 Análisis automático**: El sistema evalúa las características de la imagen
3. **⚙️ Aplicación de algoritmos**: Se ejecutan técnicas específicas según tu selección:
   - **Blanco y Negro Profesional**: Conversión con alto contraste usando CLAHE
   - **Mejora Perfecta**: Ajustes extremos de brillo, contraste y nitidez
   - **Belleza Facial**: Suavizado bilateral y optimización de colores
   - **Filtros Vintage**: Efectos retro con sepia y granulado cinematográfico
   - **Restauración**: Mejora general con filtros avanzados
4. **📊 Medición de calidad**: Cálculo de métricas técnicas (PSNR y SSIM)
5. **📥 Entrega del resultado**: Imagen mejorada lista para descargar

### Métricas de calidad explicadas

- **PSNR (Relación Señal-Ruido Pico)**: Mide qué tan diferente es la imagen procesada de la original. Valores más altos indican mejor calidad.
- **SSIM (Índice de Similitud Estructural)**: Evalúa qué tan similares se ven las imágenes para el ojo humano. Valores cercanos a 1 indican alta similitud visual.

## 🎓 Contexto académico

Este proyecto fue desarrollado como trabajo final para la materia **"Procesamiento de Imagen"** en el **IFTS 24 - Ciencia de Datos e Inteligencia Artificial**.

### Objetivos de aprendizaje cumplidos

- ✅ **Fundamentos de visión computacional**: Aplicación práctica de algoritmos de procesamiento de imágenes
- ✅ **Desarrollo web con Python**: Creación de aplicaciones interactivas usando Flask
- ✅ **Ingeniería de software**: Diseño modular, documentación y buenas prácticas
- ✅ **Evaluación cuantitativa**: Uso de métricas objetivas para medir el rendimiento
- ✅ **Despliegue en la nube**: Publicación de aplicaciones en plataformas modernas

### Metodología de desarrollo

1. **Análisis de requisitos**: Identificación de necesidades reales de mejora de imágenes
2. **Investigación técnica**: Estudio de algoritmos y bibliotecas disponibles
3. **Diseño de arquitectura**: Planificación de componentes y flujo de datos
4. **Implementación modular**: Desarrollo por componentes reutilizables
5. **Testing exhaustivo**: Validación funcional y de rendimiento
6. **Documentación completa**: Creación de guías para usuarios y desarrolladores

## 📈 Resultados y rendimiento

### Rendimiento técnico
- **Velocidad**: Procesamiento completo en menos de 2 segundos por imagen
- **Compatibilidad**: Funciona en computadoras estándar sin requerir hardware especial
- **Escalabilidad**: Maneja imágenes desde pequeños thumbnails hasta fotos de alta resolución
- **Confiabilidad**: Sistema robusto con manejo automático de errores

### Casos de uso exitosos
- **Restauración de fotos antiguas**: Recuperación de imágenes deterioradas por el tiempo
- **Optimización para web**: Preparación de imágenes para sitios web y redes sociales
- **Mejora fotográfica**: Corrección de problemas comunes en fotografía digital
- **Procesamiento por lotes**: Mejora masiva de colecciones de imágenes

## 🤝 Información del proyecto

### Equipo de desarrollo
- **Estudiante**: Ana Lujan
- **Profesor**: Matías Barreto
- **Institución**: IFTS 24
- **Materia**: Procesamiento de Imagen
- **Año**: 2025


## 📄 Licencia y uso

Este proyecto se distribuye bajo **licencia MIT**, lo que significa que puedes:

- ✅ Usarlo libremente para fines personales y comerciales
- ✅ Modificar el código según tus necesidades
- ✅ Distribuir copias del proyecto
- ✅ Usarlo en proyectos educativos y de investigación

## 🌟 Impacto y futuro

Este sistema demuestra cómo la tecnología moderna puede hacer accesibles técnicas avanzadas de procesamiento de imágenes. En el futuro, podríamos expandir las capacidades con:

- 🤖 **Inteligencia Artificial**: Modelos de aprendizaje profundo para mejoras aún más sofisticadas
- 📱 **Aplicaciones móviles**: Versión nativa para teléfonos y tablets
- 🎨 **Filtros personalizables**: Permitir a los usuarios crear sus propios estilos de mejora
- 📊 **Analytics avanzado**: Seguimiento detallado del uso y rendimiento
- 🌐 **API pública**: Integración con otras aplicaciones y servicios

---

**¡Gracias por explorar nuestro sistema de restauración de imágenes!**

Si tienes preguntas, sugerencias o quieres contribuir al proyecto, no dudes en contactarnos. Juntos podemos seguir mejorando el mundo del procesamiento digital de imágenes.

🎨✨📸

**Proyecto Final - Procesamiento de Imagen**
**IFTS 24 - Ciencia de Datos e Inteligencia Artificial**
