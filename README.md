Markdown

# EfectosDeVoz 

Proyecto desarrollado para la asignatura de **Procesamiento Digital de Señales (PDS)** en el Grado de Ingeniería Informática, de la **Universidad de Granada (UGR)**.

## 📝 Descripción
**EfectosDeVoz** es un procesador digital de audio interactivo que permite aplicar transformaciones acústicas sobre señales de voz utilizando técnicas fundamentales de procesamiento de señales. El sistema cuenta con una **interfaz gráfica (GUI)** que permite trabajar en dos modalidades: procesar archivos `.wav` pregrabados o aplicar los **efectos en tiempo real** utilizando el micrófono del sistema.

## 🚀 Características y Efectos Implementados
El motor de audio (`effects.py`) incluye la implementación manual de los siguientes algoritmos matemáticos:
* **Efecto Alvin (Phase Vocoder):** Pitch shifting avanzado mediante la manipulación de la fase y magnitud en el dominio de la frecuencia (STFT/ISTFT) para alterar el tono sin modificar la duración temporal.
* **Radio Antigua (Filtro Pasa-banda):** Implementación de filtros IIR (Butterworth de orden 5) evaluando la ecuación en diferencias para limitar el ancho de banda.
* **Reverberación Vectorizada:** Simulación de reflexión acústica mediante filtros recursivos IIR optimizados computacionalmente (`scipy.signal.lfilter`).
* **Eco (Delay):** Retardos temporales con memoria implementados mediante buffers circulares (ideal para procesamiento por bloques en vivo).
* **Voz de Robot:** Modulación de Amplitud (AM) multiplicando la señal en el dominio temporal por una portadora sinusoidal de baja frecuencia.
* **Distorsión:** Modificación no lineal mediante amplificación de ganancia y recorte estático (clipping) con umbrales definidos.

## 📊 Evaluación Técnica y Análisis Crítico
Siguiendo las directrices académicas, el proyecto genera métricas automáticamente:
1.  **Análisis Espectral:** Cálculo de la **STFT (Short-Time Fourier Transform)** para generar espectrogramas comparativos (Original vs. Procesado).
2.  **Exportación Automática:** Generación automática de gráficas de alta resolución en la carpeta `/reportes` para documentar la alteración frecuencial.
3.  **Métricas de Rendimiento:** Medición precisa del coste computacional (tiempo de ejecución en milisegundos) de los algoritmos aplicados en procesamientos estáticos.
4.  **Normalización:** Garantía de fidelidad de audio para prevenir saturación pre y post procesamiento.

## 🛠️ Requisitos Técnicos
El proyecto requiere **Python 3** y las siguientes librerías científicas y de hardware:
* `numpy`: Manipulación de señales, buffers y operaciones matriciales.
* `scipy`: Diseño de filtros LTI, resampleo y transformadas (STFT/ISTFT).
* `matplotlib`: Generación y exportación de espectrogramas.
* `soundfile`: Gestión de lectura y escritura de audio estático.
* `sounddevice`: Captura y reproducción de flujos de audio de baja latencia.
* `tkinter`: Librería nativa de Python utilizada para la interfaz gráfica.

## 💻 Instalación y Uso

**1. Dependencias del sistema (Audio):**
Para el uso del micrófono en tiempo real, el sistema operativo necesita la librería base de audio C/C++ `portaudio`.
* **Linux (Arch/Manjaro):** `sudo pacman -S portaudio`
* **Linux (Ubuntu/Debian):** `sudo apt install libportaudio2`
* **Windows/macOS:** Normalmente preinstalado o gestionado internamente por `sounddevice`.

**2. Clonar el repositorio y crear el Entorno Virtual:**
```bash
git clone [https://github.com/TU_USUARIO/EfectosDeVoz.git](https://github.com/TU_USUARIO/EfectosDeVoz.git)
cd EfectosDeVoz
```

**3. Instalar dependencias de Python:**
```bash

pip install numpy scipy matplotlib soundfile sounddevice
```
**4. Ejecutar la aplicación:**


## 📂 Estructura del Proyecto
* `src/`: Código fuente del motor de efectos (`effects.py`) e interfaz (`main.py`).
* `audio/`: Archivos de audio de entrada y salida para pruebas.
* `notebooks/`: Entornos de prueba para análisis rápido.
* `docs/`: Documentación técnica y memorias del proyecto.

## 👥 Autores

* **Daniel López Cerrillo**
* **José María La Casa Molina**
* **Juan Antonio Miras Marín**
* **Ignacio Escalona Blanca**



**Tutor:** José Andrés González López  
**Curso:** 2025/2026  
**Institución:** ETSIIT - Universidad de Granada.