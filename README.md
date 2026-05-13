# EfectosDeVoz (VoiceLab) 🎙️

Proyecto desarrollado para la asignatura de **Procesamiento Digital de Señales (PDS)** en el Grado de Ingeniería Informática, de la **Universidad de Granada (UGR)**.

## 📝 Descripción
**EfectosDeVoz** es un procesador digital de audio que permite aplicar transformaciones acústicas sobre señales de voz utilizando técnicas fundamentales de procesamiento de señales. El sistema permite cargar archivos `.wav`, procesarlos mediante diversos algoritmos matemáticos y evaluar el impacto tanto visualmente (espectrogramas) como en rendimiento (coste computacional).

## 🚀 Características y Efectos
El sistema incluye la implementación manual de los siguientes efectos:
* **Eco (Delay):** Basado en retardos temporales con memoria.
* **Reverberación:** Simulación de reflexión acústica mediante filtros recursivos.
* **Radio Antigua (Filtro Pasa-banda):** Implementación de filtros IIR (Butterworth) para limitar el ancho de banda.
* **Voz de Robot:** Modulación de Amplitud (AM) multiplicando la señal por una portadora sinusoidal.
* **Efecto Alvin:** Pitch shifting mediante técnicas de resampling.
* **Distorsión:** Clipping de señal mediante ganancia y umbral de saturación.

## 📊 Evaluación Técnica
Siguiendo las directrices académicas, el proyecto incluye:
1.  **Análisis Espectral:** Visualización de la **STFT (Short-Time Fourier Transform)** mediante espectrogramas para validar los cambios en el dominio de la frecuencia.
2.  **Métricas de Rendimiento:** Medición precisa del coste computacional en milisegundos por cada efecto aplicado.
3.  **Normalización:** Garantía de fidelidad de audio mediante normalización de amplitud pre y post procesamiento.

## 🛠️ Requisitos Técnicos
Para ejecutar este proyecto, necesitas Python 3 y las siguientes librerías científicas:
* `NumPy`: Manipulación de señales como vectores.
* `SciPy`: Diseño de filtros y procesamiento avanzado.
* `Matplotlib`: Generación de gráficas y espectrogramas.
* `Soundfile`: Gestión de lectura y escritura de audio.

## 💻 Instalación y Uso
1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/TU_USUARIO/EfectosDeVoz.git
    cd EfectosDeVoz
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install numpy scipy matplotlib soundfile
    ```
3.  **Ejecutar la aplicación:**
    ```bash
    python src/main.py
    ```

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