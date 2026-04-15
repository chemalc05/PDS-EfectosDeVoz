import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import time

def cargar_y_normalizar(ruta_archivo):
    """
   DOCUMENTO DE PRUEBA
    """
    # [Requisito 1.5.2]: Carga y normalización
    data, fs = sf.read(ruta_archivo)

    # Convertir a mono si es necesario
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalización
    data_norm = data / np.max(np.abs(data))
    return data_norm, fs

def guardar_audio(ruta_salida, data, fs):
    """Guarda la señal procesada en un archivo .wav."""
    sf.write(ruta_salida, data, fs)

def evaluar_efecto(original, procesado, fs, nombre_efecto, tiempo_ms):
    """
    Genera la evaluación cuantitativa y visual exigida por el profesor.
    Muestra espectrogramas y tiempo de ejecución.
    """
    # [Requisito 1.5.3]: Evaluación y Salida
    print(f"\n--- Análisis de: {nombre_efecto} ---")
    print(f"Coste computacional: {tiempo_ms:.2f} ms")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Espectrograma señal original
    ax1.specgram(original, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    ax1.set_title(f"Espectrograma Original")
    ax1.set_ylabel("Frecuencia [Hz]")

    # Espectrograma señal procesada
    ax2.specgram(procesado, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    ax2.set_title(f"Espectrograma con {nombre_efecto}")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_xlabel("Tiempo [s]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Este bloque
