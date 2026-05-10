import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import time
import effects

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
    # 1. Rutas de los archivos
    ruta_entrada = "/home/txema/Escritorio/PDS-EfectosDeVoz/audio/audio.wav" # Asegúrate de tener este audio en la carpeta
    ruta_salida = "audio_modificado.wav"

    print("Iniciando VoiceLab...")
    try:
        # Carga la señal (Esto corresponde al módulo de Ignacio)
        audio_original, fs = cargar_y_normalizar(ruta_entrada)
        print("Audio cargado y normalizado correctamente.")

        # 2. APLICAR EFECTO Y VALIDAR TIEMPO (Tu parte)
        # Puedes cambiar effects.radio por effects.robot, effects.eco, effects.Alvin...
        nombre_del_efecto = "Efecto Radio Antigua"

        start_time = time.time()

        audio_procesado = effects.radio(audio_original, fs) # <-- Aplica el efecto aquí

        end_time = time.time()
        tiempo_ejecucion_ms = (end_time - start_time) * 1000

        # 3. Guardar el resultado para poder escucharlo
        guardar_audio(ruta_salida, audio_procesado, fs)
        print(f"Audio procesado guardado con éxito en: {ruta_salida}")

        # 4. EVALUACIÓN Y ESPECTROGRAMAS (Tu parte)
        evaluar_efecto(audio_original, audio_procesado, fs, nombre_del_efecto, tiempo_ejecucion_ms)

    except FileNotFoundError:
        print(f"⚠️ Error: No se ha encontrado el archivo '{ruta_entrada}'. Por favor, pon un archivo .wav de prueba en la misma carpeta.")
