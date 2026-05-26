import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import effects

from IPython.display import Audio


def cargar_y_normalizar(ruta_archivo):
    """
   DOCUMENTO DE PRUEBA
    """
    # [Requisito 1.5.2]: Carga y normalizacion
    data, fs = sf.read(ruta_archivo)

    # Convertir a mono si es necesario
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalizacion
    data_norm = data / np.max(np.abs(data))
    return data_norm, fs


def guardar_audio(ruta_salida, data, fs):
    """Guarda la senal procesada en un archivo .wav."""
    sf.write(ruta_salida, data, fs)


def reproducir_audio(data, fs):
    """Reproduce un audio usando sounddevice."""
    sd.stop()
    sd.play(data, fs)


def evaluar_efecto(original, procesado, fs, nombre_efecto, tiempo_ms):
    """
    Genera la evaluacion cuantitativa y visual exigida por el profesor.
    Muestra espectrogramas y tiempo de ejecucion.
    """
    # [Requisito 1.5.3]: Evaluacion y Salida
    print(f"\n--- Analisis de: {nombre_efecto} ---")
    print(f"Coste computacional: {tiempo_ms:.2f} ms")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Espectrograma senal original
    ax1.specgram(original, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    ax1.set_title("Espectrograma Original")
    ax1.set_ylabel("Frecuencia [Hz]")

    # Espectrograma senal procesada
    ax2.specgram(procesado, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    ax2.set_title(f"Espectrograma con {nombre_efecto}")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_xlabel("Tiempo [s]")
    plt.tight_layout()
    plt.show()


def crear_interfaz():
    ruta_entrada = "audio/audio.wav"
    ruta_salida = "audio_modificado.wav"
    audio_original = None
    audio_procesado = None
    fs = None

    efectos = {
        "Radio Antigua": lambda audio, fs_audio: effects.radio(audio, fs_audio),
        "Robot": lambda audio, fs_audio: effects.robot(audio, fs_audio),
        "Eco": lambda audio, fs_audio: effects.eco(audio, fs_audio),
        "Alvin": lambda audio, fs_audio: effects.Alvin(audio),
        "Distorsion": lambda audio, fs_audio: effects.distortion(audio,4,0.7),
        "Reverb": lambda audio, fs_audio: effects.reverb(audio, fs_audio, delay=0.04, decay=0.45),
    }

    ventana = tk.Tk()
    ventana.title("VoiceLab - Efectos de voz")
    ventana.geometry("430x350")
    ventana.resizable(False, False)

    ruta_var = tk.StringVar(value=ruta_entrada)
    efecto_var = tk.StringVar(value="Radio Antigua")
    estado_var = tk.StringVar(value="Carga un audio para empezar.")
    tiempo_var = tk.StringVar(value="Coste computacional: -")

    def cargar_audio_interfaz():
        nonlocal ruta_entrada, audio_original, audio_procesado, fs

        ruta = filedialog.askopenfilename(
            initialdir="audio",
            title="Selecciona un archivo de audio",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )

        if not ruta:
            return

        try:
            ruta_entrada = ruta
            audio_original, fs = cargar_y_normalizar(ruta_entrada)
            audio_procesado = None
            ruta_var.set(ruta_entrada)
            duracion = len(audio_original) / fs
            estado_var.set(f"Audio cargado: {duracion:.2f} s, {fs} Hz")
            tiempo_var.set("Coste computacional: -")
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo cargar el audio:\n{error}")

    def aplicar_efecto_interfaz():
        nonlocal audio_procesado

        if audio_original is None:
            messagebox.showwarning("Aviso", "Primero carga un audio.")
            return

        nombre_del_efecto = efecto_var.get()
        start_time = time.time()
        audio_procesado = efectos[nombre_del_efecto](audio_original, fs)
        end_time = time.time()

        tiempo_ejecucion_ms = (end_time - start_time) * 1000
        tiempo_var.set(f"Coste computacional: {tiempo_ejecucion_ms:.2f} ms")
        estado_var.set(f"Efecto aplicado: {nombre_del_efecto}")

        evaluar_efecto(audio_original, audio_procesado, fs, nombre_del_efecto, tiempo_ejecucion_ms)

    def reproducir_audio_interfaz():
        if audio_procesado is None:
            messagebox.showwarning("Aviso", "Primero aplica un efecto.")
            return

        reproducir_audio(audio_procesado, fs)
        estado_var.set("Reproduciendo audio procesado...")

    def guardar_audio_interfaz():
        if audio_procesado is None:
            messagebox.showwarning("Aviso", "Primero aplica un efecto.")
            return

        guardar_audio(ruta_salida, audio_procesado, fs)
        estado_var.set(f"Audio procesado guardado en: {ruta_salida}")
        messagebox.showinfo("Guardado", f"Audio guardado en {ruta_salida}")

    ttk.Label(ventana, text="VoiceLab", font=("Arial", 18, "bold")).pack(pady=(14, 8))

    ttk.Label(ventana, text="Archivo de entrada:").pack(anchor="w", padx=20)
    ttk.Label(ventana, textvariable=ruta_var, wraplength=380).pack(anchor="w", padx=20, pady=(0, 8))
    ttk.Button(ventana, text="Cargar audio", command=cargar_audio_interfaz).pack(fill="x", padx=20)

    ttk.Label(ventana, text="Efecto de voz:").pack(anchor="w", padx=20, pady=(14, 0))
    ttk.Combobox(
        ventana,
        textvariable=efecto_var,
        values=list(efectos.keys()),
        state="readonly",
    ).pack(fill="x", padx=20)

    ttk.Button(ventana, text="Aplicar efecto", command=aplicar_efecto_interfaz).pack(
        fill="x", padx=20, pady=(16, 6)
    )

    ttk.Button(
        ventana,
        text="Reproducir audio modificado",
        command=reproducir_audio_interfaz
    ).pack(fill="x", padx=20, pady=(0, 6))

    ttk.Button(ventana, text="Guardar audio modificado", command=guardar_audio_interfaz).pack(
        fill="x", padx=20
    )

    ttk.Label(ventana, textvariable=tiempo_var).pack(anchor="w", padx=20, pady=(14, 0))
    ttk.Label(ventana, textvariable=estado_var, wraplength=380).pack(anchor="w", padx=20, pady=(4, 0))

    ventana.mainloop()


if __name__ == "__main__":
    crear_interfaz()