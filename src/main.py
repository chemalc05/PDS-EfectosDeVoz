import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import effects
import os

try:
    import sounddevice as sd
except ImportError:
    sd = None


class ProcesadorTiempoReal:
    """Procesa bloques de microfono manteniendo la memoria de cada efecto."""

    def __init__(self, nombre_efecto, fs):
        self.nombre_efecto = nombre_efecto
        self.fs = fs
        self.sample_index = 0

        self.radio_b = None
        self.radio_a = None
        self.radio_zi = None
        self.eco_buffer = None
        self.eco_pos = 0
        self.reverb_buffer = None
        self.reverb_pos = 0
        self.alvin_vocoder = None

        if nombre_efecto == "Radio Antigua":
            nyq = 0.5 * fs
            self.radio_b, self.radio_a = signal.butter(
                5, [300.0 / nyq, 3000.0 / nyq], btype="band"
            )
            self.radio_zi = np.zeros(
                max(len(self.radio_a), len(self.radio_b)) - 1, dtype=np.float32
            )
        elif nombre_efecto == "Eco":
            self.eco_buffer = np.zeros(int(0.5 * fs), dtype=np.float32)
        elif nombre_efecto == "Reverb":
            self.reverb_buffer = np.zeros(int(0.04 * fs), dtype=np.float32)
        elif nombre_efecto == "Alvin":
            self.alvin_vocoder = effects.PhaseVocoderPitchShift(pitch_factor=1.5)

    def procesar(self, bloque):
        bloque = np.asarray(bloque, dtype=np.float32)

        if self.nombre_efecto == "Radio Antigua":
            salida, self.radio_zi = signal.lfilter(
                self.radio_b, self.radio_a, bloque, zi=self.radio_zi
            )
        elif self.nombre_efecto == "Robot":
            indices = np.arange(len(bloque), dtype=np.float32) + self.sample_index
            portadora = np.sin(2 * np.pi * 40.0 * indices / self.fs)
            salida = bloque * portadora
            self.sample_index += len(bloque)
        elif self.nombre_efecto == "Eco":
            salida = self._procesar_eco(bloque)
        elif self.nombre_efecto == "Alvin":
            salida = self.alvin_vocoder.process(bloque)
        elif self.nombre_efecto == "Distorsion":
            salida = effects.distortion(bloque)
        elif self.nombre_efecto == "Reverb":
            salida = self._procesar_reverb(bloque)
        else:
            salida = bloque

        return np.clip(salida, -1.0, 1.0).astype(np.float32)

    def _procesar_eco(self, bloque):
        salida = np.empty_like(bloque)

        for i, muestra in enumerate(bloque):
            retrasada = self.eco_buffer[self.eco_pos]
            salida[i] = muestra + 0.6 * retrasada
            self.eco_buffer[self.eco_pos] = muestra
            self.eco_pos = (self.eco_pos + 1) % len(self.eco_buffer)

        return salida

    def _procesar_reverb(self, bloque):
        salida = np.empty_like(bloque)

        for i, muestra in enumerate(bloque):
            retrasada = self.reverb_buffer[self.reverb_pos]
            salida[i] = muestra + 0.45 * retrasada
            self.reverb_buffer[self.reverb_pos] = salida[i]
            self.reverb_pos = (self.reverb_pos + 1) % len(self.reverb_buffer)

        return salida


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
    if sd is None:
        raise RuntimeError("sounddevice no esta instalado.")

    sd.stop()
    sd.play(data, fs)

import os

def evaluar_efecto(original, procesado, fs, nombre_efecto, tiempo_ms):
    """
    Genera la evaluacion cuantitativa y visual exigida por el profesor.
    Muestra espectrogramas, tiempo de ejecucion y GUARDA la imagen.
    """
    print(f"\n--- Analisis de: {nombre_efecto} ---")
    print(f"Coste computacional: {tiempo_ms:.2f} ms")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Aumentamos NFFT a 4096 para mayor resolucion frecuencial (poder ver bandas AM de 40Hz)
    nfft_val = 4096
    noverlap_val = 2048

    # Espectrograma senal original
    ax1.specgram(original, Fs=fs, NFFT=nfft_val, noverlap=noverlap_val, cmap='viridis')
    ax1.set_title("Espectrograma Original")
    ax1.set_ylabel("Frecuencia [Hz]")
    ax1.set_ylim(0, 5000)  # ZOOM: Solo nos interesa el rango vocal hasta 5kHz

    # Espectrograma senal procesada
    ax2.specgram(procesado, Fs=fs, NFFT=nfft_val, noverlap=noverlap_val, cmap='viridis')
    ax2.set_title(f"Espectrograma con {nombre_efecto} (Coste: {tiempo_ms:.2f} ms)")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylim(0, 5000)  # ZOOM: Solo nos interesa el rango vocal hasta 5kHz

    plt.tight_layout()
    
    # Guardado automático
    os.makedirs("reportes", exist_ok=True)
    nombre_archivo = f"reportes/espectrograma_{nombre_efecto.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"Gráfica guardada en: {nombre_archivo}")
    
    plt.show()

def crear_interfaz():
    ruta_entrada = "audio/audio.wav"
    ruta_salida = "audio_modificado.wav"
    audio_original = None
    audio_procesado = None
    fs = None
    stream_tiempo_real = None
    bloques_originales = []
    bloques_procesados = []
    visualizacion_lock = threading.Lock()
    visualizacion = {
        "original": np.zeros(1024, dtype=np.float32),
        "procesado": np.zeros(1024, dtype=np.float32),
    }

    efectos = {
        "Radio Antigua": lambda audio, fs_audio: effects.radio(audio, fs_audio),
        "Robot": lambda audio, fs_audio: effects.robot(audio, fs_audio),
        "Eco": lambda audio, fs_audio: effects.eco(audio, fs_audio),
        "Alvin": lambda audio, fs_audio: effects.AlvinVocoder(audio, fs_audio),
        "Distorsion": lambda audio, fs_audio: effects.distortion(audio),
        "Reverb": lambda audio, fs_audio: effects.reverb(audio, fs_audio, delay=0.04, decay=0.45),
    }

    ventana = tk.Tk()
    ventana.title("Efectos de voz")
    ventana.geometry("520x650")
    ventana.resizable(False, False)

    style = ttk.Style(ventana)
    style.configure("Title.TLabel", font=("Arial", 19, "bold"))
    style.configure("Section.TLabelframe", padding=12)
    style.configure("Primary.TButton", padding=(10, 6))
    style.configure("Status.TLabel", foreground="#334155")

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
            actualizar_visualizacion(audio_original, np.zeros_like(audio_original[-4096:]))
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
        actualizar_visualizacion(audio_original, audio_procesado)

        evaluar_efecto(audio_original, audio_procesado, fs, nombre_del_efecto, tiempo_ejecucion_ms)

    def reproducir_audio_interfaz():
        if audio_procesado is None:
            messagebox.showwarning("Aviso", "Primero aplica un efecto.")
            return

        try:
            reproducir_audio(audio_procesado, fs)
            estado_var.set("Reproduciendo audio procesado...")
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo reproducir el audio:\n{error}")

    def guardar_audio_interfaz():
        if audio_procesado is None:
            messagebox.showwarning("Aviso", "Primero aplica un efecto.")
            return

        guardar_audio(ruta_salida, audio_procesado, fs)
        estado_var.set(f"Audio procesado guardado en: {ruta_salida}")
        messagebox.showinfo("Guardado", f"Audio guardado en {ruta_salida}")

    def actualizar_visualizacion(bloque_original=None, bloque_procesado=None):
        with visualizacion_lock:
            if bloque_original is not None:
                visualizacion["original"] = np.asarray(
                    bloque_original[-4096:], dtype=np.float32
                ).copy()
            if bloque_procesado is not None:
                visualizacion["procesado"] = np.asarray(
                    bloque_procesado[-4096:], dtype=np.float32
                ).copy()

    def dibujar_canal(canvas, bloque, centro_y, altura, color, etiqueta, ancho):
        canvas.create_line(0, centro_y, ancho, centro_y, fill="#cbd5e1")
        canvas.create_text(10, centro_y - altura + 8, text=etiqueta, anchor="w", fill="#475569")

        if bloque is None or len(bloque) == 0:
            return

        muestras = bloque
        puntos_visibles = max(2, ancho - 20)
        if len(muestras) > puntos_visibles:
            indices = np.linspace(0, len(muestras) - 1, puntos_visibles).astype(int)
            muestras = muestras[indices]

        maximo = max(float(np.max(np.abs(muestras))), 0.05)
        escala = (altura - 14) / maximo
        paso_x = (ancho - 20) / max(len(muestras) - 1, 1)
        puntos = []

        for i, muestra in enumerate(muestras):
            x = 10 + i * paso_x
            y = centro_y - float(muestra) * escala
            puntos.extend((x, y))

        if len(puntos) >= 4:
            canvas.create_line(puntos, fill=color, width=2, smooth=True)

    def dibujar_visualizacion():
        with visualizacion_lock:
            bloque_original = visualizacion["original"].copy()
            bloque_procesado = visualizacion["procesado"].copy()

        ancho = max(onda_canvas.winfo_width(), 440)
        alto = max(onda_canvas.winfo_height(), 160)
        onda_canvas.delete("all")
        onda_canvas.create_rectangle(0, 0, ancho, alto, fill="#f8fafc", outline="#cbd5e1")

        dibujar_canal(
            onda_canvas,
            bloque_original,
            alto * 0.32,
            alto * 0.24,
            "#2563eb",
            "Entrada",
            ancho,
        )
        dibujar_canal(
            onda_canvas,
            bloque_procesado,
            alto * 0.76,
            alto * 0.24,
            "#16a34a",
            "Salida",
            ancho,
        )

        ventana.after(50, dibujar_visualizacion)

    def obtener_fs_tiempo_real():
        if sd is None:
            return 44100

        try:
            return int(sd.query_devices(kind="input")["default_samplerate"])
        except Exception:
            return 44100

    def iniciar_tiempo_real():
        nonlocal stream_tiempo_real, bloques_originales, bloques_procesados, fs

        if sd is None:
            messagebox.showerror(
                "Dependencia no instalada",
                "Para usar audio en tiempo real instala sounddevice:\n"
                "python -m pip install sounddevice",
            )
            return

        if stream_tiempo_real is not None:
            messagebox.showwarning("Aviso", "Ya hay una grabacion en tiempo real activa.")
            return

        nombre_del_efecto = efecto_var.get()

        try:
            fs = obtener_fs_tiempo_real()
            procesador = ProcesadorTiempoReal(nombre_del_efecto, fs)
            bloques_originales = []
            bloques_procesados = []
            actualizar_visualizacion(
                np.zeros(1024, dtype=np.float32),
                np.zeros(1024, dtype=np.float32),
            )

            def callback(indata, outdata, frames, time_info, status):
                bloque_original = indata[:, 0].copy()
                bloque_procesado = procesador.procesar(bloque_original)

                outdata[:, 0] = bloque_procesado
                bloques_originales.append(bloque_original)
                bloques_procesados.append(bloque_procesado.copy())
                actualizar_visualizacion(bloque_original, bloque_procesado)

            stream_tiempo_real = sd.Stream(
                samplerate=fs,
                channels=1,
                dtype="float32",
                blocksize=1024,
                latency="low",
                callback=callback,
            )
            stream_tiempo_real.start()

            ruta_var.set("Microfono en tiempo real")
            tiempo_var.set("Coste computacional: tiempo real")
            estado_var.set(f"Grabando con efecto: {nombre_del_efecto}")
            boton_iniciar_rt.state(["disabled"])
            boton_detener_rt.state(["!disabled"])
        except Exception as error:
            stream_tiempo_real = None
            messagebox.showerror("Error", f"No se pudo iniciar el audio en tiempo real:\n{error}")

    def detener_tiempo_real():
        nonlocal stream_tiempo_real, audio_original, audio_procesado

        if stream_tiempo_real is None:
            messagebox.showwarning("Aviso", "No hay ninguna grabacion activa.")
            return

        try:
            stream_tiempo_real.stop()
            stream_tiempo_real.close()
        finally:
            stream_tiempo_real = None
            boton_iniciar_rt.state(["!disabled"])
            boton_detener_rt.state(["disabled"])

        if bloques_originales and bloques_procesados:
            audio_original = np.concatenate(bloques_originales)
            audio_procesado = np.concatenate(bloques_procesados)
            duracion = len(audio_procesado) / fs
            estado_var.set(f"Grabacion finalizada: {duracion:.2f} s, {fs} Hz")
        else:
            audio_original = None
            audio_procesado = None
            estado_var.set("Grabacion finalizada sin muestras.")

    def cerrar_ventana():
        nonlocal stream_tiempo_real

        if stream_tiempo_real is not None:
            try:
                stream_tiempo_real.stop()
                stream_tiempo_real.close()
            finally:
                stream_tiempo_real = None

        ventana.destroy()

    ventana.protocol("WM_DELETE_WINDOW", cerrar_ventana)

    contenedor = ttk.Frame(ventana, padding=18)
    contenedor.pack(fill="both", expand=True)

    ttk.Label(contenedor, text="EfectosDeVoz", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        contenedor,
        text="Efectos de voz en archivo y microfono",
        style="Status.TLabel",
    ).pack(anchor="w", pady=(0, 12))

    frame_archivo = ttk.LabelFrame(
        contenedor,
        text="Archivo",
        style="Section.TLabelframe",
    )
    frame_archivo.pack(fill="x", pady=(0, 10))

    ttk.Label(frame_archivo, textvariable=ruta_var, wraplength=440).pack(
        anchor="w", pady=(0, 8)
    )
    ttk.Button(
        frame_archivo,
        text="Cargar audio",
        command=cargar_audio_interfaz,
        style="Primary.TButton",
    ).pack(fill="x")

    frame_efecto = ttk.LabelFrame(
        contenedor,
        text="Efecto",
        style="Section.TLabelframe",
    )
    frame_efecto.pack(fill="x", pady=(0, 10))

    ttk.Combobox(
        frame_efecto,
        textvariable=efecto_var,
        values=list(efectos.keys()),
        state="readonly",
    ).pack(fill="x")

    frame_acciones = ttk.Frame(contenedor)
    frame_acciones.pack(fill="x", pady=(0, 10))
    ttk.Button(
        frame_acciones,
        text="Aplicar efecto",
        command=aplicar_efecto_interfaz,
        style="Primary.TButton",
    ).pack(side="left", fill="x", expand=True, padx=(0, 4))
    ttk.Button(
        frame_acciones,
        text="Reproducir",
        command=reproducir_audio_interfaz,
        style="Primary.TButton",
    ).pack(side="left", fill="x", expand=True, padx=4)
    ttk.Button(
        frame_acciones,
        text="Guardar",
        command=guardar_audio_interfaz,
        style="Primary.TButton",
    ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    frame_tiempo_real = ttk.LabelFrame(
        contenedor,
        text="Tiempo real",
        style="Section.TLabelframe",
    )
    frame_tiempo_real.pack(fill="x", pady=(0, 12))
    boton_iniciar_rt = ttk.Button(
        frame_tiempo_real,
        text="Iniciar",
        command=iniciar_tiempo_real,
        style="Primary.TButton",
    )
    boton_iniciar_rt.pack(side="left", fill="x", expand=True, padx=(0, 5))
    boton_detener_rt = ttk.Button(
        frame_tiempo_real,
        text="Detener",
        command=detener_tiempo_real,
        style="Primary.TButton",
    )
    boton_detener_rt.pack(side="left", fill="x", expand=True, padx=(5, 0))
    boton_detener_rt.state(["disabled"])

    frame_ondas = ttk.LabelFrame(
        contenedor,
        text="Ondas",
        style="Section.TLabelframe",
    )
    frame_ondas.pack(fill="x", pady=(0, 12))
    onda_canvas = tk.Canvas(
        frame_ondas,
        height=170,
        bg="#f8fafc",
        highlightthickness=0,
    )
    onda_canvas.pack(fill="x")

    frame_estado = ttk.Frame(contenedor)
    frame_estado.pack(fill="x")
    ttk.Separator(frame_estado).pack(fill="x", pady=(0, 10))
    ttk.Label(frame_estado, textvariable=tiempo_var, style="Status.TLabel").pack(
        anchor="w"
    )
    ttk.Label(
        frame_estado,
        textvariable=estado_var,
        wraplength=455,
        style="Status.TLabel",
    ).pack(anchor="w", pady=(4, 0))

    dibujar_visualizacion()
    ventana.mainloop()


if __name__ == "__main__":
    crear_interfaz()
