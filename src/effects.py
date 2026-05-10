# -*- coding: utf-8 -*-

''' 
Biblioteca de funciones de efectos

Efectos implementados 
=====================
    -Alvin
    -Reverb
    -distorsión
    -Delay
    -Eco
    -Radio
    

'''
import numpy as np
from scipy.signal import resample ,butter, lfilter
import soundfile as sf

def reverb(signal, Fs, delay, decay):
    """
    Añade un efecto de reverberación a la señal

    Parametros
    ----------
    signal : array_like
        Señal de entrada
    Fs : num
        Frecuencia de muestreo de la señal
    Delay:
        Ajuste del delay de la señal
    Decay
        Ajuste del decay del delay de la señal

    Returns
    -------
    arr : ndarray
        Señal con el efecto aplicado
    """

    delay_samples = int(delay * Fs)

    y = np.copy(signal).astype(np.float32)

    for i in range(delay_samples, len(signal)):
        y[i] += decay * y[i - delay_samples]

    return y

def Alvin(signal, speedup=1.5):
    """ Implementa un efecto "estilo Alvin y las ardillas modificando la cantidad de muestras de la señal
    Parametros
    ----------
    signal : array_like
            Señal de entrada.

    Speedup : Aceleración de la señal

    Returns
    -------
    arr : ndarray
        Señal con el efecto aplicado
    """
    N = int(len(signal) / speedup)
    shifted =resample(signal, N)
    return shifted

def distortion(signal, gain= 2.0 , threshold= 0.8):

    """ Implementa un efecto de distorsión amplificando la señal y filtrando a partir de un umbral
    signal : array_like
            Señal de entrada.

    Gain : Ganancia de la señal
 
    Threshold : Umbral de filtrado

    Returns
    -------
    arr : ndarray
        Señal con el efecto aplicado
    """

    y = signal * gain
    return np.clip(y,-threshold,threshold)


def eco(signal, Fs, delay_sec=0.5, attenuation=0.6):
    """
    Añade un efecto de eco simple a la señal mediante un retardo temporal.
    Cumple con el objetivo de retardos temporales con memoria.
    """
    delay_samples = int(delay_sec * Fs)
    # Creamos un array más largo para que la "cola" del eco no se corte
    y = np.zeros(len(signal) + delay_samples, dtype=np.float32)
    y[:len(signal)] += signal
    y[delay_samples:] += signal * attenuation
    return y

def radio(signal_in, Fs, lowcut=300.0, highcut=3000.0, order=5):
    """
    Aplica un filtro IIR pasa-banda para simular una radio antigua o megáfono.
    Recorta los graves y los agudos extremos de la voz.
    """
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal_in)
    return y

def robot(signal_in, Fs, freq_mod=40.0):
    """
    Genera un efecto de voz metálica/robótica mediante Modulación en Amplitud (AM).
    Multiplica la señal por una onda portadora de baja frecuencia.
    """
    t = np.arange(len(signal_in)) / Fs
    # Señal portadora (sinusoide oscilando a freq_mod)
    portadora = np.sin(2 * np.pi * freq_mod * t)
    y = signal_in * portadora
    return y
