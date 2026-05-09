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
from scipy.signal import resample
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

