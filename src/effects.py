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


class PhaseVocoderPitchShift:
    """Pitch shifter por phase vocoder preparado para bloques en tiempo real."""

    def __init__(self, pitch_factor=1.5, n_fft=1024, hop_length=256):
        self.pitch_factor = pitch_factor
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bin_count = (n_fft // 2) + 1

        self.window = np.hanning(n_fft).astype(np.float32)
        self.expected_phase = (
            2 * np.pi * hop_length * np.arange(self.bin_count) / n_fft
        )
        self.prev_phase = np.zeros(self.bin_count, dtype=np.float32)
        self.output_phase = np.zeros(self.bin_count, dtype=np.float32)

        self.input_buffer = np.zeros(n_fft, dtype=np.float32)
        self.output_buffer = np.zeros(n_fft, dtype=np.float32)
        self.ola_gain = hop_length / np.sum(self.window ** 2)

    def process(self, block):
        """Procesa un bloque y devuelve otro bloque con la misma longitud."""
        block = np.asarray(block, dtype=np.float32)
        original_length = len(block)

        if original_length == 0:
            return block

        resto = original_length % self.hop_length
        if resto:
            padding = self.hop_length - resto
            block = np.pad(block, (0, padding))

        output_chunks = []
        for start in range(0, len(block), self.hop_length):
            chunk = block[start:start + self.hop_length]

            self.input_buffer[:-self.hop_length] = self.input_buffer[self.hop_length:]
            self.input_buffer[-self.hop_length:] = chunk

            frame = self.input_buffer * self.window
            shifted_frame = self._process_frame(frame)

            self.output_buffer += shifted_frame * self.window * self.ola_gain
            output_chunks.append(self.output_buffer[:self.hop_length].copy())

            self.output_buffer[:-self.hop_length] = self.output_buffer[self.hop_length:]
            self.output_buffer[-self.hop_length:] = 0.0

        return np.concatenate(output_chunks)[:original_length].astype(np.float32)

    def _process_frame(self, frame):
        spectrum = np.fft.rfft(frame)
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        phase_delta = phases - self.prev_phase - self.expected_phase
        phase_delta = (phase_delta + np.pi) % (2 * np.pi) - np.pi
        true_phase_increment = self.expected_phase + phase_delta
        self.prev_phase = phases

        output_magnitudes = np.zeros(self.bin_count, dtype=np.float32)
        output_increments = np.zeros(self.bin_count, dtype=np.float32)
        output_weights = np.zeros(self.bin_count, dtype=np.float32)

        for source_bin in range(1, self.bin_count - 1):
            target_bin = source_bin * self.pitch_factor
            lower_bin = int(np.floor(target_bin))
            fraction = target_bin - lower_bin
            increment = true_phase_increment[source_bin] * self.pitch_factor

            if 0 < lower_bin < self.bin_count:
                weight = 1.0 - fraction
                output_magnitudes[lower_bin] += magnitudes[source_bin] * weight
                output_increments[lower_bin] += increment * weight
                output_weights[lower_bin] += weight

            upper_bin = lower_bin + 1
            if 0 < upper_bin < self.bin_count and fraction > 0:
                output_magnitudes[upper_bin] += magnitudes[source_bin] * fraction
                output_increments[upper_bin] += increment * fraction
                output_weights[upper_bin] += fraction

        active_bins = output_weights > 0
        self.output_phase[active_bins] += (
            output_increments[active_bins] / output_weights[active_bins]
        )

        output_spectrum = np.zeros(self.bin_count, dtype=np.complex64)
        output_spectrum[active_bins] = output_magnitudes[active_bins] * np.exp(
            1j * self.output_phase[active_bins]
        )

        return np.fft.irfft(output_spectrum, self.n_fft).astype(np.float32)


def AlvinVocoder(signal, Fs=None, pitch_factor=1.5, n_fft=1024, hop_length=256):
    """Efecto Alvin con cambio de tono mediante phase vocoder."""
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) == 0:
        return signal

    vocoder = PhaseVocoderPitchShift(
        pitch_factor=pitch_factor,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    padded_signal = np.pad(signal, (n_fft // 2, n_fft // 2))
    output = np.zeros_like(padded_signal, dtype=np.float32)
    normalization = np.zeros_like(padded_signal, dtype=np.float32)

    for start in range(0, len(padded_signal) - n_fft + 1, hop_length):
        frame = padded_signal[start:start + n_fft] * vocoder.window
        shifted_frame = vocoder._process_frame(frame)
        output[start:start + n_fft] += shifted_frame * vocoder.window
        normalization[start:start + n_fft] += vocoder.window ** 2

    valid = normalization > 1e-6
    output[valid] /= normalization[valid]
    output = output[n_fft // 2:n_fft // 2 + len(signal)]

    return np.clip(output, -1.0, 1.0).astype(np.float32)


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
