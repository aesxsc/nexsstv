import numpy as np
from scipy.signal import correlate

class Preamble:
    # Barker-11: 111 - - 1 - - 1 - 1
    BARKER_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])

    def __init__(self, samples_per_bit=16):
        self.samples_per_bit = samples_per_bit
        self.template = np.repeat(self.BARKER_11, samples_per_bit)
        
    def generate_signal(self, fs, f_center=2000):
        t = np.arange(len(self.template)) / fs
        carrier = np.cos(2 * np.pi * f_center * t)
        return self.template * carrier

    @staticmethod
    def detect(received_signal, template_signal):
        if len(received_signal) < len(template_signal): return 0, 0
        corr = correlate(received_signal, template_signal, mode='valid')
        peak_idx = np.argmax(np.abs(corr))
        peak_val = np.abs(corr[peak_idx]) / (np.linalg.norm(template_signal) * np.linalg.norm(received_signal[peak_idx:peak_idx+len(template_signal)]) + 1e-9)
        return peak_idx, peak_val
