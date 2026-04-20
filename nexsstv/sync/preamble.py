import numpy as np
from scipy.signal import chirp, correlate


class Preamble:
    """Chirp + BPSK marker preamble for robust speaker/mic synchronization."""

    BARKER_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.float32)

    def __init__(self, samples_per_bit=20, chirp_duration=0.018):
        self.samples_per_bit = samples_per_bit
        self.chirp_duration = chirp_duration

    def generate_signal(self, fs, f_min, f_max):
        chirp_len = int(self.chirp_duration * fs)
        t = np.linspace(0, self.chirp_duration, chirp_len, endpoint=False)
        sweep = chirp(t, f0=f_min, f1=f_max, t1=self.chirp_duration, method="linear").astype(np.float32)

        marker = np.repeat(self.BARKER_11, self.samples_per_bit)
        mt = np.arange(len(marker), dtype=np.float32) / fs
        center = 0.5 * (f_min + f_max)
        carrier = np.cos(2 * np.pi * center * mt).astype(np.float32)
        bpsk = marker * carrier

        out = np.concatenate([sweep, bpsk]).astype(np.float32)
        out /= np.max(np.abs(out)) + 1e-9
        return out

    @staticmethod
    def detect(received_signal, template_signal):
        if len(received_signal) < len(template_signal):
            return 0, 0.0
        corr = correlate(received_signal, template_signal, mode="valid")
        peak_idx = int(np.argmax(np.abs(corr)))
        segment = received_signal[peak_idx : peak_idx + len(template_signal)]
        denom = (np.linalg.norm(template_signal) * np.linalg.norm(segment)) + 1e-9
        peak = float(np.abs(corr[peak_idx]) / denom)
        return peak_idx, peak
