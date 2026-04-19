import numpy as np
from scipy.fftpack import fft, ifft

class Modem:
    def __init__(self, fs, f_min, f_max, n_subcarriers, cp_ratio=0.125):
        self.fs = fs
        self.cp_ratio = cp_ratio
        
        # 1024 FFT for 26ms symbol duration
        self.fft_size = 1024
        self.freq_res = fs / self.fft_size

        min_bin = max(2, int(np.ceil(f_min / self.freq_res)))
        max_bin = min((self.fft_size // 2) - 2, int(np.floor(f_max / self.freq_res)))
        available = max(1, max_bin - min_bin + 1)
        self.n_subcarriers = min(int(n_subcarriers), int(available))
        self.carrier_bins = np.arange(min_bin, min_bin + self.n_subcarriers)
        self.mirror_bins = (-self.carrier_bins) % self.fft_size

        self.last_phases = np.ones(self.n_subcarriers, dtype=np.complex128)

    def dqpsk_map(self, bits):
        mapping = { (0,0): 1+0j, (0,1): 0+1j, (1,1): -1+0j, (1,0): 0-1j }
        symbols = []
        for i in range(0, len(bits), 2):
            chunk = tuple(bits[i:i+2])
            symbols.append(mapping.get(chunk, 1+0j))
        return np.array(symbols)

    def dqpsk_demap(self, diff_symbols):
        bits = []
        for s in diff_symbols:
            angle = np.angle(s)
            if -np.pi/4 <= angle < np.pi/4: bits.extend([0,0])
            elif np.pi/4 <= angle < 3*np.pi/4: bits.extend([0,1])
            elif -3*np.pi/4 <= angle < -np.pi/4: bits.extend([1,0])
            else: bits.extend([1,1])
        return bits

    def modulate_symbol(self, symbols, is_pilot=False):
        if is_pilot:
            self.last_phases = np.ones(self.n_subcarriers, dtype=np.complex128)
            current_phases = self.last_phases
        else:
            if len(symbols) < self.n_subcarriers:
                symbols = np.concatenate([symbols, np.ones(self.n_subcarriers - len(symbols))])
            current_phases = symbols[:self.n_subcarriers] * self.last_phases
            self.last_phases = current_phases

        spectrum = np.zeros(self.fft_size, dtype=np.complex128)
        spectrum[self.carrier_bins] = current_phases
        spectrum[self.mirror_bins] = np.conj(current_phases)
        
        time_data = np.real(ifft(spectrum))
        cp_len = int(self.fft_size * self.cp_ratio)
        return np.concatenate([time_data[-cp_len:], time_data])

    def demodulate_symbol(self, time_data, is_pilot=False):
        cp_len = int(self.fft_size * self.cp_ratio)
        symbol_data = time_data[cp_len : cp_len + self.fft_size]
        spectrum = fft(symbol_data)
        
        current_phases = spectrum[self.carrier_bins]
        
        if is_pilot:
            self.last_phases = current_phases
            return None
        
        diff = current_phases * np.conj(self.last_phases)
        self.last_phases = current_phases
        return diff
