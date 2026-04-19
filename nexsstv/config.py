import numpy as np

class Config:
    # General
    FS = 44100  # Sample rate
    BITS_PER_SYMBOL = 2  # DQPSK
    CP_RATIO = 0.125
    
    # Frequency Bands
    CLASSIC = {
        'f_min': 1000,
        'f_max': 4000,
        'n_subcarriers': 128,
    }
    
    ULTRA = {
        'f_min': 14000,
        'f_max': 19000, # 5kHz safe-zone
        'n_subcarriers': 128, # Lower density for 1024 FFT speed
    }

    @staticmethod
    def get_mode_params(mode='classic'):
        if mode == 'classic': return Config.CLASSIC
        return Config.ULTRA

    # Image Details
    STRIPE_HEIGHT = 8
    NUM_STRIPES = 75
    TARGET_RES = (800, 600)
    
    # V10 Timing: Optimized for ~30 seconds
    GAP = 0 
    SYMBOLS_PER_STRIPE = 12 # 12 symbols * 256 bits = 384 bytes + header space
