import numpy as np

class Config:
    # General
    FS = 44100  # Sample rate
    BITS_PER_SYMBOL = 2  # DQPSK
    CP_RATIO = 0.125
    
    # Frequency Bands
    NORMAL = {
        'f_min': 500,
        'f_max': 3000,
        'n_subcarriers': 56,
    }
    
    ULTRA = {
        'f_min': 13000,
        'f_max': 19000,
        'n_subcarriers': 128,
    }

    @staticmethod
    def get_mode_params(mode='normal'):
        if mode == 'classic':
            mode = 'normal'
        if mode == 'normal':
            return Config.NORMAL
        return Config.ULTRA

    # Image Details
    STRIPE_HEIGHT = 12
    NUM_STRIPES = 50
    TARGET_RES = (800, 600)
    
    # Transmission controls (target: <= 60s)
    DEFAULT_QUALITY = 38
    MAX_STRIPE_BYTES = 292
    INTERLEAVER_DEPTH = 40
    CODING_RATE = "2/3"
    GAP = 0 
    SYMBOLS_PER_STRIPE = 32
