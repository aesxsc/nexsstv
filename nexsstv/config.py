import numpy as np


class Config:
    # General
    FS = 44100
    BITS_PER_SYMBOL = 2  # DQPSK
    CP_RATIO = 0.125

    # Transmission controls
    MAX_TX_SECONDS = 60.0
    LEAD_IN_SAMPLES = 2048
    DEFAULT_QUALITY = 40
    MAX_STRIPE_BYTES = 360
    INTERLEAVER_DEPTH = 48
    CODING_RATE = "2/3"
    SYMBOLS_PER_FRAME = 34
    REDUNDANCY_FACTOR = 1.30

    # Image details
    STRIPE_HEIGHT = 12
    NUM_STRIPES = 50
    TARGET_RES = (800, 600)

    # Frequency bands
    NORMAL = {
        "name": "normal",
        "f_min": 500,
        "f_max": 3000,
        "n_subcarriers": 56,
    }

    ULTRA = {
        "name": "ultra",
        "f_min": 13000,
        "f_max": 19000,
        "n_subcarriers": 128,
    }

    @staticmethod
    def get_mode_params(mode="normal"):
        if mode == "classic":
            mode = "normal"
        if mode == "normal":
            return Config.NORMAL
        return Config.ULTRA

    @staticmethod
    def estimate_seconds(mode="normal", frame_count=None):
        params = Config.get_mode_params(mode)
        n_sub = params["n_subcarriers"]
        bits_per_symbol = n_sub * Config.BITS_PER_SYMBOL
        symbol_samples = 1024 + int(1024 * Config.CP_RATIO)
        bits_per_frame = bits_per_symbol * Config.SYMBOLS_PER_FRAME
        frames = frame_count if frame_count is not None else int(np.ceil(Config.NUM_STRIPES * Config.REDUNDANCY_FACTOR))
        sync_samples = 0.03 * Config.FS
        total_samples = Config.LEAD_IN_SAMPLES + int(frames * (sync_samples + symbol_samples * (1 + Config.SYMBOLS_PER_FRAME)))
        return total_samples / Config.FS, bits_per_frame
